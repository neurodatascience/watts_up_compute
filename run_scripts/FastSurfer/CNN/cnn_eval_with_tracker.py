
# This is modified from https://github.com/Deep-MI/FastSurfer/blob/master/FastSurferCNN/eval.py
# Author: Nikhil Bhagwat
# Date: 20 July 2021


# IMPORTS
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
from datetime import datetime
import time
import sys
import glob
import os.path as op
import logging
import torch
import torch.nn as nn
# import torch.nn.utils.prune as prune
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, utils

from scipy.ndimage.filters import median_filter, gaussian_filter
from skimage.measure import label, regionprops
from skimage.measure import label

from collections import OrderedDict
from os import makedirs

# Add paths (singularity should see these)
# FastSurfer and carbon trackers are in the mounted dir as these repos keep getting updated.
# TODO replace this with setup.py once the dependencis become stable
#sys.path.append('../../../../FastSurfer/')
#sys.path.append('../../../../experiment-impact-tracker/')
#sys.path.append('../../../../codecarbon/')

from FastSurferCNN.data_loader.load_neuroimaging_data import OrigDataThickSlices
from FastSurferCNN.data_loader.load_neuroimaging_data import map_label2aparc_aseg
from FastSurferCNN.data_loader.load_neuroimaging_data import map_prediction_sagittal2full
from FastSurferCNN.data_loader.load_neuroimaging_data import get_largest_cc
from FastSurferCNN.data_loader.load_neuroimaging_data import load_and_conform_image
from FastSurferCNN.data_loader.augmentation import ToTensorTest

from FastSurferCNN.models.networks import FastSurferCNN

# Model arch costs
from ptflops import get_model_complexity_info
from pypapi import events, papi_high as high

# Carbon costs
from experiment_impact_tracker.compute_tracker import ImpactTracker
from codecarbon import OfflineEmissionsTracker, EmissionsTracker
from carbontracker.tracker import CarbonTracker
from carbontracker import parser


HELPTEXT = """
Script to run FastSurferCNN module
"""

def options_parse():
    """
    Command line option parser
    """
    parser = argparse.ArgumentParser(description=HELPTEXT, epilog='$Id: fast_surfer_cnn, v 1.0 2019/09/30$')

    # 1. Directory information (where to read from, where to write to)
    parser.add_argument('--i_dir', '--input_directory', dest='input', help='path to directory of input volume(s).')
    parser.add_argument('--csv_file', '--csv_file', help="CSV-file with directories to process", default=None)
    parser.add_argument('--o_dir', '--output_directory', dest='output',
                        help='path to output directory. Will be created if it does not already exist')

    # 2. Options for the MRI volumes (name of in and output, order of interpolation if not conformed)
    parser.add_argument('--in_name', '--input_name', dest='iname', help='name of file to process. Default: orig.mgz',
                        default='orig.mgz')
    parser.add_argument('--out_name', '--output_name', dest='oname', default='aparc.DKTatlas+aseg.deep.mgz',
                        help='name under which segmentation will be saved. Default: aparc.DKTatlas+aseg.deep.mgz. '
                             'If a separate subfolder is desired (e.g. FS conform, add it to the name: '
                             'mri/aparc.DKTatlas+aseg.deep.mgz)')
    parser.add_argument('--order', dest='order', type=int, default=1,
                        help="order of interpolation (0=nearest,1=linear(default),2=quadratic,3=cubic)")

    # 3. Options for log-file and search-tag
    parser.add_argument('--t', '--tag', dest='search_tag', default="*",
                        help='Search tag to process only certain subjects. If a single image should be analyzed, '
                             'set the tag with its id. Default: processes all.')
    parser.add_argument('--log', dest='logfile', help='name of log-file. Default: deep-seg.log',
                        default='deep-seg.log')

    # 4. Pre-trained weights
    parser.add_argument('--network_sagittal_path', dest='network_sagittal_path',
                        help="path to pre-trained weights of sagittal network",
                        default='./checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl')
    parser.add_argument('--network_coronal_path', dest='network_coronal_path',
                        help="pre-trained weights of coronal network",
                        default='./checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl')
    parser.add_argument('--network_axial_path', dest='network_axial_path',
                        help="pre-trained weights of axial network",
                        default='./checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl')
    # Prune options                    
    parser.add_argument('--prune_type', dest='prune_type',
                        help="prune type: Global or layerwise sparsity",
                        default=None)
    parser.add_argument('--prune_percent', dest='prune_percent',
                        help="desired sparsity",
                        type=float, default=0.2)

    parser.add_argument('--mock_run', dest='mock_run',
                        help="run without inference: 1, run only Axial model: 2",
                        type=int, default=1)

    # Tracker options
    parser.add_argument('--geo_loc', dest='geo_loc',
                        help="(lat,log) coords for experiment impact tracker",
                        type=str, default='45.4972159,-73.6103642') #MTL Beluga
    parser.add_argument('--tracker_log_dir', dest='tracker_log_dir',
                        help="log dir for experiment impact tracker",
                        type=str, default='./tracker_logs/')
    parser.add_argument('--CC_offline',
                        help="Run CC in offline mode",
                        action='store_true')                 
    parser.add_argument('--TZ', dest='TZ',
                        help="TimeZone",
                        type=str, default='America/New_York')
    parser.add_argument('--iso_code', dest='iso_code',
                        help="Country ISO code",
                        type=str, default='USA')

    # PAPI
    parser.add_argument('--count_FLOPs', dest='count_FLOPs',
                    help="Count FLOPs using PAPI",
                    action='store_true') 

    # 5. Options for model parameters setup (only change if model training was changed)
    parser.add_argument('--num_filters', type=int, default=64,
                        help='Filter dimensions for DenseNet (all layers same). Default=64')
    parser.add_argument('--num_classes_ax_cor', type=int, default=79,
                        help='Number of classes to predict in axial and coronal net, including background. Default=79')
    parser.add_argument('--num_classes_sag', type=int, default=51,
                        help='Number of classes to predict in sagittal net, including background. Default=51')
    parser.add_argument('--num_channels', type=int, default=7,
                        help='Number of input channels. Default=7 (thick slices)')
    parser.add_argument('--kernel_height', type=int, default=5, help='Height of Kernel (Default 5)')
    parser.add_argument('--kernel_width', type=int, default=5, help='Width of Kernel (Default 5)')
    parser.add_argument('--stride', type=int, default=1, help="Stride during convolution (Default 1)")
    parser.add_argument('--stride_pool', type=int, default=2, help="Stride during pooling (Default 2)")
    parser.add_argument('--pool', type=int, default=2, help='Size of pooling filter (Default 2)')

    # 6. Clean up and GPU/CPU options (disable cuda, change batchsize)
    parser.add_argument('--clean', dest='cleanup', help="Flag to clean up segmentation", action='store_true')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for inference. Default: 8")
    parser.add_argument('--simple_run', action='store_true', default=False,
                        help='Simplified run: only analyse one given image specified by --in_name (output: --out_name). '
                             'Need to specify absolute path to both --in_name and --out_name if this option is chosen.')

    sel_option = parser.parse_args()

    if sel_option.input is None and sel_option.csv_file is None and not sel_option.simple_run:
        parser.print_help(sys.stderr)
        sys.exit('----------------------------\nERROR: Please specify data directory or input volume\n')

    if sel_option.output is None and not sel_option.simple_run:
        parser.print_help(sys.stderr)
        sys.exit('----------------------------\nERROR: Please specify data output directory '
                 '(can be same as input directory)\n')

    return sel_option


### Prune module addition
def prune_model(model, prune_type, prune_percent):
    ''' Sparsifies (L1) model weights with either global or layerwise prune_percent. Currently only pruning Conv2D.
    '''
    if prune_type == 'global':
        print('Globally pruning all Conv2d layers with {} sparsity'.format(prune_percent))
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                parameters_to_prune.append((module,'weight'))
        
        prune.global_unstructured(tuple(parameters_to_prune), pruning_method=prune.L1Unstructured, amount=prune_percent)

    elif prune_type == 'layerwise':
        print('Layerwise pruning all Conv2d layers with {} sparsity'.format(prune_percent))
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=prune_percent)

    else:
        print('Unknown pruning method: {}'.format(prune_type))

    # make pruning permenant
    # otherwise subsequent Coronal and Sagittal model calls fail due to weight name mismatch
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')
    return model
###


def run_network(img_filename, orig_data, prediction_probability, plane, ckpts, params_model, model, logger):
    """
    Inference run for single network on a given image.

    :param str img_filename: name of image file
    :param np.ndarray orig_data: image data
    :param torch.tensor prediction_probability: default tensor to hold prediction probabilities
    :param str plane: Which plane to predict (Axial, Sagittal, Coronal)
    :param str ckpts: Path to pretrained weights of network
    :param dict params_model: parameters to set up model (includes device, use_cuda, model_parallel, batch_size)
    :param torch.nn.Module model: Model to use for prediction
    :param logging.logger logger: Logging instance info messages will be written to
    :return:
    """
    # Set up DataLoader
    test_dataset = OrigDataThickSlices(img_filename, orig_data, plane=plane,
                                       transforms=transforms.Compose([ToTensorTest()]))

    test_data_loader = DataLoader(dataset=test_dataset, shuffle=False,
                                  batch_size=params_model["batch_size"])

    # Set up state dict for model
    logger.info("Loading {} Net from {}".format(plane, ckpts))

    model_state = torch.load(ckpts, map_location=params_model["device"])

    new_state_dict = OrderedDict()

    for k, v in model_state["model_state_dict"].items():

        if k[:7] == "module." and not params_model["model_parallel"]:
            new_state_dict[k[7:]] = v

        elif k[:7] != "module." and params_model["model_parallel"]:
            new_state_dict["module." + k] = v

        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    model.eval()

    ### Prune module addition
    if params_model['prune_type'] is not None:
        model = prune_model(model, params_model['prune_type'], params_model['prune_percent'])
        n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_zero_params = []
        n_conv_params = []
        logger.info('\nPruning parameters...')
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                zero_conv_params = float(torch.sum(module.weight == 0))
                total_conv_params = float(module.weight.nelement())
                logger.info("Sparsity in {}: {:.2f}%".format(name, 100. * zero_conv_params /total_conv_params))
                n_zero_params.append(zero_conv_params)
                n_conv_params.append(total_conv_params)

        logger.info('\nTotal params in the model, trainable:{}, conv:{}, ({:3.2f}), zero:{}, ({:3.2f})'.format(n_trainable_params,np.sum(n_conv_params),
        100*np.sum(n_conv_params)/n_trainable_params,np.sum(n_zero_params),100*np.sum(n_zero_params)/n_trainable_params))

    logger.info("{} model loaded.".format(plane))
    with torch.no_grad():

        start_index = 0
        for batch_idx, sample_batch in enumerate(test_data_loader):

            images_batch = Variable(sample_batch["image"])

            if params_model["use_cuda"]:
                images_batch = images_batch.cuda()

            temp = model(images_batch)

            if plane == "Axial":
                temp = temp.permute(3, 0, 2, 1)
                prediction_probability[:, start_index:start_index + temp.shape[1], :, :] += torch.mul(temp.cpu(), 0.4)
                start_index += temp.shape[1]

            elif plane == "Coronal":
                temp = temp.permute(2, 3, 0, 1)
                prediction_probability[:, :, start_index:start_index + temp.shape[2], :] += torch.mul(temp.cpu(), 0.4)
                start_index += temp.shape[2]

            else:
                temp = map_prediction_sagittal2full(temp).permute(0, 3, 2, 1)
                prediction_probability[start_index:start_index + temp.shape[0], :, :, :] += torch.mul(temp.cpu(), 0.2)
                start_index += temp.shape[0]

            logger.info("--->Batch {} {} Testing Done.".format(batch_idx, plane))

    return prediction_probability


def fastsurfercnn(img_filename, save_as, logger, args):
    """
    Cortical parcellation of single image with FastSurferCNN.

    :param str img_filename: name of image file
    :param parser.Argparse args: Arguments (passed via command line) to set up networks
            * args.network_sagittal_path: path to sagittal checkpoint (stored pretrained network)
            * args.network_coronal_path: path to coronal checkpoint (stored pretrained network)
            * args.network_axial_path: path to axial checkpoint (stored pretrained network)
            * args.cleanup: Whether to clean up the segmentation (medial filter on certain labels)
            * args.no_cuda: Whether to use CUDA (GPU) or not (CPU)
            * args.batch_size: Input batch size for inference (Default=8)
            * args.num_classes_ax_cor: Number of classes to predict in axial/coronal net (Default=79)
            * args.num_classes_sag: Number of classes to predict in sagittal net (Default=51)
            * args.num_channels: Number of input channels (Default=7, thick slices)
            * args.num_filters: Number of filter dimensions for DenseNet (Default=64)
            * args.kernel_height and args.kernel_width: Height and width of Kernel (Default=5)
            * args.stride: Stride during convolution (Default=1)
            * args.stride_pool: Stride during pooling (Default=2)
            * args.pool: Size of pooling filter (Default=2)
    :param logging.logger logger: Logging instance info messages will be written to
    :param str save_as: name under which to save prediction.

    :return None: saves prediction to save_as
    """
    start_total = time.time()

    mock_run = args.mock_run
    if mock_run != 0:
        print('\n********Doing a mock run with level: {} (1: No inference, 2:Axial, 3:Coronal, 4:Sagittal********\n'.format(mock_run))

    # # PAPI
    # papi_csv = '{}/{}/compute_costs_flop.csv'.format(args.tracker_log_dir,args.search_tag)
    # papi_df = pd.DataFrame(columns=['task','start_time','duration','DP'])
    # high.start_counters([events.PAPI_DP_OPS,]) #default: PAPI_FP_OPS

    # setup
    start_time = time.time()
    
    logger.info("Reading volume {}".format(img_filename))
    header_info, affine_info, orig_data = load_and_conform_image(img_filename, interpol=1, logger=logger)

    # Set up model for axial and coronal networks
    params_network = {'num_channels': args.num_channels, 'num_filters': args.num_filters,
                      'kernel_h': args.kernel_height, 'kernel_w': args.kernel_width,
                      'stride_conv': args.stride, 'pool': args.pool,
                      'stride_pool': args.stride_pool, 'num_classes': args.num_classes_ax_cor,
                      'kernel_c': 1, 'kernel_d': 1}

    # Select the model
    model = FastSurferCNN(params_network)

    # Put it onto the GPU or CPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info("Cuda available: {}, # Available GPUS: {}, "
                "Cuda user disabled (--no_cuda flag): {}, "
                "--> Using device: {}".format(torch.cuda.is_available(),
                                              torch.cuda.device_count(),
                                              args.no_cuda, device))

    if use_cuda and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model_parallel = True
    else:
        model_parallel = False

    model.to(device)

    ### Prune module addition
    prune_type = args.prune_type
    prune_percent = float(args.prune_percent)

    logger.info("prune params, type:{}, percent:{}".format(prune_type,prune_percent))
    params_model = {'device': device, "use_cuda": use_cuda, "batch_size": args.batch_size,
                    "model_parallel": model_parallel, 
                    'prune_type': prune_type, 'prune_percent':prune_percent}
    ###

    # Set up tensor to hold probabilities
    pred_prob = torch.zeros((256, 256, 256, args.num_classes_ax_cor), dtype=torch.float)

    end_time = time.time()
    setup_time = end_time - start_time

    # DP = high.stop_counters()[0]
    # papi_df.loc[0] = ['setup', start_time, setup_time, DP]

    # MAC counters (ignoring this part from FLOP counter)
    n_channels = args.num_channels
    input_size = 64
    macs, params = get_model_complexity_info(model, (n_channels, input_size, input_size), as_strings=False,
                                        print_per_layer_stat=False)


    logger.info("MACs:{} params: {}".format(macs,params))


    # Axial Prediction #trainable:1799206 
    # high.start_counters([events.PAPI_DP_OPS,]) #default: PAPI_FP_OPS
    start_time = time.time()

    if mock_run in [0, 2]:        
        pred_prob = run_network(img_filename,
                                orig_data, pred_prob, "Axial",
                                args.network_axial_path,
                                params_model, model, logger)

        print("Axial View Tested in {:0.4f} seconds".format(time.time() - start_time))
    
    end_time = time.time()
    axial_time = end_time - start_time

    # DP = high.stop_counters()[0]
    # papi_df.loc[1] = ['axial', start_time, axial_time, DP]

    # Coronal Prediction #trainable:1799206
    # high.start_counters([events.PAPI_DP_OPS,]) #default: PAPI_FP_OPS
    start_time = time.time()

    if mock_run in [0, 3]:
        pred_prob = run_network(img_filename,
                                orig_data, pred_prob, "Coronal",
                                args.network_coronal_path,
                                params_model, model, logger)

        logger.info("Coronal View Tested in {:0.4f} seconds".format(time.time() - start_time))

    end_time = time.time()
    coronal_time = end_time - start_time

    # DP = high.stop_counters()[0]
    # papi_df.loc[2] = ['Coronal', start_time, coronal_time, DP]

    # Sagittal Prediction #trainable:1797386
    # high.start_counters([events.PAPI_DP_OPS,]) #default: PAPI_FP_OPS
    start_time = time.time()

    if mock_run in [0, 4]:
        params_network["num_classes"] = args.num_classes_sag
        params_network["num_channels"] = args.num_channels

        model = FastSurferCNN(params_network)

        if model_parallel:
            model = nn.DataParallel(model)

        model.to(device)

        pred_prob = run_network(img_filename, orig_data, pred_prob, "Sagittal",
                                args.network_sagittal_path,
                                params_model, model, logger)

        logger.info("Sagittal View Tested in {:0.4f} seconds".format(time.time() - start_time))
            
    end_time = time.time()
    sagittal_time = end_time - start_time

    # DP = high.stop_counters()[0]
    # papi_df.loc[3] = ['Sagittal', start_time, sagittal_time, DP]

    # Aggregation and postprocessing:
    # high.start_counters([events.PAPI_DP_OPS,]) #default: PAPI_FP_OPS
    
    start_time  = time.time()

    if mock_run != 1:
        # Get predictions and map to freesurfer label space
        _, pred_prob = torch.max(pred_prob, 3)
        pred_prob = pred_prob.numpy()
        pred_prob = map_label2aparc_aseg(pred_prob)

        # Post processing - Splitting classes
        # Quick Fix for 2026 vs 1026; 2029 vs. 1029; 2025 vs. 1025
        rh_wm = get_largest_cc(pred_prob == 41)
        lh_wm = get_largest_cc(pred_prob == 2)
        rh_wm = regionprops(label(rh_wm, background=0))
        lh_wm = regionprops(label(lh_wm, background=0))
        centroid_rh = np.asarray(rh_wm[0].centroid)
        centroid_lh = np.asarray(lh_wm[0].centroid)

        labels_list = np.array([1003, 1006, 1007, 1008, 1009, 1011,
                                1015, 1018, 1019, 1020, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1034, 1035])

        for label_current in labels_list:

            label_img = label(pred_prob == label_current, connectivity=3, background=0)

            for region in regionprops(label_img):

                if region.label != 0:  # To avoid background

                    if np.linalg.norm(np.asarray(region.centroid) - centroid_rh) < np.linalg.norm(
                            np.asarray(region.centroid) - centroid_lh):
                        mask = label_img == region.label
                        pred_prob[mask] = label_current + 1000

        # Quick Fixes for overlapping classes
        aseg_lh = gaussian_filter(1000 * np.asarray(pred_prob == 2, dtype=np.float), sigma=3)
        aseg_rh = gaussian_filter(1000 * np.asarray(pred_prob == 41, dtype=np.float), sigma=3)

        lh_rh_split = np.argmax(np.concatenate((np.expand_dims(aseg_lh, axis=3), np.expand_dims(aseg_rh, axis=3)), axis=3),
                                axis=3)

        # Problematic classes: 1026, 1011, 1029, 1019
        for prob_class_lh in [1011, 1019, 1026, 1029]:
            prob_class_rh = prob_class_lh + 1000
            mask_lh = ((pred_prob == prob_class_lh) | (pred_prob == prob_class_rh)) & (lh_rh_split == 0)
            mask_rh = ((pred_prob == prob_class_lh) | (pred_prob == prob_class_rh)) & (lh_rh_split == 1)

            pred_prob[mask_lh] = prob_class_lh
            pred_prob[mask_rh] = prob_class_rh
 
        # Clean-Up
        if args.cleanup is True:
            start_time = time.time()
            labels = [2, 4, 5, 7, 8, 10, 11, 12, 13, 14,
                    15, 16, 17, 18, 24, 26, 28, 31, 41, 43, 44,
                    46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 63,
                    77, 1026, 2026]

            pred_prob_medfilt = median_filter(pred_prob, size=(3, 3, 3))
            mask = np.zeros_like(pred_prob)
            tolerance = 25

            for current_label in labels:
                current_class = (pred_prob == current_label)
                label_image = label(current_class, connectivity=3)

                for region in regionprops(label_image):

                    if region.area <= tolerance:
                        mask_label = (label_image == region.label)
                        mask[mask_label] = 1

            pred_prob[mask == 1] = pred_prob_medfilt[mask == 1]
            # logger.info("Segmentation Cleaned up in {:0.4f} seconds.".format(time.time() - start_time)
            print("Segmentation Cleaned up in {:0.4f} seconds.".format(time.time() - start_time))

        # Saving image
        header_info.set_data_dtype(np.int16)
        mapped_aseg_img = nib.MGHImage(pred_prob, affine_info, header_info)
        mapped_aseg_img.to_filename(save_as)
        logger.info("Saving Segmentation to {}".format(save_as))
        logger.info("Total processing time: {:0.4f} seconds.".format(time.time() - start_total))
    
    
    end_time = time.time()
    agg_time = end_time - start_time

    # DP = high.stop_counters()[0]
    # papi_df.loc[4] = ['aggregate', start_time, agg_time, DP]
    # papi_df['MAC'] = macs
    # papi_df['params'] = params
    # papi_df.to_csv(papi_csv)


if __name__ == "__main__":
    # Command Line options and error checking done here
    options = options_parse()   

    # Set up the logger
    logger = logging.getLogger("eval")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))

    now = datetime.now()

    # Trackers
    geo_loc = options.geo_loc
    CC_offline = options.CC_offline
    TZ = options.TZ
    iso_code = options.iso_code

    tracker_log_dir = f'{options.tracker_log_dir}/{options.search_tag}/'

    # Tracker logs
    log_dir_EIT = f'{tracker_log_dir}/EIT/'
    log_dir_CC = f'{tracker_log_dir}/CC/'
    log_dir_CT = f'{tracker_log_dir}/CT/'

    for d in [log_dir_EIT,log_dir_CC,log_dir_CT]:
        if not op.exists(d):
            makedirs(d)

     # FLOPs
    count_FLOPs = options.count_FLOPs #TODO implement this in the main function

    # Set HPC location
    ly,lx = float(geo_loc.split(',')[0]), float(geo_loc.split(',')[1])
    coords = (ly,lx)
    
    ## exp-impact-tracker (this exits automatically after script ends)
    tracker_EIT = ImpactTracker(log_dir_EIT,coords)
    tracker_EIT.launch_impact_monitor()

    ## code-carbon tracker
    print(f'Using offline mode for CC tracker: {CC_offline}')
    if CC_offline:
        print(f'Using {TZ} timezone and {iso_code} country iso code')
    
    if CC_offline:
        tracker_CC = OfflineEmissionsTracker(output_dir=log_dir_CC, country_iso_code=iso_code)        
    else:
        tracker_CC = EmissionsTracker(output_dir=log_dir_CC) 
        

    tracker_CC.start()

    ## CarbonTracker (keep epoch=1 for non DL models)
    tracker_CT = CarbonTracker(epochs=1, log_dir=log_dir_CT)
    tracker_CT.epoch_start()
    
    if options.simple_run:
        print('Staring a simple run...')
        # Check if output subject directory exists and create it otherwise
        sub_dir, out = op.split(options.oname)

        if not op.exists(sub_dir):
            makedirs(sub_dir)

        fastsurfercnn(options.iname, options.oname, logger, options)

    else:
        print('Staring a subject list run...')
        # Prepare subject list to be processed
        if options.csv_file is not None:
            with open(options.csv_file, "r") as s_dirs:
                subject_directories = [line.strip() for line in s_dirs.readlines()]

        else:
            search_path = op.join(options.input, options.search_tag)
            subject_directories = glob.glob(search_path)

        # Report number of subjects to be processed and loop over them
        data_set_size = len(subject_directories)
        logger.info("Total Dataset Size is {}".format(data_set_size))

        for current_subject in subject_directories:

            subject = current_subject.split("/")[-1]

            # Define volume to process, log-file and name under which final prediction will be saved
            if options.csv_file:

                dataset = current_subject.split("/")[-2]
                invol = op.join(current_subject, options.iname)
                logfile = op.join(options.output, dataset, subject, options.logfile)
                save_file_name = op.join(options.output, dataset, subject, options.oname)

            else:

                invol = op.join(current_subject, options.iname)
                logfile = op.join(options.output, subject, options.logfile)
                save_file_name = op.join(options.output, subject, options.oname)

            logger.info("Running Fast Surfer on {}".format(subject))

            # Check if output subject directory exists and create it otherwise
            sub_dir, out = op.split(save_file_name)

            if not op.exists(sub_dir):
                makedirs(sub_dir)

            # Prepare the log-file (logging to File in subject directory)
            fh = logging.FileHandler(logfile, mode='w')
            logger.addHandler(fh)

            # Run network
            fastsurfercnn(invol, save_file_name, logger, options)

            logger.removeHandler(fh)
            fh.close()

            # Check experiment tracker status
            # Optional. Adding this will ensure that your experiment stops if impact tracker throws an exception and exit.
            tracker_EIT.get_latest_info_and_check_for_errors()

        ## code-carbon tracker
        tracker_CC.stop()
        
        ## CarbonTracker
        tracker_CT.epoch_end()

        print(f"Please find your experiment logs in: {tracker_log_dir}")

        sys.exit(0)
