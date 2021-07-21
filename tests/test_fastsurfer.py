
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
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms, utils

from scipy.ndimage.filters import median_filter, gaussian_filter
from skimage.measure import label, regionprops
from skimage.measure import label

from collections import OrderedDict
from os import makedirs

# FastSurfer repo
from FastSurferCNN.data_loader.load_neuroimaging_data import OrigDataThickSlices
from FastSurferCNN.data_loader.load_neuroimaging_data import map_label2aparc_aseg
from FastSurferCNN.data_loader.load_neuroimaging_data import map_prediction_sagittal2full
from FastSurferCNN.data_loader.load_neuroimaging_data import get_largest_cc
from FastSurferCNN.data_loader.load_neuroimaging_data import load_and_conform_image
from FastSurferCNN.data_loader.augmentation import ToTensorTest

from FastSurferCNN.models.networks import FastSurferCNN

# Carbon costs
from experiment_impact_tracker.compute_tracker import ImpactTracker


HELPTEXT = """
Script to run FastSurferCNN module
"""

def options_parse():
    """
    Command line option parser
    """
    parser = argparse.ArgumentParser()

    # Options for model parameters setup (only change if model training was changed)
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

    sel_option = parser.parse_args()

    return sel_option

  
def load_pretrained(pretrained_ckpt, params_model, model):
    model_state = torch.load(pretrained_ckpt, map_location=params_model["device"])
    new_state_dict = OrderedDict()

    # FastSurfer model specific configs
    for k, v in model_state["model_state_dict"].items():

        if k[:7] == "module." and not params_model["model_parallel"]:
            new_state_dict[k[7:]] = v

        elif k[:7] != "module." and params_model["model_parallel"]:
            new_state_dict["module." + k] = v

        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)

    model.eval()
    
    return model


if __name__ == "__main__":

    args = options_parse() 

    plane = "Axial"
    FastSurfer_dir = '/home/nikhil/projects/green_comp_neuro/FastSurfer/'
    pretrained_ckpt = f'{FastSurfer_dir}/checkpoints/{plane}_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl'

    # Put it onto the GPU or CPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Set up model for axial and coronal networks
    params_model = {'num_channels': args.num_channels, 'num_filters': args.num_filters,
                      'kernel_h': args.kernel_height, 'kernel_w': args.kernel_width,
                      'stride_conv': args.stride, 'pool': args.pool,
                      'stride_pool': args.stride_pool, 'num_classes': args.num_classes_ax_cor,
                      'kernel_c': 1, 'kernel_d': 1,
                      'model_parallel': False,
                      'device': device
                      }

    # Select the model
    model = FastSurferCNN(params_model)
    model.to(device)
 
    # Load pretrained weights
    model = load_pretrained(pretrained_ckpt, params_model, model)

    print(f'{plane} model loaded successfully!')