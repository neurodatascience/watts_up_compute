import os
import sys
import numpy as np
import pandas as pd
import pickle
import torch 
import torchvision.models as models

sys.path.append('../')
from ml.loss import DiceLoss
from ml.data import *
from ml.model import *

from medpy.filter.binary import largest_connected_component

model_name_size_dict = {'unet':32,'unet_xlarge':128,'unet_large':64,'unet_medium':16,'unet_small':8,'unet_tiny':4}

def dsc(y_pred, y_true, lcc=True):
    if lcc and np.any(y_pred):
        y_pred = np.round(y_pred).astype(int)
        y_true = np.round(y_true).astype(int)
        # y_pred = largest_connected_component(y_pred)
    return np.sum(y_pred[y_true == 1]) * 2.0 / (np.sum(y_pred) + np.sum(y_true))

def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    dsc_list = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index : index + num_slices[p]])
        y_true = np.array(validation_true[index : index + num_slices[p]])
        dsc_list.append(dsc(y_pred, y_true))
        index += num_slices[p]
    return dsc_list

def get_model_perfs(dataset_name, root_data_dir, results_dir, experiment_list, loss_type, test_cases=10, batch_size=4):

    data_path = '{}{}'.format(root_data_dir,dataset_name)

    _, valid_loader, _ = get_kaggle_dataset(data_path, test_cases=test_cases, batch_size=batch_size, subset='valid')
    _, _, test_loader = get_kaggle_dataset(data_path, test_cases=test_cases, batch_size=batch_size, subset='test')

    for exp_name in experiment_list:
        device = 'cuda'
        exp_csv = '{}{}/{}/experiment_config.csv'.format(results_dir, exp_name, device)
        exp_df = pd.read_csv(exp_csv)
        test_perf = exp_df['test_perf'].values[0]
        model_name = exp_df['model'].values[0]
        model_path = '{}{}/{}/{}.pth'.format(results_dir, exp_name, device, model_name)
        print('\nExperiment: {}, model path: {}'.format(exp_name, model_path))
        
        if model_name == 'ResNet_1':
            model = ResNet_1(device, ResidualBlock, [2, 2, 2]).to(device)
        elif model_name in model_name_size_dict.keys():
            model_size = model_name_size_dict[model_name]
            model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                in_channels=3, out_channels=1, init_features=model_size, pretrained=False)

        # Analyzing on a laptop
        device = 'cpu'        
        checkpoint = torch.load(model_path,map_location=torch.device(device))
        model.load_state_dict(checkpoint)
        model.eval()

        loader_dict = {'valid': valid_loader, 'test':test_loader}
        model_perf_dict = {}
        
        for subset, loader in loader_dict.items():
            print('subset: {}'.format(subset))
            if loss_type == 'dice':
                criterion = DiceLoss()
            
            running_loss = []
            volume_perf_list = []
            for i, (images, labels) in enumerate(loader):
                # get the inputs; data is a list of [inputs, labels]
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss.append(loss.item())

                if loss_type == 'dice':
                    dsc_volumes = dsc_per_volume(outputs.detach().numpy(),
                                                    labels.detach().numpy(),
                                                    loader.dataset.patient_slice_index)

                    volume_perf_list.append(dsc_volumes)

            slice_percent_perf = 100 * (1 - np.mean(running_loss))
            
            # append best performance to the exp_csv
            exp_df['{}_perf_selected'.format(subset)] = slice_percent_perf
            exp_df.to_csv(exp_csv)

            model_perf_dict[(exp_name,subset)] = (slice_percent_perf, volume_perf_list)

    return model_perf_dict