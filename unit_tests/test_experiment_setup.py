import numpy as np
import pandas as pd
import sys
import pytest

sys.path.append('../')
sys.path.append('./')

from ml.data import *
from ml.model import *
from ml.kaggle_dataset import *
from lib.io_utils import *

DATA_ROOT_DIR = '../datasets/' #'/home/nikhil/projects/green_comp_neuro/watts_up_compute/datasets/'

# test data
def test_data_loaders():
    datasets = ['cifar10','kaggle_3m_small']
    for dataset in datasets:
        if dataset == 'cifar10':
            dataset_path = '{}{}/'.format(DATA_ROOT_DIR, dataset)
            train_loader, test_loader = get_cifar10_dataset(dataset_path, 4)
        
        elif dataset == 'kaggle_3m_small':
            dataset_path = '{}{}/'.format(DATA_ROOT_DIR, dataset)
            train_loader, test_loader = get_kaggle_dataset(dataset_path)
        
        else:
            print('Unknown dataset {}'.format(dataset))
            
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        _,n_channels,dim1,dim2 = images.shape
        train_input_shape = (n_channels,dim1,dim2)

        dataiter = iter(test_loader)
        images, labels = dataiter.next()
        _,n_channels,dim1,dim2 = images.shape
        test_input_shape = (n_channels,dim1,dim2)

        assert train_input_shape == test_input_shape


def test_model_output():
    device = 'cpu'
    model = ResNet_1(device, ResidualBlock, [2, 2, 2])
    x = np.random.random((1, 3, 32, 32))
    x_tensor = torch.from_numpy(x).float()
    y = model(x_tensor.clone().detach())
    y_shape = y.detach().numpy().shape
    assert y_shape == (1,10)