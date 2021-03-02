import time
import datetime
import os
import sys
import pandas as pd
import numpy as np
import argparse

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from cpuinfo import get_cpu_info
from ptflops import get_model_complexity_info
from pyJoules.energy_meter import EnergyContext
from pyJoules.handler.pandas_handler import PandasHandler

sys.path.append('../')
from ml.loss import DiceLoss
from ml.data import *
from ml.model import *


###
# Sample cmds:
# python run_experiment.py # This by default runs cifar10 experiment 
# python run_experiment.py --experiment_name Exp_pytorch_kaggle --dataset_name kaggle_3m_small --model_name unet --loss_type dice --monitor_joules 0
###

parser = argparse.ArgumentParser(description='Runs a compute cost experiment with DL models')

parser.add_argument('--experiment_name', type=str, default='Exp_test_run', help='')
parser.add_argument('--dataset_name', type=str, default='cifar10', help='')
parser.add_argument('--data_dir', type=str, default='../datasets/', help='')
parser.add_argument('--model_name', type=str, default='ResNet_1', help='')
parser.add_argument('--loss_type', type=str, default='cross-entropy', help='')
parser.add_argument('--optimizer_name', type=str, default='adam', help='')
parser.add_argument('--n_epochs', type=int, default=1, help='')
parser.add_argument('--batch_size', type=int, default=4, help='')
parser.add_argument('--test_cases', type=int, default=10, help='')
parser.add_argument('--monitor_joules', type=bool, default=True, help='')
parser.add_argument('--monitor_interval', type=int, default=10, help='')
parser.add_argument('--output_dir', type=str, default='../results/', help='')

def inference(model, data_loader, criterion, loss_type, device): 
        model.eval()
        with torch.no_grad():
            running_loss = []
            correct = 0
            total = 0  
            for images, labels in data_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss.append(loss.item())

                if loss_type == 'cross-entropy':
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    percent_perf = 100 * correct / total
                    
                elif loss_type == 'dice':
                    percent_perf = 100 * (1 - np.mean(running_loss))

                else:
                    print('unknown loss type: {}'.format(loss_type))
                    break

        return running_loss, percent_perf

def main():

    args = parser.parse_args()
    experiment_name = args.experiment_name #'Exp_pytorch_kaggle' #Exp_pytorch_cifar

    dataset_name = args.dataset_name #'kaggle_3m' #'kaggle_3m_small' #'cifar10'
    root_data_dir = args.data_dir #'/home/nikhil/scratch/deep_learning/datasets/' #'../datasets/'
    data_path = '{}{}'.format(root_data_dir,dataset_name)

    model_name = args.model_name #'unet' #'ResNet_1'
    loss_type = args.loss_type #'cross-entropy' #'cross-entropy' #'dice'
    optimizer_name = args.optimizer_name #'adam' or 'SGD'
    n_epochs = args.n_epochs #1
    batch_size = args.batch_size #4
    test_cases = args.test_cases #10

    monitor_joules = bool(args.monitor_joules) #True
    monitor_interval = args.monitor_interval
    output_dir = args.output_dir #'../results/'
    
    tic_time = datetime.datetime.now()
    print('\nStarting experiment: {} at {}'.format(experiment_name, tic_time))
    print('using root data dir: {} and dataset: {}\n'.format(root_data_dir,dataset_name))

    # experiment start time
    exp_start_time = time.time()

    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))
    
    # check / create experiment subdir
    experiment_dir = '{}{}/{}'.format(output_dir,experiment_name,device)
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    print('experiment output dir: {}'.format(experiment_dir))
    
    # experiment config
    exp_csv = '{}/experiment_config.csv'.format(experiment_dir)
    exp_cols = ['experiment_name','proc','arch','count','python_version', 'device', 'model', 'MAC', 'params', 'n_epochs', 'batch_size', 'optimizer']
    exp_df = pd.DataFrame(columns=exp_cols)

    # cpu data
    cpu_df = pd.DataFrame(get_cpu_info().items(),columns=['field','value']).set_index('field')

    # laptop and cluter CPUs have difference in "cpuinfo" labels
    if 'brand' in cpu_df.index:
        brand_str = 'brand'
    else:
        brand_str = 'brand_raw'
    
    cpu_info = list(np.hstack(cpu_df.loc[[brand_str,'arch','count','python_version']].values))
    
    # energy tracker
    joules_csv = '{}/joules.csv'.format(experiment_dir)
    pd_handler = PandasHandler()

    # iter tracker
    iter_csv = '{}/iter.csv'.format(experiment_dir)
    # epoch tracker
    epoch_csv = '{}/epoch.csv'.format(experiment_dir)

    # data
    if dataset_name == 'cifar10':
        train_loader, valid_loader, test_loader = get_cifar10_dataset(data_path, test_cases=None, batch_size=100)
    elif dataset_name in ['kaggle_3m','kaggle_3m_small']:
        train_loader, valid_loader, test_loader = get_kaggle_dataset(data_path, test_cases=test_cases, batch_size=batch_size)
    else:
        print('Unknown dataset: {}'.format(dataset_name))

    print('\nMonitoring loss and energy trace every: {} iters'.format(monitor_interval))
    print('\nNumber of test_cases (same as validation cases): {}'.format(test_cases))

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    _,n_channels,input_size,_ = images.shape
    
    print('input size: {}'.format([input_size,input_size,n_channels]))
    print('train samples: {}, valid samples: {}, test samples: {}'.format(len(train_loader),len(valid_loader),len(test_loader)))

    # model definition
    print('Using {} model'.format(model_name))
    model_path = '{}/{}.pth'.format(experiment_dir,model_name)

    if model_name == 'ResNet_1':
        model = ResNet_1(device, ResidualBlock, [2, 2, 2]).to(device)

    elif model_name == 'unet':
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=3, out_channels=1, init_features=32, pretrained=False)
    elif model_name == 'unet_medium':
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=3, out_channels=1, init_features=16, pretrained=False)
    elif model_name == 'unet_small':
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=3, out_channels=1, init_features=8, pretrained=False)
    elif model_name == 'unet_tiny':
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=3, out_channels=1, init_features=4, pretrained=False)
            
    else:
        print('Unknown model: {}'.format(model_name))

    # model complexity
    macs, params = get_model_complexity_info(model, (n_channels, input_size, input_size), as_strings=True,
                                        print_per_layer_stat=False)
    
    model.to(device) # get_model_complexity has a device mismatch otherwise.

    # populate experiment config
    exp_df.loc[0] = [experiment_name] + cpu_info + [device, model_name, macs, params, n_epochs, batch_size, optimizer_name]

    # optimizer
    if loss_type == 'dice':
        criterion = DiceLoss()
    elif loss_type == 'cross-entropy':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    
    # train
    tic_time = datetime.datetime.now()

    print('\nBegin training at {} (n_epochs={}, batch_size={})'.format(tic_time, n_epochs,batch_size))
    
    # train start time
    train_start_time = time.time()
    
    epoch_df = pd.DataFrame(columns=['epoch','compute_time','train_loss','valid_loss'])
    avg_loss_train = 0
    iter_loss_train = []
    iter_loss_valid = []
    iter_loss_test = []
    lowest_loss = 100 #used to decide whether to save model or not

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        # epoch start time
        start_time = time.time()

        running_loss = 0.0
        # for i, data in enumerate(trainloader, 0):
        for i, (images, labels) in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            images = images.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Monitor joules sparingly
            if (i % monitor_interval) == (monitor_interval-1):
                if monitor_joules:
                    # pyjoules
                    with EnergyContext(handler=pd_handler, start_tag='forward') as ctx:
                        # forward + backward + optimize
                        outputs = model(images)
                        ctx.record(tag='loss')
                        loss = criterion(outputs, labels)
                        ctx.record(tag='backward')  
                        loss.backward()
                        ctx.record(tag='step')
                        optimizer.step()
                        ctx.record(tag='overhead')

                # print statistics
                # train loss
                running_loss += loss.item()
                avg_loss_train = running_loss / monitor_interval
                # valid loss
                valid_loss, percent_perf = inference(model, valid_loader, criterion, loss_type, device)
                avg_loss_valid = np.mean(valid_loss)
                # test loss (for reference)
                test_loss, percent_perf = inference(model, test_loader, criterion, loss_type, device)
                avg_loss_test = np.mean(test_loss)

                print('epoch:{}, iter:{}, train_loss: {:4.3f}, valid_loss: {:4.3f}, test_loss: {:4.3f}'.format(epoch + 1, i + 1, avg_loss_train, 
                avg_loss_valid,avg_loss_test))

                iter_loss_train.append(avg_loss_train)
                iter_loss_valid.append(avg_loss_valid)
                iter_loss_test.append(avg_loss_test)

                running_loss = 0.0

                if avg_loss_valid < lowest_loss:
                    lowest_loss = avg_loss_valid 
                    # save model 
                    print('Saving model at: {}'.format(model_path))
                    torch.save(model.state_dict(), model_path)

            else:
                outputs = model(images)                
                loss = criterion(outputs, labels)            
                loss.backward()                
                optimizer.step()
                running_loss += loss.item()

        # epoch end time
        end_time = time.time()
        compute_time = (end_time - start_time)/60.0
        epoch_df.loc[epoch] = [epoch,compute_time,avg_loss_train,avg_loss_valid]

    # train end time
    train_end_time = time.time()
    train_compute_time = (train_end_time - train_start_time)/60.0
    toc_time = datetime.datetime.now()
    print('\nFinished training at {}, compute time:{}'.format(toc_time,train_compute_time))

    # test the model
    # test start time
    test_start_time = time.time()
    print('Evaluating on test set')

    running_loss, percent_perf = inference(model, test_loader, criterion, loss_type, device)
    exp_df['test_perf'] = percent_perf
    print('Percent test perf: {:4.3f}'.format(percent_perf))

    # test end time
    test_end_time = time.time()
    test_compute_time = (test_end_time - test_start_time)/60.0
    toc_time = datetime.datetime.now()
    print('\nFinished testing at {}, compute time:{}'.format(toc_time,test_compute_time))

    # experiment end time
    exp_end_time = time.time()
    exp_compute_time = (exp_end_time - exp_start_time)/60.0

    exp_df.loc[:,['train_compute_time','test_compute_time','experiment_compute_time']] = [train_compute_time,test_compute_time,exp_compute_time]

    

    # save iter data
    print('Saving iter df at: {}'.format(iter_csv))
    if monitor_joules:
        joules_df = pd_handler.get_dataframe()
        print('Saving joules trace at: {}'.format(joules_csv))
        joules_df.to_csv(joules_csv)

    iter_df = pd.DataFrame()
    iter_df['train_loss'] = iter_loss_train
    iter_df['valid_loss'] = iter_loss_valid
    iter_df['test_loss'] = iter_loss_test
    iter_df.to_csv(iter_csv)

    # save epoch data
    print('Saving epoch df at: {}'.format(epoch_csv))
    epoch_df.to_csv(epoch_csv)

    # save experiment data
    print('Saving experiment config df at: {}'.format(exp_csv))
    exp_df.to_csv(exp_csv)

    toc_time = datetime.datetime.now()
    print('\nFinished experiment {} at {}, compute time: {}'.format(experiment_name, toc_time, exp_compute_time))


if __name__=='__main__':
   main()
