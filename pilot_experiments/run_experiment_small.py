import time
import datetime
import os
import pandas as pd
import numpy as np
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

from loss import DiceLoss
from data import *
from model import *

experiment_name = 'Exp_pytorch_kaggle' #Exp_pytorch_cifar
data_path = '../datasets/kaggle_3m' #'/home/nikhil/scratch/deep_learning/datasets/kaggle_3m' #'../datasets/kaggle_3m' #'../datasets/cifar10'

dataset_name = 'kaggle' #'cifar'
model_name = 'unet' #'ResNet_1'
loss_type = 'dice'
optimizer_name = 'adam'
n_epochs = 10
batch_size = 4
monitor_joules = True
monitor_interval = 50 #2000

output_dir = '../results/'

def main():
    
    tic_time = datetime.datetime.now()
    print('\nStarting experiment: {} at {}'.format(experiment_name, tic_time))

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

    # epoch tracker
    epoch_csv = '{}/epoch.csv'.format(experiment_dir)

    # data
    if dataset_name == 'cifar':
        train_loader, test_loader = get_cifar10_dataset(data_path, batch_size)
    elif dataset_name == 'kaggle':
        train_loader, test_loader = get_kaggle_dataset(data_path)
    else:
        print('Unknown dataset: {}'.format(dataset_name))

    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    _,n_channels,input_size,_ = images.shape
    
    print('input size: {}'.format([input_size,input_size,n_channels]))
    # model definition
    print('Using {} model'.format(model_name))
    model_path = '{}/{}.pth'.format(experiment_dir,model_name)

    if model_name == 'ResNet_1':
        model = ResNet_1(device, ResidualBlock, [2, 2, 2]).to(device)
    elif model_name == 'unet':
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=3, out_channels=1, init_features=32, pretrained=False)
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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
    # train
    tic_time = datetime.datetime.now()

    print('\nBegin training at {} (n_epochs={}, batch_size={})'.format(tic_time, n_epochs,batch_size))
    
    # train start time
    train_start_time = time.time()
    
    epoch_df = pd.DataFrame(columns=['epoch','compute_time','loss'])
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

                # print statistics
                running_loss += loss.item()
                avg_loss = running_loss / monitor_interval
                print('epoch:{}, iter:{}, loss: {:4.3f}'.format(epoch + 1, i + 1, avg_loss))
                running_loss = 0.0

            else:
                outputs = model(images)                
                loss = criterion(outputs, labels)            
                loss.backward()                
                optimizer.step()

        # epoch end time
        end_time = time.time()
        compute_time = (end_time - start_time)/60.0
        epoch_df.loc[epoch] = [epoch,compute_time,avg_loss]

    # train end time
    train_end_time = time.time()
    train_compute_time = (train_end_time - train_start_time)/60.0
    toc_time = datetime.datetime.now()
    print('\nFinished training at {}, compute time:{}'.format(toc_time,train_compute_time))

    # test the model
    # test start time
    test_start_time = time.time()
    print('Evaluating on test set')
    model.eval()
    with torch.no_grad():
        if loss_type == 'cross-entropy':
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            test_perf = 100 * correct / total
            print('Accuracy of the model on the test images: {} %'.format(test_perf))
    
        elif loss_type == 'dice':
            running_loss = []
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss.append(loss.item())

            test_perf = 100 * (1 - np.mean(running_loss))
            print('Dice performance (%) of the model on the test images: {}'.format(test_perf))

        else:
            print('unknown loss type: {}'.format(loss_type))

    exp_df['test_perf'] = test_perf

    # test end time
    test_end_time = time.time()
    test_compute_time = (test_end_time - test_start_time)/60.0
    toc_time = datetime.datetime.now()
    print('\nFinished testing at {}, compute time:{}'.format(toc_time,test_compute_time))

    # experiment end time
    exp_end_time = time.time()
    exp_compute_time = (exp_end_time - exp_start_time)/60.0

    exp_df.loc[:,['train_compute_time','test_compute_time','experiment_compute_time']] = [train_compute_time,test_compute_time,exp_compute_time]

    # Save 
    print('Saving model at: {}'.format(model_path))
    torch.save(model.state_dict(), model_path)

    if monitor_joules:
        print('Saving joules trace at: {}'.format(joules_csv))
        pd_handler.get_dataframe().to_csv(joules_csv)

    print('Saving epoch df at: {}'.format(epoch_csv))
    epoch_df.to_csv(epoch_csv)

    print('Saving experiment config df at: {}'.format(exp_csv))
    exp_df.to_csv(exp_csv)

    toc_time = datetime.datetime.now()
    print('\nFinished experiment {} at {}, compute time: {}'.format(experiment_name, toc_time, exp_compute_time))


if __name__=='__main__':
   main()
