import time
import datetime
import os
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from ptflops import get_model_complexity_info

from pyJoules.energy_meter import EnergyContext
from pyJoules.handler.pandas_handler import PandasHandler

from data import *
from model import *

experiment_name = 'Exp_pytorch_cifar'
data_path = './data/'

model_name = 'ResNet_1'
optimizer_name = 'adam'
n_epochs = 1
batch_size = 4

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
    experiment_dir = '{}{}'.format(output_dir,experiment_name)
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    print('experiment output dir: {}'.format(experiment_dir))
    
    # experiment config
    exp_csv = '{}/experiment_config.csv'.format(experiment_dir)
    exp_cols = ['experiment_name', 'device', 'model', 'MAC', 'params', 'n_epochs', 'batch_size', 'optimizer']
    exp_df = pd.DataFrame(columns=exp_cols)
    
    # energy tracker
    joules_csv = '{}/joules.csv'.format(experiment_dir)
    pd_handler = PandasHandler()

    # epoch tracker
    epoch_csv = '{}/epoch.csv'.format(experiment_dir)

    # data
    train_loader, test_loader = get_cifar10_dataset(data_path, batch_size)
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    _,n_channels,input_size,_ = images.shape
    
    # model definition
    model_path = '{}/{}.pth'.format(experiment_dir,model_name)

    if model_name == 'ResNet_1':
        model = ResNet_1(ResidualBlock, [2, 2, 2]).to(device)
    elif model_name == 'ResNet_2':
        model = ResNet_2(ResidualBlock, [2, 2, 2]).to(device)
    else:
        print('Unknown model: {}'.format(model_name))

    # model complexity
    macs, params = get_model_complexity_info(model, (n_channels, input_size, input_size), as_strings=True,
                                        print_per_layer_stat=False)

    # populate experiment config
    exp_df.loc[0] = [experiment_name, device, model_name, macs, params, n_epochs, batch_size, optimizer_name]

    # optimizer
    criterion = nn.CrossEntropyLoss()

    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    
    # train
    print('Begin training (n_epochs={}, batch_size={})'.format(n_epochs,batch_size))
    
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
            if i % 2000 == 1999:    # print every 2000 mini-batches
                avg_loss = running_loss / 2000
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, avg_loss))
                running_loss = 0.0

        # epoch end time
        end_time = time.time()
        compute_time = (end_time - start_time)/60.0
        epoch_df.loc[epoch] = [epoch,compute_time,avg_loss]

    # train end time
    train_end_time = time.time()
    train_compute_time = (train_end_time - train_start_time)/60.0
    print('\nFinished Training at {}'.format(tic_time))

    # test the model
    # test start time
    test_start_time = time.time()
    print('Evaluating on test set')
    model.eval()
    with torch.no_grad():
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
    
    exp_df['test_perf'] = test_perf

    # test end time
    test_end_time = time.time()
    test_compute_time = (test_end_time - test_start_time)/60.0

    # experiment end time
    exp_end_time = time.time()
    exp_compute_time = (exp_end_time - exp_start_time)/60.0

    exp_df.loc[:,['train_compute_time','test_compute_time','experiment_compute_time']] = [train_compute_time,test_compute_time,exp_compute_time]

    # Save 
    print('Saving model at: {}'.format(model_path))
    torch.save(model.state_dict(), model_path)

    print('Saving joules trace at: {}'.format(joules_csv))
    pd_handler.get_dataframe().to_csv(joules_csv)

    print('Saving epoch df at: {}'.format(epoch_csv))
    epoch_df.to_csv(epoch_csv)

    print('Saving experiment config df at: {}'.format(exp_csv))
    exp_df.to_csv(exp_csv)

    print('\nFinished experiment {} at {}'.format(experiment_name, tic_time))


if __name__=='__main__':
   main()