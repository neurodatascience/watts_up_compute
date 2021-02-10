import torch 
import torchvision.models as models
import pandas as pd
import numpy as np
import argparse
import datetime
import time

###
# Sample cmd: 
# python pytorch_test.py --input_size 256 --n_channels 1 --init_features 128 --max_epochs 10 --output_csv ./output.csv
###

parser = argparse.ArgumentParser(description='unet test')

# Train params
# parser.add_argument('--lr', default=0.1, help='')
# parser.add_argument('--batch_size', type=int, default=768, help='')
# parser.add_argument('--max_epochs', type=int, default=4, help='')

# Test params
parser.add_argument('--input_size', type=int, default=256, help='')
parser.add_argument('--n_channels', type=int, default=1, help='')
parser.add_argument('--init_features', type=int, default=32, help='')
parser.add_argument('--max_epochs', type=int, default=10, help='')
parser.add_argument('--output_csv', type=str, default='./output.csv', help='')

# parser.add_argument('--num_workers', type=int, default=0, help='')

MODEL_NAME = 'unet'

def predict(x, model):
    x_tensor = torch.from_numpy(x).float().cuda() #compute on gpu
    model.eval()
    y = model(x_tensor.clone().detach())
    return y.cpu().detach().numpy() #move it to cpu for numpy


def main():
    # tic
    now = datetime.datetime.now()
    print('Starting analysis at {} ...'.format(now)) 
    start_time = time.time()
    
    # args
    args = parser.parse_args()
    input_size = args.input_size #i.e. 256x256
    n_channels = args.n_channels
    init_features = args.init_features
    max_epochs = args.max_epochs
    output_csv = args.output_csv

    print('Configs:\ninit_features={}, input_size={}, n_channels={}, max_epochs={}'.format(init_features,input_size,n_channels,max_epochs))
    print('Saving perf output at: {}'.format(output_csv))

    # epoch loop
    perf_df = pd.DataFrame(columns=['model','input_size','init_features','n_channels','epoch','compute_time']) 
    for epoch in range(max_epochs):
        # epoch start time
        start_time = time.time()
        
        # input
        x = np.random.random((1, n_channels, input_size, input_size))

        # model
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=n_channels, out_channels=1, init_features=init_features, pretrained=False,verbose=False)
        
        model.cuda() #to compute on gpu

        # output
        y = predict(x,model)

        # epoch end time
        end_time = time.time()
        compute_time = (end_time - start_time)/60.0
        perf_df.loc[epoch] = [MODEL_NAME,input_size,init_features,n_channels,epoch,compute_time]

    # save output
    perf_df.to_csv(output_csv)

    # toc
    end_time = time.time()
    compute_time = (end_time - start_time)/60.0

    now = datetime.datetime.now()
    print('\n... ending analysis at {}'.format(now)) 
    print(' with compute time of {:3.2f} mins'.format(compute_time))


if __name__=='__main__':
   main()
