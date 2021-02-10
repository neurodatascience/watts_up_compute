import torch 
import torchvision.models as models
import pandas as pd
import numpy as np
import argparse
import datetime
import time
from ptflops import get_model_complexity_info
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler
from cpuinfo import get_cpu_info

###
# Sample cmd: 
# python pytorch_test.py --input_size 256 --n_channels 1 --init_features 128 --max_epochs 10 --output_dir ../results/
###

parser = argparse.ArgumentParser(description='unet test for compute cost')

# inference params
parser.add_argument('--input_size', type=int, default=256, help='')
parser.add_argument('--n_channels', type=int, default=1, help='')
parser.add_argument('--init_features', type=int, default=32, help='')
parser.add_argument('--max_epochs', type=int, default=10, help='')
parser.add_argument('--output_dir', type=str, default='../results/', help='')

MODEL_NAME = 'unet'

CPU_PARAMS = ['brand', 'hz_advertised', 'hz_actual','cpu_arch','cpu_count','python_version']
GPU_PARAMS = ['gpu_device']
INPUT_MODEL_PARAMS = ['model','input_size','init_features','n_channels','FLOPs','n_parameters']

def measure_inference_energy(x, model, handler):
    @measure_energy(handler=handler)
    def predict(x, model):
        x_tensor = torch.from_numpy(x).float()
        model.eval()
        # y = model(torch.tensor(x_tensor))
        y = model(x_tensor.clone().detach())
        return y

    predict(x,model)

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
    output_dir = args.output_dir

    print('Configs:\ninit_features={}, input_size={}, n_channels={}, max_epochs={}'.format(init_features,input_size,n_channels,max_epochs))
    print('Saving perf output at: {}'.format(output_dir))

    # cpu data
    cpu_df = pd.DataFrame(get_cpu_info().items(),columns=['field','value']).set_index('field')
    
    # gpu data
    gpu_device = None

    # model
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', 
    in_channels=n_channels, out_channels=1, init_features=init_features, pretrained=False,verbose=False)
    
    # model complexity
    flops_csv = '{}flops_cpu.csv'.format(output_dir)
    macs, params = get_model_complexity_info(model, (n_channels, input_size, input_size), as_strings=True,
                                            print_per_layer_stat=False)

    # energy tracker
    joules_csv = '{}joules_cpu.csv'.format(output_dir)
    csv_handler = CSVHandler(joules_csv)

    # epoch loop
    perf_df = pd.DataFrame(columns=['epoch','compute_time']) 
    
    for epoch in range(max_epochs):
        # epoch start time
        start_time = time.time()
        
        # input
        x = np.random.random((1, n_channels, input_size, input_size))

        # predict
        measure_inference_energy(x, model, csv_handler)
        
        # epoch end time
        end_time = time.time()
        compute_time = (end_time - start_time)/60.0
        perf_df.loc[epoch] = [epoch,compute_time]

    perf_df[CPU_PARAMS + GPU_PARAMS + INPUT_MODEL_PARAMS] = list(np.hstack(cpu_df.loc[['brand','hz_advertised','hz_actual','arch','count','python_version']].values)) + \
            [gpu_device] + [MODEL_NAME, input_size, init_features, n_channels, macs, params]

    # save output
    perf_df.to_csv(flops_csv)
    csv_handler.save_data()

    # toc
    end_time = time.time()
    compute_time = (end_time - start_time)/60.0

    now = datetime.datetime.now()
    print('\n... ending analysis at {}'.format(now)) 
    print(' with compute time of {:3.2f} mins'.format(compute_time))


if __name__=='__main__':
   main()
