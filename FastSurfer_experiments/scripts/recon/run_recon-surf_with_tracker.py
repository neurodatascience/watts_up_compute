# IMPORTS
import argparse
import nibabel as nib
import numpy as np
import time
import sys
import subprocess

# Compute costs
import pandas as pd
from ptflops import get_model_complexity_info
from pypapi import events, papi_high as high

# experiment tracker
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../experiment-impact-tracker/')
from experiment_impact_tracker.compute_tracker import ImpactTracker


HELPTEXT = """
Script to run recon-all with impact-tracker
Author: nikhil153
Date: Apr-28-2021

"""

parser = argparse.ArgumentParser(description=HELPTEXT)
# 3. Options for log-file and search-tag
parser.add_argument('--subject_id', dest='subject_id', default="sub-000",
                    help='subject id')

parser.add_argument('--img_data_dir', dest='img_data_dir', default="./",
                    help='subject id')
parser.add_argument('--seg_data_dir', dest='seg_data_dir', default="./",
                    help='seg_data_dir id')                    
parser.add_argument('--input_file_name', dest='input_file_name', default="./",
                    help='input_file_name')
parser.add_argument('--fs_license', dest='fs_license', default="./fs_license.txt",
                    help='fs_license')

parser.add_argument('--geo_loc', dest='geo_loc',
                    help="(lat,log) coords for experiment impact tracker",
                    type=str, default='45.4972159,-73.6103642') #MTL Beluga
parser.add_argument('--tracker_log_dir', dest='tracker_log_dir',
                    help="log dir for experiment impact tracker",
                    type=str, default='./tracker_logs/')


args = parser.parse_args()                    

if __name__ == "__main__":

    subject_id = args.subject_id
    img_data_dir = args.img_data_dir
    seg_data_dir = args.seg_data_dir
    input_file_name = args.input_file_name
    fs_license = args.fs_license

    print('Starting subject: {}'.format(subject_id))

    # Set up the tracker
    log_dir = '{}/{}/'.format(args.tracker_log_dir,args.subject_id)

    geo_loc = args.geo_loc
    ly,lx = float(geo_loc.split(',')[0]), float(geo_loc.split(',')[1])
    coords = (ly,lx)

    # Init tracker with log path
    tracker = ImpactTracker(log_dir,coords)

    # Start tracker in a separate process
    tracker.launch_impact_monitor()

    # PAPI
    papi_csv = '{}/{}/compute_costs_flop.csv'.format(args.tracker_log_dir,args.subject_id)
    papi_df = pd.DataFrame(columns=['task','start_time','duration','DP'])
   
    high.start_counters([events.PAPI_DP_OPS,]) #default: PAPI_FP_OPS, PAPI_SP_OPS gives zeros
    
    start_time = time.time()

    print('\nStarting recon all subprocess')
    # Spawn recon-surf script
    # cmd = "./test_cmd.sh"
 
    subject_image_file_path = '{}/{}/{}'.format(img_data_dir, subject_id, input_file_name)
    subject_seg_file_path = '{}/{}/aparc.DKTatlas+aseg.deep.mgz'.format(seg_data_dir,subject_id)

    cmd = "../run_fastsurfer.sh --fs_license {} \
    --t1 {} \
    --no_cuda \
    --sid {}  \
    --sd /output \
    --surf_only \
    --seg {} \
    --parallel".format(fs_license,subject_image_file_path,subject_id,subject_seg_file_path)


    returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
    print('returned value:', returned_value)

    DP = high.stop_counters()[0]
    end_time = time.time()
    recon_time = end_time - start_time
    papi_df.loc[0] = ['recon', start_time, recon_time, DP]

    papi_df.to_csv(papi_csv)
