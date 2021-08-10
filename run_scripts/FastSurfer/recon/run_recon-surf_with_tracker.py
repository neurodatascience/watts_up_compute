# IMPORTS
import argparse
import nibabel as nib
import numpy as np
import time
import sys
import os
import subprocess

# Compute costs
import pandas as pd
from ptflops import get_model_complexity_info
from pypapi import events, papi_high as high

# Add paths (singularity should see these)
sys.path.append('../../../FastSurfer/')

sys.path.append('../../../../experiment-impact-tracker/')
from experiment_impact_tracker.compute_tracker import ImpactTracker

sys.path.append('../../../../codecarbon/')
from codecarbon import EmissionsTracker, OfflineEmissionsTracker

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
    log_dir_EIT = f'{log_dir}/EIT/'
    log_dir_CC = f'{log_dir}/CC/'

    for d in [log_dir_EIT,log_dir_CC]:
        if not os.path.exists(d):
            os.makedirs(d)

    geo_loc = args.geo_loc
    ly,lx = float(geo_loc.split(',')[0]), float(geo_loc.split(',')[1])
    coords = (ly,lx)

    # EIT tracker
    tracker_EIT = ImpactTracker(log_dir_EIT,coords)
    tracker_EIT.launch_impact_monitor()

    # CodeCarbon tracker
    os.environ['TZ']= 'America/New_York'
    # tracker_CC = EmissionsTracker(output_dir=log_dir_CC) 
    tracker_CC = OfflineEmissionsTracker(output_dir=log_dir_CC, country_iso_code="USA")
    tracker_CC.start()

    # PAPI
    # papi_csv = '{}/{}/compute_costs_flop.csv'.format(args.tracker_log_dir,args.subject_id)
    # papi_df = pd.DataFrame(columns=['task','start_time','duration','DP'])
   
    # high.start_counters([events.PAPI_DP_OPS,]) #default: PAPI_FP_OPS, PAPI_SP_OPS gives zeros
    
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
    
    ## code-carbon tracker
    tracker_CC.stop()

    # DP = high.stop_counters()[0]
    # end_time = time.time()
    # recon_time = end_time - start_time
   
    # papi_df.loc[0] = ['recon', start_time, recon_time, DP]

    # papi_df.to_csv(papi_csv)
