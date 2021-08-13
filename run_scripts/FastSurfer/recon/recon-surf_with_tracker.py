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
# FastSurfer and carbon trackers are in the mounted dir as these repos keep getting updated.
# TODO replace this with setup.py once the dependencis become stable
# sys.path.append('../../../../FastSurfer/')
# sys.path.append('../../../../experiment-impact-tracker/')
# sys.path.append('../../../../codecarbon/')

from experiment_impact_tracker.compute_tracker import ImpactTracker
from codecarbon import EmissionsTracker, OfflineEmissionsTracker

HELPTEXT = """
Script to run recon-all with impact-tracker
Author: nikhil153
Date: Apr-28-2021
"""

parser = argparse.ArgumentParser(description=HELPTEXT)

# data
parser.add_argument('--subject_id', dest='subject_id', default="sub-000",
                    help='subject id')
parser.add_argument('--img_data_dir', dest='img_data_dir', default="./",
                    help='subject id')
parser.add_argument('--seg_data_dir', dest='seg_data_dir', default="./",
                    help='seg_data_dir id')                    
parser.add_argument('--input_file_name', dest='input_file_name', default="./",
                    help='input_file_name')
parser.add_argument('--output_data_dir', dest='output_data_dir', default="./output",
                    help='freesurfer derivative dir')
parser.add_argument('--fs_license', dest='fs_license', default="./fs_license.txt",
                    help='fs_license')

# trackers
parser.add_argument('--tracker_log_dir', dest='tracker_log_dir',
                    help="log dir for experiment impact tracker",
                    type=str, default='./tracker_logs/')
parser.add_argument('--geo_loc', dest='geo_loc',
                    help="(lat,log) coords for experiment impact tracker",
                    type=str, default='45.4972159,-73.6103642') #MTL Beluga
parser.add_argument('--CC_offline',
                    help="Run CC in offline mode",
                    action='store_true')                 
parser.add_argument('--TZ', dest='TZ',
                    help="TimeZone",
                    type=str, default='America/New_York')
parser.add_argument('--iso_code', dest='iso_code',
                    help="Country ISO code",
                    type=str, default='USA')
# PAPI
parser.add_argument('--count_FLOPs', dest='count_FLOPs',
                    help="Count FLOPs using PAPI",
                    action='store_true') 

args = parser.parse_args()                    

if __name__ == "__main__":

    # Data
    subject_id = args.subject_id
    img_data_dir = args.img_data_dir
    seg_data_dir = args.seg_data_dir
    input_file_name = args.input_file_name
    output_data_dir = args.output_data_dir
    fs_license = args.fs_license

    # FLOPs
    count_FLOPs = args.count_FLOPs

    # Trackers
    tracker_log_dir = args.tracker_log_dir
    geo_loc = args.geo_loc
    CC_offline = args.CC_offline
    TZ = args.TZ
    iso_code = args.iso_code

    if count_FLOPs:
        print('Counting flops using PAPI')

    print(f'Using offline mode for CC tracker: {CC_offline}')
    if CC_offline:
        print(f'Using {TZ} timezone and {iso_code} country iso code')
    
    print(f'Starting subject: {subject_id}')

    # Set up the trackers
    log_dir = '{}/{}/'.format(tracker_log_dir,subject_id)
    log_dir_EIT = f'{log_dir}/EIT/'
    log_dir_CC = f'{log_dir}/CC/'

    for d in [log_dir_EIT,log_dir_CC]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Use specified geo location for the HPC
    ly,lx = float(geo_loc.split(',')[0]), float(geo_loc.split(',')[1])
    coords = (ly,lx)
    print(f'Using geographical coordinates (long,lat): {coords}')

    # EIT tracker
    tracker_EIT = ImpactTracker(log_dir_EIT,coords)
    tracker_EIT.launch_impact_monitor()

    # CodeCarbon tracker
    os.environ['TZ']= TZ
    
    if CC_offline:
        tracker_CC = OfflineEmissionsTracker(output_dir=log_dir_CC, country_iso_code=iso_code)        
    else:
        tracker_CC = EmissionsTracker(output_dir=log_dir_CC)
        
    tracker_CC.start()

    # PAPI
    if count_FLOPs:
        papi_csv = '{}/{}/compute_costs_flop.csv'.format(log_dir,subject_id)
        papi_df = pd.DataFrame(columns=['task','start_time','duration','DP'])
        high.start_counters([events.PAPI_DP_OPS,]) #default: PAPI_FP_OPS, PAPI_SP_OPS gives zeros
    
    start_time = time.time()

    print('\nStarting recon all subprocess')
 
    subject_image_file_path = '{}/{}/{}'.format(img_data_dir, subject_id, input_file_name)
    subject_seg_file_path = '{}/{}/aparc.DKTatlas+aseg.deep.mgz'.format(seg_data_dir,subject_id)

    cmd = f"./run_fastsurfer.sh \
    --fs_license {fs_license} \
    --t1 {subject_image_file_path} \
    --no_cuda \
    --sid {subject_id}  \
    --sd {output_data_dir} \
    --surf_only \
    --seg {subject_seg_file_path} \
    --parallel"


    returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
    print('FastSurfer subprocess returned value:', returned_value)
    
    ## code-carbon tracker
    tracker_CC.stop()

    if count_FLOPs:
        DP = high.stop_counters()[0]
        end_time = time.time()
        recon_time = end_time - start_time
        papi_df.loc[0] = ['recon', start_time, recon_time, DP]
        papi_df.to_csv(papi_csv)
