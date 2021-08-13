# Import modules
import os
import sys
from os.path import join as opj
import pandas as pd
import time
from nipype.interfaces.freesurfer import ReconAll
from nipype.interfaces.utility import IdentityInterface
from nipype.pipeline.engine import Workflow, Node
from pypapi import events, papi_high as high
import argparse

# Add paths (singularity should see these)
# FastSurfer and carbon trackers are in the mounted dir as these repos keep getting updated.
# TODO replace this with setup.py once the dependencis become stable
sys.path.append('../../../experiment-impact-tracker/')
sys.path.append('../../../codecarbon/')

from experiment_impact_tracker.compute_tracker import ImpactTracker
from codecarbon import EmissionsTracker, OfflineEmissionsTracker

def get_reconall(recon_directive,fs_folder):
    # This node represents the actual recon-all command
    reconall = Node(ReconAll(directive=recon_directive,
                            flags='-nuintensitycor -3T',
                            subjects_dir=fs_folder),
                    name="reconall")
    return reconall
                

# This function returns for each subject the path to struct.nii.gz
def pathfinder(subject, foldername, filename):
    from os.path import join as opj
    struct_path = opj(foldername, subject, filename)
    return struct_path


def main():
    # setup
    exp_start_time = time.time()
    
    # argparse
    parser = argparse.ArgumentParser(description='Script to run freesurfer reconall with nipype and track compute costs', epilog='$Id: fast_surfer_cnn, v 1.0 2019/09/30$')

    # Data
    parser.add_argument('--experiment_dir', dest='experiment_dir', help='path to directory to store freesurfer derived data.')
    parser.add_argument('--data_dir', help="path to input data", default='/neurohub/ukbb/imaging/')
    parser.add_argument('--subject_id', dest='subject_id', help='subject_id')
    parser.add_argument('--T1_identifier', help='T1 identifier string relateive to the subject directory')

    # FreeSurfer
    parser.add_argument('--recon_directive', dest='recon_directive', help='recon_directive (autorecon 1, 2, or 3)', default='1') #MTL
    
    # Trackers
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
    parser.add_argument('--count_FLOPs', dest='count_FLOPs',help="Count FLOPs using PAPI",action='store_true') 

    args = parser.parse_args()

    # Data
    experiment_dir =  args.experiment_dir
    data_dir = args.data_dir
    subject_id = args.subject_id
    T1_identifier = args.T1_identifier

    # FreeSurfer
    recon_directive = args.recon_directive

    # FLOPs
    count_FLOPs = args.count_FLOPs

    # Trackers
    tracker_log_dir = args.tracker_log_dir
    geo_loc = args.geo_loc
    CC_offline = args.CC_offline
    TZ = args.TZ
    iso_code = args.iso_code

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

    if count_FLOPs:
        print('Counting flops using PAPI')
        flop_csv = tracker_log_dir + 'compute_costs_flop.csv'
        flop_df = pd.DataFrame(columns=['task','start_time','duration','DP'])
        

    # Start FS processing for a given subject
    subject_list = [subject_id]

    fs_folder = opj(experiment_dir, 'freesurfer')  # location of freesurfer folder

    # Create the output folder - FreeSurfer can only run if this folder exists
    os.system('mkdir -p %s' % fs_folder)

    # Specify recon workflow stages
    if recon_directive == 'all':
        recon_directives = ['autorecon1','autorecon2','autorecon3']
    else:
        recon_directives = [recon_directive] 


    for r, recon_directive in enumerate(recon_directives):
        print('\nStarting stage: {}'.format(recon_directive))

        # Create the pipeline that runs the recon-all command
        reconflow = Workflow(name="reconflow")
        reconflow.base_dir = opj(experiment_dir, 'workingdir_reconflow')

        # Some magical stuff happens here (not important for now)
        infosource = Node(IdentityInterface(fields=['subject_id']), name="infosource")
        infosource.iterables = ('subject_id', subject_list)
        
        # Specify recon-all stage based on recon-directive
        reconall = get_reconall(recon_directive, fs_folder)
        # This section connects all the nodes of the pipeline to each other
        reconflow.connect([(infosource, reconall, [('subject_id', 'subject_id')]),
                        (infosource, reconall, [(('subject_id', pathfinder,
                                                    data_dir, T1_identifier),
                                                    'T1_files')]),
                        ])
        
        if count_FLOPs:
            # start flop counter
            start_time = time.time()
            high.start_counters([events.PAPI_DP_OPS,]) #default: PAPI_FP_OPS

        # This command runs the recon-all pipeline in parallel (using n_procs cores)
        # reconflow.run('MultiProc', plugin_args={'n_procs': 4})
        reconflow.run() 

        if count_FLOPs:
            # stop flop counter
            DP = high.stop_counters()[0]
            end_time = time.time()
            duration = end_time - start_time
            print('Duration: {}, Flops: {}'.format(duration, DP))

            flop_df.loc[r] = [recon_directive,start_time, duration, DP]

    ## code-carbon tracker
    tracker_CC.stop()
    
    if count_FLOPs:
        flop_df.to_csv(flop_csv)

if __name__=='__main__':
   main()
