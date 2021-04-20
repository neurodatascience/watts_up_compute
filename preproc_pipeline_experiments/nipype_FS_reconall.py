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

# experiment tracker
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../experiment-impact-tracker/')
from experiment_impact_tracker.compute_tracker import ImpactTracker

def get_reconall(recon_directive,fs_folder):
    # This node represents the actual recon-all command
    reconall = Node(ReconAll(directive=recon_directive,
                            #flags='-nuintensitycor- 3T',
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

    parser.add_argument('--experiment_dir', dest='experiment_dir', help='path to directory to store freesurfer derived data.')
    parser.add_argument('--data_dir', help="path to input data", default='/neurohub/ukbb/imaging/')
    parser.add_argument('--subject_id', dest='subject_id', help='subject_id')
    parser.add_argument('--T1_identifier', help='T1 identifier string relateive to the subject directory')
    parser.add_argument('--tracker_output_dir', dest='tracker_output_dir', help='tracker_output_dir', default='./tracker_output/')
    parser.add_argument('--geo_loc', dest='geo_loc', help='lat-long coordinate of the compute cluster', default='45.4972159,-73.6103642') #MTL
    parser.add_argument('--recon_directive', dest='recon_directive', help='recon_directive (autorecon 1, 2, or 3)', default='1') #MTL

    args = parser.parse_args()
    
    # example cmd: 
    # python3 nipype_FS_reconall.py 
    #     --experiment_dir /home/nikhil/projects/green_comp_neuro/watts_up_compute/preproc_pipeline_experiments/output \
    #     --data_dir /home/nikhil/projects/neurodocker/nipype_tutorial/data \
    #     --subject_id sub001 \
    #     --T1_identifier struct.nii.gz \
    #     --tracker_output_dir tracker_logs/ \
    #     --geo_loc "45.4972159,-73.6103642" \
    #     --recon_directive autorecon1 

    # Specify important variables
    experiment_dir =  args.experiment_dir
    data_dir = args.data_dir
    tracker_output_dir = args.tracker_output_dir
    geo_loc = args.geo_loc
    recon_directive = args.recon_directive

    subject_id = args.subject_id
    T1_identifier = args.T1_identifier
    # T1_identifier = #'ses-2/anat/{}_ses-2_T1w.nii.gz'.format(subject_id)

    subject_list = [subject_id]

    fs_folder = opj(experiment_dir, 'freesurfer')  # location of freesurfer folder

    log_dir = '{}/{}/'.format(tracker_output_dir,subject_id)
    flop_csv = log_dir + 'compute_costs_flop.csv'

    # Create the output folder - FreeSurfer can only run if this folder exists
    os.system('mkdir -p %s' % fs_folder)

    # Specify recon workflow stages
    if recon_directive == 'all':
        recon_directives = ['autorecon1','autorecon2','autorecon3']
    else:
        recon_directives = [recon_directive] 

    flop_df = pd.DataFrame(columns=['task','start_time','duration','DP'])

    # experiment impact tracker
    ly,lx = float(geo_loc.split(',')[0]), float(geo_loc.split(',')[1])
    coords = (ly,lx) 
    print('coords: {}'.format(coords))

    tracker = ImpactTracker(log_dir,coords)
    # Start tracker in a separate process
    tracker.launch_impact_monitor()

    for r, recon_directive in enumerate(recon_directives):
        print('\nStarting stage: {}'.format(recon_directive))

        # Create the pipeline that runs the recon-all command
        reconflow = Workflow(name="reconflow")
        reconflow.base_dir = opj(experiment_dir, 'workingdir_reconflow')

        # Some magical stuff happens here (not important for now)
        infosource = Node(IdentityInterface(fields=['subject_id']),
                        name="infosource")
        infosource.iterables = ('subject_id', subject_list)
        
        # Specify recon-all stage based on recon-directive
        reconall = get_reconall(recon_directive, fs_folder)
        # This section connects all the nodes of the pipeline to each other
        reconflow.connect([(infosource, reconall, [('subject_id', 'subject_id')]),
                        (infosource, reconall, [(('subject_id', pathfinder,
                                                    data_dir, T1_identifier),
                                                    'T1_files')]),
                        ])
        
        # start flop counter
        start_time = time.time()
        high.start_counters([events.PAPI_DP_OPS,]) #default: PAPI_FP_OPS

        # This command runs the recon-all pipeline in parallel (using n_procs cores)
        # reconflow.run('MultiProc', plugin_args={'n_procs': 4})
        reconflow.run() 

        # stop flop counter
        DP = high.stop_counters()[0]
        end_time = time.time()
        duration = end_time - start_time
        print('Duration: {}, Flops: {}'.format(duration, DP))

        flop_df.loc[r] = [recon_directive,start_time, duration, DP]

    flop_df.to_csv(flop_csv)

if __name__=='__main__':
   main()
