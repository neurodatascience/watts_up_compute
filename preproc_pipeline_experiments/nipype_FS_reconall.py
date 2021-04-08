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

# Specify important variables
experiment_dir =  '/home/nikhil/green_compute/freesurfer/ukb_pilot' #'~/nipype_tutorial'           # location of experiment folder


data_dir = '/neurohub/ukbb/imaging/'
subject_list = ['sub-2017717']
T1_identifier = 'ses-2/anat/sub-2017717_ses-2_T1w.nii.gz'

fs_folder = opj(experiment_dir, 'freesurfer')  # location of freesurfer folder

log_dir = './logs/ReconAll_test_singularity/'
flop_csv = log_dir + 'compute_costs_flop.csv'


def get_reconall(recon_directive):
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

    # Create the output folder - FreeSurfer can only run if this folder exists
    os.system('mkdir -p %s' % fs_folder)

    # Specify recon workflow stages
    recon_directives = ['autorecon1'] #,'autorecon2','autorecon3'] #'autorecon1',

    flop_df = pd.DataFrame(columns=['task','start_time','duration','DP'])

    # experiment impact tracker
    # Init tracker with log path
    coords = (45.4972159,-73.6103642) #MTL
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
        reconall = get_reconall(recon_directive)
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
