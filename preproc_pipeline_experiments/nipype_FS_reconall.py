# Import modules
import os
from os.path import join as opj
import pandas as pd
from nipype.interfaces.freesurfer import ReconAll
from nipype.interfaces.utility import IdentityInterface
from nipype.pipeline.engine import Workflow, Node
from pypapi import events, papi_high as high

# Specify important variables
experiment_dir =  '/home/nipype_tutorial' #'~/nipype_tutorial'           # location of experiment folder
data_dir = opj(experiment_dir, 'data')         # location of data folder
fs_folder = opj(experiment_dir, 'freesurfer')  # location of freesurfer folder
subject_list = ['sub002']                        # subject identifier
T1_identifier = 'struct.nii.gz'                  # Name of T1-weighted image
flop_csv = 'FS_reconall_flop_2.csv'


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
    # Create the output folder - FreeSurfer can only run if this folder exists
    os.system('mkdir -p %s' % fs_folder)

    # Specify recon workflow stages
    recon_directives = ['autorecon1','autorecon2','autorecon3'] #'autorecon1',

    flop_df = pd.DataFrame(columns=['task','count'])

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
        # Papi
        # n_counters = high.num_counters()
        # print('papi counters: {}'.format(n_counters))

        # start flop counter
        high.start_counters([events.PAPI_DP_OPS,]) #default: PAPI_FP_OPS

        # This command runs the recon-all pipeline in parallel (using 8 cores)
        # reconflow.run('MultiProc', plugin_args={'n_procs': 1})
        reconflow.run() 

        # stop flop counter
        DP = high.stop_counters()
        print('Flops: {}'.format(DP))

        flop_df.loc[r] = [recon_directive,DP]

    flop_df.to_csv(flop_csv)

if __name__=='__main__':
   main()