#!/bin/bash

# Sample cmd:
# local: 
# hpc: 

SUBJECT_ID="$1"
RUN_ID="$2"
HPC="$3"

[[ -z $SUBJECT_ID ]] && exit 1
[[ -z $RUN_ID ]] && exit 1

source /opt/freesurfer-6.0.0/SetUpFreeSurfer.sh
# source /opt/freesurfer-6.0.0-min/SetUpFreeSurfer.sh

if [ -z $HPC]; then
    echo "Using local data"
	PROJECT_DIR="/home/nikhil/projects/green_comp_neuro/watts_up_compute/"
    IMG_DATA_DIR="${PROJECT_DIR}local_test_data/mni/" 
    INPUT_FILE_NAME="${SUBJECT_ID}_ses-1_run-1_desc-preproc_T1w.nii.gz"
	
	PROC_OUTPUT_DIR="/output/proc_output/FreeSurfer/local_tests/${RUN_ID}"
	TRACKER_LOG_DIR="/output/tracker_output/FreeSurfer/local_tests/${RUN_ID}"
else
    echo "Using HPC data"
	IMG_DATA_DIR="/neurohub/ukbb/imaging/"
	INPUT_FILE_NAME="ses-2/anat/${SUBJECT_ID}_ses-2_T1w.nii.gz"

	PROC_OUTPUT_DIR="/output/proc_output/FreeSurfer/hpc_tests/${RUN_ID}"
	TRACKER_LOG_DIR="/output/tracker_output/FreeSurfer/hpc_tests/${RUN_ID}"
fi

python3 nipype_reconall_with_tracker.py \
	--SUBJECT_ID $SUBJECT_ID  \
	--data_dir $IMG_DATA_DIR \
	--T1_identifier $INPUT_FILE_NAME \
	--experiment_dir $PROC_OUTPUT_DIR \
	--tracker_log_dir $TRACKER_LOG_DIR \
	--geo_loc "45.4972159,-73.6103642" \
	--recon_directive all \
	--CC_offline