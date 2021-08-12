#!/bin/bash

subject_id="$1"
run_id="$2"
hpc="$3"

[[ -z $subject_id ]] && exit 1
[[ -z $run_id ]] && exit 1

source /opt/freesurfer-6.0.0/SetUpFreeSurfer.sh
# source /opt/freesurfer-6.0.0-min/SetUpFreeSurfer.sh

if [ -z $hpc]; then
    echo "Using local data"
	INPUT_DATA_DIR="/home/nikhil/projects/green_comp_neuro/watts_up_compute/local_test_data/mni/"
	INPUT_FILE_NAME="${subject_id}_ses-1_run-1_desc-preproc_T1w.nii.gz"
	FS_OUT_DIR="/home/nikhil/projects/green_comp_neuro/watts_up_compute/proc_output/FreeSurfer/${run_id}/"
else
    echo "Using HPC data"
	INPUT_DATA_DIR="/neurohub/ukbb/imaging/"
	INPUT_FILE_NAME="ses-2/anat/${subject_id}_ses-2_T1w.nii.gz"
	FS_OUT_DIR="/home/nikhil/green_compute/ukb_pilot/freesurfer/${run_id}/"	
fi

python3 nipype_FS_reconall.py \
	--subject_id $subject_id  \
	--data_dir $INPUT_DATA_DIR \
	--T1_identifier $INPUT_FILE_NAME \
	--experiment_dir $FS_OUT_DIR \
	--tracker_log_dir tracker_logs/${run_id}/ \
	--geo_loc "45.4972159,-73.6103642" \
	--recon_directive all \
	--CC_offline