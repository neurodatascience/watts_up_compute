#!/bin/bash

subject_id="$1"
run_id="$2"
[[ -z $subject_id ]] && exit 1
[[ -z $run_id ]] && exit 1

source /opt/freesurfer-6.0.0-min/SetUpFreeSurfer.sh

python3 nipype_FS_reconall.py \
	--experiment_dir /home/nikhil/green_compute/ukb_pilot/freesurfer/${run_id}/ \
	--data_dir /neurohub/ukbb/imaging/\
	--subject_id $subject_id \
	--T1_identifier ses-2/anat/${subject_id}_ses-2_T1w.nii.gz \
	--tracker_output_dir tracker_logs/${run_id}/ \
	--geo_loc "45.4972159,-73.6103642" \
	--recon_directive all 
