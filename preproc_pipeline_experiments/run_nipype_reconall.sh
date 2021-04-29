#!/bin/bash

source /opt/freesurfer-6.0.0-min/SetUpFreeSurfer.sh

python3 nipype_FS_reconall.py \
	--experiment_dir /home/nikhil/green_compute/ukb_pilot \
	--data_dir /neurohub/ukbb/imaging/\
	--subject_id $1 \
	--T1_identifier ses-2/anat/$1_ses-2_T1w.nii.gz \
	--tracker_output_dir tracker_logs/ \
	--geo_loc "45.4972159,-73.6103642" \
	--recon_directive all 
