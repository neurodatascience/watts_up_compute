#!/bin/bash

echo "sourcing FreeSurfer env var" 
export FREESURFER_HOME=/opt/freesurfer-6.0.0/
source $FREESURFER_HOME/SetUpFreeSurfer.sh

echo "sourcing FastSurfer env var"
export FASTSURFER_HOME=/home/nikhil/projects/green_comp_neuro/FastSurfer

subject_id=$1
run_id=$2

[[ -z $subject_id ]] && exit 1
[[ -z $run_id ]] && exit 1

# IMG_DATA_DIR="/neurohub/ukbb/imaging/" 
# SEG_DATA_DIR="/home/nikhil/green_compute/ukb_pilot/fastsurfer/recon-surf/prune_50/" 
# INPUT_FILE_NAME="ses-2/anat/${subject_id}_ses-2_T1w.nii.gz"
IMG_DATA_DIR="/home/nikhil/projects/green_comp_neuro/watts_up_compute/local_test_data/mni/" 
SEG_DATA_DIR="/home/nikhil/projects/green_comp_neuro/watts_up_compute/proc_output/FastSurfer/CNN/local_tests/RUN_1/" 
INPUT_FILE_NAME="${subject_id}_ses-1_run-1_desc-preproc_T1w.nii.gz"

FS_LICENSE="$FREESURFER_HOME/license.txt"

echo $FS_LICENSE

echo "Starting recon all with tracker..."

python3 run_recon-surf_with_tracker.py \
    --subject_id  $subject_id \
    --img_data_dir $IMG_DATA_DIR \
    --seg_data_dir $SEG_DATA_DIR \
    --input_file_name $INPUT_FILE_NAME \
    --fs_license $FS_LICENSE \
    --tracker_log_dir "/output/tracker_logs/${run_id}" \
    --geo_loc "45.4972159,-73.6103642" 
