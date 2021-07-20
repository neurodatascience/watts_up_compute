#!/bin/bash

echo "sourcing FreeSurfer env"
export FREESURFER_HOME=/opt/freesurfer/
source $FREESURFER_HOME/SetUpFreeSurfer.sh

subject_id=$1
run_id=$2

[[ -z $subject_id ]] && exit 1
[[ -z $run_id ]] && exit 1

IMG_DATA_DIR="/neurohub/ukbb/imaging/" 
SEG_DATA_DIR="/home/nikhil/green_compute/ukb_pilot/fastsurfer/recon-surf/prune_50/" 
INPUT_FILE_NAME="ses-2/anat/${subject_id}_ses-2_T1w.nii.gz"
FS_LICENSE="/home/nikhil/FastSurfer/license.txt"

echo "Starting recon all with tracker..."
python3 run_recon-surf_with_tracker.py \
    --subject_id  $subject_id \
    --img_data_dir $IMG_DATA_DIR \
    --seg_data_dir $SEG_DATA_DIR \
    --input_file_name $INPUT_FILE_NAME \
    --fs_license $FS_LICENSE \
    --tracker_log_dir "./tracker_logs/${run_id}" \
    --geo_loc "45.4972159,-73.6103642" 
