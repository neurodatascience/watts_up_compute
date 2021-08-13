#!/bin/bash

# Sample cmd:
# local: 
# hpc: 

echo "sourcing FreeSurfer env var" 
export FREESURFER_HOME=/opt/freesurfer-6.0.0/
source $FREESURFER_HOME/SetUpFreeSurfer.sh

echo "sourcing FastSurfer env var"
export FASTSURFER_HOME=/home/nikhil/projects/green_comp_neuro/FastSurfer

SUBJECT_ID=$1
RUN_ID=$2
HPC=$3

[[ -z $SUBJECT_ID ]] && exit 1
[[ -z $RUN_ID ]] && exit 1

if [ -z $HPC]; then
    echo "Using local data"
    PROJECT_DIR="/home/nikhil/projects/green_comp_neuro/watts_up_compute/"
    IMG_DATA_DIR="${PROJECT_DIR}local_test_data/mni/" 
    SEG_DATA_DIR="${PROJECT_DIR}proc_output/FastSurfer/CNN/local_tests/RUN_1/" 
    INPUT_FILE_NAME="${SUBJECT_ID}_ses-1_run-1_desc-preproc_T1w.nii.gz"

    PROC_OUTPUT_DIR="/output/proc_output/FastSurfer/recon/local_tests/${RUN_ID}"
    TRACKER_LOG_DIR="/output/tracker_output/FastSurfer/recon/local_tests/${RUN_ID}"

else
    echo "Using HPC data"
    IMG_DATA_DIR="/neurohub/ukbb/imaging/" 
    SEG_DATA_DIR="/home/nikhil/green_compute/ukb_pilot/fastsurfer/recon-surf/prune_50/" 
    INPUT_FILE_NAME="ses-2/anat/${SUBJECT_ID}_ses-2_T1w.nii.gz"    

    PROC_OUTPUT_DIR="/output/proc_output/FastSurfer/recon/hpc_tests/${RUN_ID}"
    TRACKER_LOG_DIR="/output/tracker_output/FastSurfer/recon/hpc_tests/${RUN_ID}"

fi

FS_LICENSE="$FREESURFER_HOME/license.txt"
echo "FreeSurfer license at: $FS_LICENSE"

echo "Starting recon all with tracker..."

python3 recon-surf_with_tracker.py \
    --SUBJECT_ID  $SUBJECT_ID \
    --img_data_dir $IMG_DATA_DIR \
    --seg_data_dir $SEG_DATA_DIR \
    --input_file_name $INPUT_FILE_NAME \
    --fs_license $FS_LICENSE \
    --output_data_dir $FS_DERIVATIVES_DIR \
    --tracker_log_dir $TRACKER_LOG_DIR \
    --geo_loc "45.4972159,-73.6103642" \
    --CC_offline

