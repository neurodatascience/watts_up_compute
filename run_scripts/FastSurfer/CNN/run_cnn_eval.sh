#!/bin/bash

# Sample cmd:
# local: singularity exec -B /home/nikhil/projects/green_comp_neuro/watts_up_compute/:/output ../../../../FastSurfer_containers/FastSurfer.sif ./run_cnn_eval.sh sub-000 RUN_2
# hpc: 

SUBJECT_ID=$1
RUN_ID=$2
HPC=$3

echo $HPC

[[ -z $SUBJECT_ID ]] && exit 1
[[ -z $RUN_ID ]] && exit 1

if [ -z $HPC ]; then
    echo "Using local data"
	PROJECT_DIR="/home/nikhil/projects/green_comp_neuro/watts_up_compute/"
	FASTSURFER_DIR="/home/nikhil/projects/green_comp_neuro/FastSurfer/"
	EIT_DIR="/home/nikhil/projects/green_comp_neuro/experiment-impact-tracker/"
	CC_DIR="/home/nikhil/projects/green_comp_neuro/codecarbon/"

 	IMG_DATA_DIR="${PROJECT_DIR}/local_test_data/mni/" 
    INPUT_FILE_NAME="${SUBJECT_ID}_ses-1_run-1_desc-preproc_T1w.nii.gz"
	PROC_OUTPUT_DIR="${PROJECT_DIR}/proc_output/FastSurfer/CNN/local_tests/${RUN_ID}"
	TRACKER_LOG_DIR="${PROJECT_DIR}/tracker_output/FastSurfer/CNN/local_tests/${RUN_ID}"

else
    echo "Using HPC data"
	FASTSURFER_DIR="/home/nikhil/FastSurfer/"
	EIT_DIR="/home/nikhil/experiment-impact-tracker/"
	CC_DIR="home/nikhil/codecarbon/"

    IMG_DATA_DIR="/neurohub/ukbb/imaging/" 
    INPUT_FILE_NAME="ses-2/anat/${SUBJECT_ID}_ses-2_T1w.nii.gz"
	PROC_OUTPUT_DIR="/output/proc_output/FastSurfer/CNN/hpc_tests/${RUN_ID}"
	TRACKER_LOG_DIR="/output/tracker_output/FastSurfer/CNN/hpc_tests/${RUN_ID}"
fi

# install git repos
# pip install -e $FASTSURFER_DIR
# pip install -e $EIT_DIR
# pip install -e $CC_DIR

python3 cnn_eval_with_tracker.py --i_dir ${IMG_DATA_DIR} \
	--o_dir ${PROC_OUTPUT_DIR} \
	--t ${SUBJECT_ID} \
	--in_name ${INPUT_FILE_NAME} \
	--log temp_Competitive.log \
	--geo_loc "45.4972159,-73.6103642" \
	--network_sagittal_path "${FASTSURFER_DIR}/checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl" \
	--network_coronal_path "${FASTSURFER_DIR}/checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl" \
	--network_axial_path "${FASTSURFER_DIR}/checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl" \
	--CC_offline \
	--tracker_log_dir ${TRACKER_LOG_DIR} \
	--mock_run 0
