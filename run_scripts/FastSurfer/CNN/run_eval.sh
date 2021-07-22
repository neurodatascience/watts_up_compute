#!/bin/bash

SUBJECT_ID="sub-000"
RUN_ID="RUN_1"
PROJECT_DIR="/home/nikhil/projects/green_comp_neuro/watts_up_compute/"
FASTSURFER_DIR='/home/nikhil/projects/green_comp_neuro/FastSurfer/'

DATA_DIR="${PROJECT_DIR}/local_test_data/mni/"
INPUT_FILE_NAME="${SUBJECT_ID}_ses-1_run-1_desc-preproc_T1w.nii.gz"

PROC_OUTPUT_DIR="${PROJECT_DIR}/proc_output/FastSurfer/CNN/local_tests/${RUN_ID}"
TRACKER_LOG_DIR="${PROJECT_DIR}/tracker_output/FastSurfer/CNN/local_tests/${RUN_ID}"

python3 eval_with_multiple_trackers.py --i_dir ${DATA_DIR} \
	--o_dir ${PROC_OUTPUT_DIR} \
	--t ${SUBJECT_ID} \
	--in_name ${INPUT_FILE_NAME} \
	--log temp_Competitive.log \
	--network_sagittal_path ${FASTSURFER_DIR}/checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_coronal_path ${FASTSURFER_DIR}/checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_axial_path ${FASTSURFER_DIR}/checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--geo_loc '45.4972159,-73.6103642' \
	--tracker_log_dir ${TRACKER_LOG_DIR} \
	--mock_run 0