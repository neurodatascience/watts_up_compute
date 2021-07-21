#!/bin/bash

SUBJECT_ID="sub-000"
RUN_ID="RUN_1"

DATA_DIR="/home/nikhil/projects/green_comp_neuro/watts_up_compute/local_test_data/mni/"
INPUT_FILE_NAME="${SUBJECT_ID}_ses-1_run-1_desc-preproc_T1w.nii.gz"

OUTPUT_DIR=${DATA_DIR}/${RUN_ID}
TRACKER_LOG_DIR="/home/nikhil/projects/green_comp_neuro/watts_up_compute/tracker_output/FastSurfer/CNN/local_tests/${RUN_ID}"

python3 eval_with_multiple_trackers.py --i_dir ${DATA_DIR} \
	--o_dir ${OUTPUT_DIR} \
	--t ${SUBJECT_ID} \
	--in_name ${INPUT_FILE_NAME} \
	--log temp_Competitive.log \
	--network_sagittal_path ../checkpoints/Sagittal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_coronal_path ../checkpoints/Coronal_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--network_axial_path ../checkpoints/Axial_Weights_FastSurferCNN/ckpts/Epoch_30_training_state.pkl \
	--geo_loc '45.4972159,-73.6103642' \
	--tracker_log_dir ${TRACKER_LOG_DIR} \
	--mock_run 1