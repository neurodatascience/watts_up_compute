# Source FreeSurfer
export FREESURFER_HOME=/opt/freesurfer/
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Run FastSurfer
./run_fastsurfer.sh --fs_license /fs60/license.txt \
    --t1 /data/sub-0039-rep/sub-0039_ses-1_run-1_desc-preproc_T1w.nii.gz \
    --no_cuda \
    --sid sub-0039  \
    --sd /output \
    --surf_only \
    --seg /data/sub-0039-rep/aparc.DKTatlas+aseg.deep.mgz \
    --parallel