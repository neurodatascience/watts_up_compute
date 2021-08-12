# How to run pipelines with trackers

# FastSurfer
FastSurfer has two two stages: 1) CNN 2) recon. 

## CNN
FastSurfer CNN is a purely deep-learning based pipeline that can be run with minimal dependancies. The script "./FastSurfer/CNN/eval_with_multiple_tracker.py" imports FastSurfer package and wraps it within tracker processes. 

"./FastSurfer/CNN/run_eval.sh" is a sample script to drive python script with default arguments. 

## recon
FastSurfer recon is a bit more complex pipeline that heavily depends on FreeSurfer tools. It also relies on the output of FastCNN (i.e. segmentation masks). The script "./FastSurfer/recon/run_recon-surf_with_tracker.py" imports FastSurfer package, sets FreeSurfer variables, and wraps it within tracker processes. 

The FreeSurfer dependency is satisfied by the Singularity container as defined in the "../containers/Singularity_freesurfer_and_fastsurfer.def". To build the singularity image run the following build command: 
``` sudo singularity build ../FastSurfer.sif Singularity_freesurfer_and_fastsurfer.def```

Since the tracker code is evoloving, these repos are NOT installed within the Singularity container. Therefore relative paths are used while importing them from a host directory. 

To run the FastSurfer_recon task on a given subject (sub-000) with a run_tag (test_1), run the following command from the
"watts_up_compute/run_scripts/FastSurfer/recon" directory. 

```singularity exec -B ../../watts_up_compute/proc_output/FastSurfer/recon:/output ../../../../FastSurfer_containers/FastSurfer.sif ./run_recon-surf_with_args.sh sub-000 test_1```

This will run the "run_recon-surf_with_tracker.py" script which will in turn start the subproces with ""./run_fastsurfer.sh". 

Note: "run_recon-surf_with_args_slurm.sh" is an optional script to run this task on HPC using slurm. 