#!/bin/bash
#SBATCH --account=def-jbpoline
#SBATCH --nodes 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-23:00
#SBATCH --output=logs/%N-%j.out
#SBATCH --array=1-73


echo "Starting task $SLURM_ARRAY_TASK_ID"
subject_id=$(sed -n "${SLURM_ARRAY_TASK_ID}p" subject_ids.txt)
run_id=$1

module load singularity/3.8

singularity exec --overlay /project/rpp-aevans-ab/neurohub/ukbb/imaging/neurohub_ukbb_t1_ses2_0_bids.squashfs:ro \
            ../../FastSurfer.sif \
            ./run_nipype_reconall.sh sub-$subject_id $run_id "hpc"
