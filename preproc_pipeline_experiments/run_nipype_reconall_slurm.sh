#!/bin/bash
#SBATCH --account=def-jbpoline
#SBATCH --nodes 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-23:00
#SBATCH --output=logs/%N-%j.out
#SBATCH --array=1-73

echo "Starting task $SLURM_ARRAY_TASK_ID"
SUBJECT_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" subject_ids.txt)
module load singularity/3.6

singularity exec --overlay /project/rpp-aevans-ab/neurohub/ukbb/imaging/neurohub_ukbb_t1_ses2_0_bids.squashfs:ro ../../FreeSurfer_tracker.simg ./run_nipype.sh sub-$SUBJECT_ID
