
#!/bin/bash
#SBATCH --nodes 1          # Request 1 nodes so all resources are in two nodes.
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-02:50
#SBATCH --output=%N-%j.out
#SBATCH --array=1-73

echo "Starting task $SLURM_ARRAY_TASK_ID"
subject_id=$(sed -n "${SLURM_ARRAY_TASK_ID}p" subject_ids.txt)
run_id=$1

module load python/3.8

if [ -z $gpu]; then
    echo "using gpu"

    singularity exec --nv --overlay /project/rpp-aevans-ab/neurohub/ukbb/imaging/neurohub_ukbb_t1_ses2_0_bids.squashfs:ro \
    -B /home/nikhil/green_compute/ukb_pilot/fastsurfer/recon-surf/prune_50:/output \
    ../../FastSurfer.sif \
    ./run_cnn_eval.sh sub-$subject_id $run_id "hpc"
else
    echo "using cpu"

    singularity exec --overlay /project/rpp-aevans-ab/neurohub/ukbb/imaging/neurohub_ukbb_t1_ses2_0_bids.squashfs:ro \
    -B /home/nikhil/green_compute/ukb_pilot/fastsurfer/recon-surf/prune_50:/output \
    ../../FastSurfer.sif \
    ./run_cnn_eval.sh sub-$subject_id $run_id "hpc"
fi
