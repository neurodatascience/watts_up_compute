#!/bin/bash
#SBATCH --nodes 1          # Request 2 nodes so all resources are in two nodes.
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-00:50
#SBATCH --output=%N-%j.out

module load python/3.6

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index pandas
pip install --no-index torch torchvision

python /home/nikhil/projects/def-jbpoline/nikhil/deep_learning/code/watts_up_compute/scripts/pytorch_test_gpu.py --input_size 512 --n_channels 1 --init_features 64 --max_epochs 10 --output_csv /home/nikhil/projects/def-jbpoline/nikhil/deep_learning/code/watts_up_compute/results/gpu_output.csv

#python pytorch-test.py
