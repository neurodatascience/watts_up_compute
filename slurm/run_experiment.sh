#!/bin/bash
#SBATCH --nodes 1          # Request 1 nodes so all resources are in two nodes.
#SBATCH --gres=gpu:v100l:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-00:50
#SBATCH --output=%N-%j.out

module load python/3.7
source /home/nikhil/projects/def-jbpoline/nikhil/deep_learning/code/env/bin/activate

python /home/nikhil/projects/def-jbpoline/nikhil/deep_learning/code/watts_up_compute/ml/run_experiment.py
