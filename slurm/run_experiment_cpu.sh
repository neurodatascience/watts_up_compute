#!/bin/bash
#SBATCH --account=def-jbpoline
#SBATCH --nodes 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-02:00
#SBATCH --output=logs/%N-%j.out

module load python/3.7
source /home/nikhil/projects/def-jbpoline/nikhil/deep_learning/code/env/bin/activate

python /home/nikhil/projects/def-jbpoline/nikhil/deep_learning/code/watts_up_compute/ml/run_experiment.py
