#!/bin/bash

env_path=$1
echo "env path:"$env_path
module load python/3.7

virtualenv --no-download $env_path
source $env_path/bin/activate

pip install --no-index --upgrade pip
pip install --no-index pandas
pip install --no-index torch torchvision
pip install ptflops
pip install pyJoules
pip install py3nvml
pip install py-cpuinfo

echo "env creation complete"
