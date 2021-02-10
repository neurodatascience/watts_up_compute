#!/bin/bash

env_path=$1
echo "env path:"$env_path
module load python/3.6

virtualenv --no-download $env_path
source $env_path/bin/activate

pip install --no-index --upgrade pip
pip install --no-index pandas
pip install --no-index torch torchvision
pip install --no-index ptflops
pip install --no-index pyJoules
pip install --no-index cpuinfo

echo "env creation complete"