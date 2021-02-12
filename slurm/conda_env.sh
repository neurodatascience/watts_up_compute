#!/bin/bash

env_path=$1
echo "env path:"$env_path

conda create --prefix env_path python=3.7
source activate env_path
conda install -c pytorch pytorch 
conda install pandas
conda install -c conda-forge py3nvml
conda install -c conda-forge py-cpuinfo 

pip install pyJoules
pip install ptflops

echo "Conda env created"