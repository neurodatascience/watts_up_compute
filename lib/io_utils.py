import os
import numpy as np
import pandas as pd
import pickle
import re

def read_joules(f,device):
    ''' Reads a joules trace csv generated by run_experiment.py'''
    joules_df = pd.read_csv(f)
    jcols = joules_df.columns

    regex = re.compile('package_.')
    package_cols = [string for string in jcols if re.match(regex, string)]

    regex = re.compile('dram_.')
    dram_cols = [string for string in jcols if re.match(regex, string)]
    
    joules_df['package_total'] = joules_df[package_cols].sum(axis=1)
    joules_df['dram_total'] = joules_df[dram_cols].sum(axis=1)

    if device == 'cuda':
        joules_df['nvidia_total'] = joules_df['nvidia_gpu_0']
    else:
        joules_df['nvidia_total'] = 0

    joules_df['process_total'] = joules_df['package_total'] + joules_df['dram_total'] + joules_df['nvidia_total']
    joules_df['device'] = device

    return joules_df