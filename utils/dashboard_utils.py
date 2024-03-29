# experiment tracker
import sys
import os
import numpy as np
import pandas as pd
from dask import compute, delayed

sys.path.append('../../')
sys.path.append('../')
sys.path.append('../../experiment-impact-tracker/')

from experiment_impact_tracker.data_interface import DataInterface
from experiment_impact_tracker.data_utils import *
from experiment_impact_tracker.data_utils import (load_data_into_frame,
                                                  load_initial_info,
                                                  zip_data_and_info)


def compute_aggregate_power(df, info, PUE, task_epoch_df,use_cuda):
    ''' Aggregates and partitions power consumption based on task interval timpestamps. Allows to see breakdown of power consumptions for different subtasks.
    
    '''

    # time calcs
    exp_end_timestamp = datetime.timestamp(info["experiment_end"])
    
    exp_len = exp_end_timestamp - datetime.timestamp(info["experiment_start"])
    exp_len_hours = exp_len / 3600.0
   
    time_differences = df["timestamp_orig"].diff()
    time_differences[0] = df["timestamp_orig"][0] - datetime.timestamp(
        info["experiment_start"]
    )

    # Add final timestamp and extrapolate last row of power estimates
    time_differences.loc[len(time_differences)] = (
        exp_end_timestamp - df["timestamp_orig"][len(df["timestamp_orig"]) - 1]
    )

    time_differences_in_hours = time_differences / 3600.0

    # rapl calcs
    power_draw_rapl_kw = df["rapl_estimated_attributable_power_draw"] / 1000.0
    power_draw_rapl_kw.loc[len(power_draw_rapl_kw)] = power_draw_rapl_kw.loc[
        len(power_draw_rapl_kw) - 1
    ]

    kw_hr_rapl = (
        np.multiply(time_differences_in_hours, power_draw_rapl_kw)
        if power_draw_rapl_kw is not None
        else None
    )

    # nvidia calcs
    if use_cuda:        
        num_gpus = len(info["gpu_info"])
        nvidia_power_draw_kw = df["nvidia_estimated_attributable_power_draw"] / 1000.0
        nvidia_power_draw_kw.loc[len(nvidia_power_draw_kw)] = nvidia_power_draw_kw.loc[
            len(nvidia_power_draw_kw) - 1
        ]

        # elementwise multiplication and sum
        kw_hr_nvidia = np.multiply(time_differences_in_hours, nvidia_power_draw_kw)

    # apply PUE 
    if use_cuda and (kw_hr_rapl is not None):
        total_power_per_timestep = PUE * (kw_hr_nvidia + kw_hr_rapl)
    elif kw_hr_rapl is not None:
        total_power_per_timestep = PUE * (kw_hr_rapl)
    elif use_cuda:
        total_power_per_timestep = PUE * (kw_hr_nvidia)
    else:
        raise ValueError("Unable to get either GPU or CPU metric.")

    # interpolate power based on timesteps

    # Append last row which implies power draw from last sample extrapolated till the end of experiment
    df.loc[len(df)] = df.loc[len(df) - 1] ## Duplicating last row to match length of total_power_per_timestep
    df.loc[len(df)-1,'timestamp'] = task_epoch_df.loc[len(task_epoch_df)-1,'epoch_timestamp'] #update the timestamp to match end of experiment
    df['total_power_per_timestep'] = total_power_per_timestep.copy()

    task_power_df = pd.DataFrame(columns=['task','power'])
    if total_power_per_timestep is not None:
        # end-to-end power consumption
        task_power_df.loc[0] = ['Experiment', total_power_per_timestep.sum()]

        prev_epoch_power = 0
        print('number of timestamps: {}'.format(len(total_power_per_timestep)))
        
        # power consumption per task
        for i in range(len(task_epoch_df)):
            task = task_epoch_df.loc[i,'task']
            epoch = task_epoch_df.loc[i,'epoch_timestamp']
            epoch_idx = len(df[df['timestamp'] <= epoch])
            current_epoch_power = total_power_per_timestep[:epoch_idx].sum()
            task_power_df.loc[i+1] = [task, current_epoch_power - prev_epoch_power ]
            prev_epoch_power = current_epoch_power

    return df, task_power_df


def get_EIT_tracker_data(logdir, use_cuda, read_flops):
    ''' Fetches experiment impact tracker data from data_interface and separates it into 1) end-to-end experiment df 2) power consumption per sampling epoch df and 3) flops and power consumption per task df
    '''
    # try:
    info = load_initial_info(logdir)
    
    # Get total values from default data interface for the entire experiment 
    data_interface = DataInterface([logdir])
    total_power = data_interface.total_power
    total_carbon = data_interface.kg_carbon
    PUE = data_interface.PUE
    exp_len_hours = data_interface.exp_len_hours

    # Calculate your own sepeartely for each subtask in the experiment
    # impact tracker log
    tracker_df =  load_data_into_frame(logdir)

    if use_cuda:
        power_df = tracker_df[0][['timestamp','rapl_power_draw_absolute','rapl_estimated_attributable_power_draw','nvidia_draw_absolute','nvidia_estimated_attributable_power_draw']].copy()
        power_df.loc[:,'total_attributable_power_draw'] = power_df['rapl_estimated_attributable_power_draw'] + power_df['nvidia_estimated_attributable_power_draw']

    else:
        power_df = tracker_df[0][['timestamp','rapl_power_draw_absolute','rapl_estimated_attributable_power_draw']].copy()
        power_df.loc[:,'total_attributable_power_draw'] = power_df['rapl_estimated_attributable_power_draw']
        
    # start time from 0
    power_df.loc[:,'timestamp_orig'] = power_df['timestamp']
    power_df.loc[:,'timestamp'] = power_df['timestamp'] - power_df['timestamp'][0]

    # papi log
    flops_df = None
    total_duration = 0
    if read_flops:
        compute_flops_csv = logdir + 'compute_costs_flop.csv'
        flops_df = pd.read_csv(compute_flops_csv)
    
        flops_df.loc[:,'start_time'] = flops_df['start_time'] - flops_df['start_time'][0]

        # Aggregate power draws per epoch for each papi context calculation (i.e. setup, axial, aggr etc))
        epoch_power_draw_list = []
        epoch_timestamps = list(flops_df['start_time'].values[1:]) + [flops_df['start_time'].values[-1] + flops_df['duration'].values[-1]]

        task_epoch_df = pd.DataFrame()
        task_epoch_df.loc[:,'task'] = flops_df['task'].values
        task_epoch_df.loc[:,'epoch_timestamp'] = epoch_timestamps
        
        power_df, task_power_df = compute_aggregate_power(power_df, info, PUE, task_epoch_df, use_cuda)
        flops_df = pd.merge(flops_df,task_power_df,on='task',how='left')

        print('total_power sanity check: default: {:6.5f}, calculated: {:6.5f}, {:6.5f}'.format(total_power, task_power_df.loc[0,'power'],power_df['total_power_per_timestep'].sum()))
    
    total_duration_papi = (power_df['timestamp'].values[-1]-power_df['timestamp'].values[0])/3600

    tracker_summary_df = pd.DataFrame(columns=['total_power','total_carbon','PUE','total_duration_papi','total_duration_impact_tracker'])
    tracker_summary_df.loc[0] = [total_power,total_carbon,PUE,total_duration_papi,exp_len_hours]

    return power_df, flops_df, tracker_summary_df

    # except:
        # print(f'No valid experiment impact tracker log found at {logdir}')
        # return None
 

def collate_EIT_tracker_data(tracker_log_dir_list, use_cuda, read_flops):
    ''' Collates EIT tracker data from a set of experiments e.g. FastSurfer results for all subjects
    '''
    power_df_concat = pd.DataFrame()
    flops_df_concat = pd.DataFrame()
    tracker_summary_df_concat = pd.DataFrame()

    values = [delayed(get_EIT_tracker_data)(tracker_log_dir, use_cuda, read_flops) 
              for tracker_log_dir in tracker_log_dir_list]

    tracker_data_list = compute(*values, scheduler='threads',num_workers=4) 

    i = 0
    for td in tracker_data_list:
        if td is not None:
            power_df, flops_df, tracker_summary_df = td
            power_df_concat = power_df_concat.append(power_df)
            flops_df_concat = flops_df_concat.append(flops_df)
            tracker_summary_df_concat = tracker_summary_df_concat.append(tracker_summary_df)

    return tracker_summary_df_concat, flops_df_concat, power_df_concat


def collate_CC_tracker_data(log_dirs):
    ''' Collates CodeCarbon tracker data from a set of experiments e.g. FastSurfer results for all subjects
    '''
    CC_df = pd.DataFrame()
    for log_dir in log_dirs:
        df = pd.read_csv(f'{log_dir}/emissions.csv')
        CC_df = CC_df.append(df)
    return CC_df