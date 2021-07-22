import sys
import tempfile
import torch
import pandas as pd
import numpy as np

sys.path.append('../../experiment-impact-tracker/')

from experiment_impact_tracker.compute_tracker import ImpactTracker
from experiment_impact_tracker.data_interface import DataInterface

from codecarbon import EmissionsTracker, OfflineEmissionsTracker

from carbontracker.tracker import CarbonTracker
from carbontracker import parser

import logging
logging.basicConfig(level="DEBUG")

class Experiment:
    def __init__(self):
        device = torch.device("cpu")

        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        N, D_in, H, D_out = 1024, 10000, 1000, 100

        # Create random input and output data
        self.x = torch.randn(N, D_in, device=device)
        self.y = torch.randn(N, D_out, device=device)

        # Randomly initialize weights
        self.w1 = torch.randn(D_in, H, device=device)
        self.w2 = torch.randn(H, D_out, device=device)
        self.learning_rate = 1e-6

    def train(self):
        # Forward pass: compute predicted y
        h = self.x.mm(self.w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(self.w2)

        # Compute and print loss; loss is a scalar, and is stored in a PyTorch Tensor
        # of shape (); we can get its value as a Python number with loss.item().
        loss = (y_pred - self.y).pow(2).sum()

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - self.y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(self.w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = self.x.t().mm(grad_h)

        # Update weights using gradient descent
        self.w1 -= self.learning_rate * grad_w1
        self.w2 -= self.learning_rate * grad_w2


def my_experiment(log_dir, n_steps=100) -> None:

    ## code-carbon tracker
    tracker_CC = EmissionsTracker(output_dir=log_dir) #measure_power_secs=1 (also tried)
    tracker_CC.start()

    ## exp-impact-tracker
    tracker_EIT = ImpactTracker(log_dir)
    tracker_EIT.launch_impact_monitor()

    ## CarbonTracker
    tracker_CT = CarbonTracker(epochs=1, log_dir=log_dir)
    tracker_CT.epoch_start()
        
    exp = Experiment()

    for t in range(n_steps):        
        if t % (n_steps//10) == 0:
            print(f"Pass: {t}")
            # Optional. Adding this will ensure that your experiment stops if impact tracker throws an exception and exit.
            tracker_EIT.get_latest_info_and_check_for_errors()
        exp.train()

    tracker_CC.stop()
    
    tracker_CT.epoch_end()

    print(f"Please find your experiment logs in: {log_dir}")


if __name__ == "__main__":

    log_dir = tempfile.mkdtemp()
    n_steps = 1000

    my_experiment(log_dir, n_steps=n_steps)

    # exp-impact-tracker logs
    data_interface1 = DataInterface([log_dir])
    EIT_energy_consumed = data_interface1.total_power
    EIT_duration = data_interface1.exp_len_hours * 3600

    # code-carbon logs
    CC_df = pd.read_csv(f'{log_dir}/emissions.csv')
    CC_duration = CC_df['duration'].values[0]
    CC_energy_consumed = CC_df['energy_consumed'].values[0]

    # CarbonTracker logs
    logs = parser.parse_all_logs(log_dir=log_dir)   
    first_log = logs[0]
    CT_duration = first_log['actual']['duration (s)']
    CT_energy_consumed = first_log['actual']['energy (kWh)']
    
    print('')
    print(f'runtimes (sec): \nexp-impact-tracker: {EIT_duration:.3f}\ncode-carbon: {CC_duration:.3f}\ncarbon-tracker: {CT_duration:.3f}')
    print('')
    print(f'total power (kwh):\nexp-impact-tracker: {EIT_energy_consumed:.6f}\nenergy consumed from code-carbon: {CC_energy_consumed:.6f}\ncarbon-tracker: {CT_energy_consumed:.6f}')



