import sys
import tempfile
import torch

sys.path.append('../../experiment-impact-tracker/')

from experiment_impact_tracker.compute_tracker import ImpactTracker

from codecarbon import EmissionsTracker, OfflineEmissionsTracker

import logging
logging.basicConfig(level="DEBUG")

class Experiment:
    def __init__(self):
        device = torch.device("cpu")
        # device = torch.device('cuda') # Uncomment this to run on GPU

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


def my_experiment(log_dir, region_coords=None) -> None:
    # Init tracker with log path
    tracker = ImpactTracker(log_dir,region_coords)
    # Start tracker in a separate process
    tracker.launch_impact_monitor()

    print(tracker.initial_info['region'])
    print('')
    print(tracker.initial_info['region_carbon_intensity_estimate'])

    tracker_CC = EmissionsTracker(output_dir=log_dir)
    tracker_CC.start()

    exp = Experiment()

    for t in range(1000):
        if t % 100 == 0:
            print(f"Pass: {t}")
            # Optional. Adding this will ensure that your experiment stops if impact tracker throws an exception and exit.
            tracker.get_latest_info_and_check_for_errors()
        exp.train()

    tracker_CC.stop()
    print(f"Please find your experiment logs in: {log_dir}")


if __name__ == "__main__":

    # Example latitude and longitude by city
    # MTL:(45.4972159,-73.6103642), NYC:(40.741895,-73.989308), Pune:(18.521428,73.8544541), Paris:(48.8566969,2.3514616)
    REGION_COORDS =  (45.4972159,-73.6103642)  
    log_dir = '../tmp/experiment-impact-tracker/'
    my_experiment(log_dir)