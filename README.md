# watts_up_compute

## Code repo measuring several computational metrics associated with power consumptions of neuroimaging analysis

### Experiment objectives:
- measure compute cost metrics:
  - model parameters
  - model FLOPs/MACs
    - [general purpose](http://www.bnikolic.co.uk/blog/python/flops/2019/09/27/python-counting-events.html)
    - [pytorch:ptflops](https://github.com/sovrasov/flops-counter.pytorch)
  - train and inference times
  - train and inference energy [pyjoules](https://github.com/powerapi-ng/pyJoules)
  - model performance (accuracy / dice score)
- compare
  - hardware: cpu vs gpu
  - datasets: cifar, kaggle (tumor MRI)
  - models
    - classification vs segmentation (unet)
    - model sizes 
    - input sizes

- Other baselines
  - profile compute costs of feature engineering (e.g. [fMRI tangent space](https://nilearn.github.io/auto_examples/03_connectivity/plot_group_level_connectivity.html#sphx-glr-auto-examples-03-connectivity-plot-group-level-connectivity-py)) 
