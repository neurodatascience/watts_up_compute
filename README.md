# watts_up_compute

## Code repo measuring several compute costs of neuroimaging analyses

## Experiment objectives:
- Measure compute cost metrics:
  - model parameters
  - model FLOPs/MACs
    - [general purpose](http://www.bnikolic.co.uk/blog/python/flops/2019/09/27/python-counting-events.html)
    - [pytorch:ptflops](https://github.com/sovrasov/flops-counter.pytorch)
  - model energy/power consumption
    - [pyJoules](https://github.com/powerapi-ng/pyJoules)
    - [experiment-impact-tracker](https://github.com/Breakend/experiment-impact-tracker)
  - model duration: development (e.g. model traning and selection) and deployment (e.g. model run / inference) times
  - model performance (accuracy / dice score)
- compare
  - hardware: cpu vs gpu
  - datasets: cifar, kaggle (tumor MRI), ukb
  - models
    - Traditional neuroimaging pipelines (e.g. FreeSurfer)
    - DL alternatives (e.g. FastSurfer)

## Repo organization
```
.
├── FastSurfer_experiments
├── figures
├── lib
├── LICENSE
├── notebooks
├── pilot_experiments
├── preproc_pipeline_experiments
├── README.md
├── requirements.txt
├── scripts
├── slurm
└── unit_tests
```

### Notes:  
- All power consumption analyses are performed using this [repo](https://github.com/nikhil153/experiment-impact-tracker)
- FastSurfer_experiments are run using this [repo](https://github.com/nikhil153/FastSurfer)
- Pilot experiments with kaggle dataset are based one this [repo](https://github.com/mateuszbuda/brain-segmentation-pytorch)

### Example analysis
TODO 