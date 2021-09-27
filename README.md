# problem-med-aspect
This repository contains my scripts, result reports and visualization for my bachelor thesis "Medical concept PROBLEM: Polarity, Modality and Temporal Relations"

Repository structure is as follows:

```bash
.
├── result - Result from experiments, each consists of analysis, test and training reports
│   ├── output_mod - 
│   │   ├── analysis
│   │   │   ├── ...
│   │   ├── test
│   │   │   ├── ...
│   │   └── train
│   │       ├── ...
│   ├── output_pol
│   │   ├── analysis
│   │   │   ├── ...
│   │   ├── test
│   │   │   ├── ...
│   │   └── train
│   │       ├── ...
│   ├── output_univ
│   │   ├── nosep
│   │   │   ├── test
│   │   │   │   ├── ...
│   │   │   └── train
│   │   │       ├── ...
│   │   ├── sep
│   │   │   ├── analysis
│   │   │   │   ├── ...
│   │   │   ├── test
│   │   │   │   ├── ...
│   │   │   └── train
│   │   │       ├── ...
│   │   ├── sep100
│   │   │   ├── test
│   │   │   │   ├── ..
│   │   │   └── train
│   │   │       ├── ..
│   │   └── sep60
│   │       ├── test
│   │       │   ├── ...
│   │       └── train
│   │           ├── ...
│   └── output_within
│       ├── analysis
│       │   ├── ...
│       ├── test
│       │   ├── ...
│       └── train
│           ├── ...
├── scripts
│   ├── README.md
│   ├── __init__.py
│   ├── eval
│   │   ├── __init__.py
│   │   └── plotters.py
│   ├── main.py
│   ├── nn
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── metrics.py
│   │   ├── models.py
│   │   └── processors.py
│   ├── prep
│   │   ├── __init__.py
│   │   └── reformatter.py
│   ├── requirements.txt
│   └── utils
│       ├── __init__.py
│       ├── commanders.py
│       ├── loaders.py
│       └── savers.py
└── visualization
```
