# problem-med-aspect
This repository contains my scripts, result reports and visualization for my bachelor thesis "Medical concept PROBLEM: Polarity, Modality and Temporal Relations"

Repository structure is as follows:

```bash
.
├── result - Result from experiments, each consists of analysis, test and training reports
│   ├── output_mod - in Modality task
│   │   ├── analysis
│   │   │   ├── ...
│   │   ├── test
│   │   │   ├── ...
│   │   └── train
│   │       ├── ...
│   ├── output_pol - in Polarity task
│   │   ├── analysis
│   │   │   ├── ...
│   │   ├── test
│   │   │   ├── ...
│   │   └── train
│   │       ├── ...
│   ├── output_univ - in Temporal Relation task with "universal" model
│   │   ├── nosep - sentences in 1 instance are not marked through SEP, take context window size as the maximum length a pair can have
│   │   │   ├── test
│   │   │   │   ├── ...
│   │   │   └── train
│   │   │       ├── ...
│   │   ├── sep - 
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
│   └── output_within - in Temporal Relation task in "within-cases"
│       ├── analysis
│       │   ├── ...
│       ├── test
│       │   ├── ...
│       └── train
│           ├── ...
├── scripts - scripts to run the whole experiments
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

