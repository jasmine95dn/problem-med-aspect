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
│       │   ├── inferTEMP.txt
│       │   ├── tables.json
│       │   └── test.json
│       ├── test
│       │   ├── bertflair9_choose.txt
│       │   ├── berthunflair8.txt
│       │   ├── bertlarge10_choose.txt
│       │   ├── bertlarge8.txt
│       │   ├── clinicalbert10.txt
│       │   ├── clinicalbert7_choose.txt
│       │   ├── clinicalbert_bl10.txt
│       │   ├── clinicalbert_bl8_choose.txt
│       │   ├── clinicalbertflair10.txt
│       │   ├── clinicalberthunflair10_choose.txt
│       │   ├── clinicalberthunflair8.txt
│       │   ├── flair10_choose.txt
│       │   └── hunflair10.txt
│       └── train
│           ├── bertflair_train.txt
│           ├── berthunflair_train.txt
│           ├── bertlarge_train_20eps.txt
│           ├── bertwwm_train.txt
│           ├── clinicalbert_train.txt
│           ├── clinicalbert_train_bl.txt
│           ├── clinicalbertflair_train.txt
│           ├── clinicalberthunflair_train.txt
│           ├── flair_train.txt
│           └── hunflair_train.txt
├── scripts
│   ├── README.md
│   ├── __init__.py
│   ├── eval
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   └── plotters.cpython-38.pyc
│   │   ├── evaluators.py
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
│   │   ├── elements.py
│   │   ├── embeddings.py
│   │   └── reformatter.py
│   ├── requirements.txt
│   ├── separate
│   │   ├── another_finetune.py
│   │   ├── bilstm.py
│   │   ├── bilstm_eval.py
│   │   ├── bilstm_flair.py
│   │   ├── bilstm_flair_cb.py
│   │   ├── bilstm_flair_cb_test.py
│   │   ├── bilstm_flair_eval.py
│   │   ├── bilstm_infer.py
│   │   ├── evaluate.py
│   │   ├── finetune.py
│   │   ├── gru.py
│   │   ├── gru_eval.py
│   │   ├── gru_flair.py
│   │   ├── gru_flair_cb_cond.py
│   │   ├── gru_flair_clinicalbert.py
│   │   ├── gru_flair_clinicalbert_eval.py
│   │   ├── gru_flair_clinicalbert_infer.py
│   │   ├── gru_flair_continue.py
│   │   ├── gru_flair_mod_extra_test.py
│   │   ├── gru_flair_test.py
│   │   ├── gru_infer.py
│   │   ├── infer.py
│   │   ├── mod_analysis
│   │   │   ├── factual_test.json
│   │   │   ├── inferNONFACTUAL.txt
│   │   │   └── nonfactual_test.json
│   │   ├── output
│   │   │   ├── finetune_output_bert-base_result.txt
│   │   │   └── finetune_output_bert-base_result_self.txt
│   │   ├── pol_analysis
│   │   │   ├── POS_wrong.json
│   │   │   ├── inferNEG.txt
│   │   │   ├── inferPOS.txt
│   │   │   ├── neg_test.json
│   │   │   ├── neg_wrong.json
│   │   │   └── pos_test.json
│   │   └── temp_within_analysis
│   │       ├── inferTEMP.txt
│   │       └── tables.json
│   └── utils
│       ├── __init__.py
│       ├── commanders.py
│       ├── loaders.py
│       ├── logger.py
│       └── savers.py
└── visualization
```
