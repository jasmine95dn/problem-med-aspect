# Medical concept PROBLEM: Polarity, Modality and Temporal Relations

This repository contains my scripts and some visualizations for my bachelor thesis "Medical concept PROBLEM: Polarity, Modality and Temporal Relations".

PROBLEM is considered one of the key medical concepts as it plays a vital role in research for medical information of a patient.
In my thesis, different experiments to extract 3 aspects of this concept were conducted based on the guidelines and annotated data from the i2b2 2010 and i2b2 2012 () datasets.

This thesis is to show the effects of different techniques on clinical data. 
This includes **Bi-RNN-based models** (*Bi-GRU*, *Bi-LSTM*), **contextual embeddings** (*BERT*, *FLAIR*),
**domain-specific data the embeddings were pre-trained on** (*ClinicalBERT*, *HunFLAIR*), and **fine-tuning**.
Also, to observe the importance of entity markup (how the entities are distinguished when fed into pre-trained Language Model),
based on research from Li et al. 2019, *XML* and *non-XML markers* were under experiments.

----------


## Examples

### Polarity and Modality

Each PROBLEM entity has 2 values, **polarity** (NEG, POS) and **modality** (FACTUAL, NONFACTUAL).




### Temporal Relations

All **temporal relations** (BEFORE, AFTER, OVERLAP) towards a pair of **< PROBLEM, OTHER EVENT >** are reported.

---

## Structure

```bash
.
├── scripts - scripts to run the whole experiments
└── visualization - some interesting visualizations during analysis models in Temporal Relation Task
```

---

## References




