# Medical concept PROBLEM: Polarity, Modality and Temporal Relations

This repository contains my scripts and some visualizations for my bachelor thesis "Medical concept PROBLEM: Polarity, Modality and Temporal Relations".

PROBLEM is considered one of the key medical concepts as it plays a vital role in research for medical information of a patient.
In my thesis, different experiments to extract 3 aspects of this concept were conducted based on the guidelines and annotated data from the i2b2 2010 and i2b2 2012 ((Uzuner et al., 2011)[https://academic.oup.com/jamia/article/18/5/552/830538], (Sun et al., 2013)[https://www.sciencedirect.com/science/article/pii/S1532046413001032]) datasets.

This thesis is to show the effects of different techniques on clinical data. 
This includes **Bi-RNN-based models** (*Bi-GRU*, *Bi-LSTM*), **contextual embeddings** (*BERT*, *FLAIR*),
**domain-specific data the embeddings were pre-trained on** (*ClinicalBERT*, *HunFLAIR*), and **fine-tuning** (Lin et al., 2018)[https://aclanthology.org/2020.bionlp-1.7].
Also, to observe the importance of entity markup (how the entities are distinguished when fed into pre-trained Language Model),
based on research from (Lin et al., 2019)[https://www.aclweb.org/anthology/W19-1908], *XML* and *non-XML markers* were under experiments.

----------


## Examples

### Polarity and Modality

Each PROBLEM entity has 2 values, **polarity** (NEG, POS) and **modality** (FACTUAL, NONFACTUAL).

![polmod](https://github.com/jasmine95dn/problem-med-aspect/blob/main/modpol.png)


### Temporal Relations

All **temporal relations** (BEFORE, AFTER, OVERLAP) towards a pair of **< PROBLEM, OTHER EVENT >** inter- and intrasententially are reported.

![temprel](https://github.com/jasmine95dn/problem-med-aspect/blob/main/temprel_figure.png)

---

## Structure

```bash
.
├── scripts - scripts to run the whole experiments
└── visualization - some interesting visualizations during analysis models in Temporal Relation Task
```

---

## References
* Chen Lin, Timothy Miller, Dmitriy Dligach, Steven Bethard, and Guergana Savova. A BERT-based Universal Model for Both Within- and Cross-sentence Clinical Temporal Relation Extraction. (URL)[https://www.aclweb.org/anthology/W19-1908]
* Chen Lin, Timothy Miller, Dmitriy Dligach, Farig Sadeque, Steven Bethard, and Guergana Savova. A BERT-based One-Pass Multi-Task Model for Clinical Temporal Relation Extraction. (URL)[https://aclanthology.org/2020.bionlp-1.7]



