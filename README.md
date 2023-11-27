# Medical concept PROBLEM: Polarity, Modality and Temporal Relations

This repository contains my scripts and some visualizations for my bachelor thesis "Medical concept PROBLEM: Polarity, Modality and Temporal Relations".

PROBLEM is considered one of the key medical concepts as it plays a vital role in research for medical information of a patient.
In my thesis, different experiments to extract 3 aspects of this concept were conducted based on the guidelines and annotated data from the **i2b2 2010** and **i2b2 2012** ([Uzuner et al., 2011](https://academic.oup.com/jamia/article/18/5/552/830538), [Sun et al., 2013](https://www.sciencedirect.com/science/article/pii/S1532046413001032)) datasets.

This thesis is to show the effects of different techniques on clinical data. 
This includes **Bi-RNN-based models** (*Bi-GRU*, *Bi-LSTM*), **contextual embeddings** (*BERT*, *FLAIR*),
**domain-specific data the embeddings were pre-trained on** (*ClinicalBERT*, *HunFLAIR*), and **fine-tuning** ([Lin et al., 2018](https://aclanthology.org/2020.bionlp-1.7)).

Experiments were conducted as follows:
- **Bi-GRU**-models were used for *Polarity* and *Modality* Tasks
- **Bi-LSTM**-model was used for *Temporal Relations* Task in terms of entities within sentences (sentence boundaries were observed)
- A "universal model" with the technique of **fine-tuning** different pretrained **BERT**-models were for *Temporal Relations* task with data including cases out-of-sentences (in short relations of entities in the whole text if there are)

Also, to observe the importance of entity markup (how the entities are distinguished when fed into pre-trained Language Model),
based on research from [Lin et al., 2019](https://www.aclweb.org/anthology/W19-1908), *XML* and *non-XML markers* were under experiments.

----------


## Examples

### Polarity and Modality

Each PROBLEM entity has 2 values, **polarity** (NEG, POS) and **modality** (FACTUAL, NONFACTUAL).

<p align="center"><img src="https://github.com/jasmine95dn/problem-med-aspect/blob/main/modpol.png" alt="polmod" height="55%" width="55%"/></p>


### Temporal Relations

All **temporal relations** (BEFORE, AFTER, OVERLAP) towards a pair of **< PROBLEM, OTHER EVENT >** inter- and intrasententially are reported.

<p align="center"><img src="https://github.com/jasmine95dn/problem-med-aspect/blob/main/temprel_figure.png" alt="temprel" height="65%" width="65%"/></p>

---

## Structure

```bash
.
├── scripts - scripts to run the whole experiments
├── __init__.py
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
│   │   ├── elements.py
│   │   ├── embeddings.py
│   │   └── reformatter.py
│   ├── requirements.txt
│   └── utils
│       ├── __init__.py
│       ├── commanders.py
│       ├── loaders.py
│       └── savers.py
└── visualization - some interesting visualizations during analysis models in Temporal Relation Task
```

### Run the scripts

- Clone the repo and redirect to `scripts`
- Install `requirements.txt`
```bash
pip install -r requirements.txt
```
#### Manual

1. **Command Line Execution:**
    
    Run the script with the desired command `main.py`.

   ```bash
   python main.py <command> [<args>]
   ```

   Replace `<command>` with one of the available subcommands (`prep`, `mod`, or `plot`). Also, provide any required or optional arguments specific to the chosen subcommand.

2. **Available Commands:**
   - `prep`: Run preprocessing.
   - `mod`: Run the model.
   - `plot`: Run the plotter.

3. **Subcommand-specific Arguments:**
   - For each subcommand, there are specific arguments you need to provide.

     - **Preprocessing (`prep`):**
       ```bash
       python main.py prep <input_file/directory> -o <output_folder> --dtype <train/test> --compress
       ```
       More options run `python main.py prep -h` for help

     - **Modeling (`mod`):**
       ```bash
       python main.py mod <train_file> <test_file> --labels 0 1 2 --train_prop 0.9 --model_path bert-base-uncased
       ```
       More options run `python main.py mod -h` for help

     - **Plotting (`plot`):**
       ```bash
       python main.py plot <inferred_json_file> --output <output_folder> --ptype cf_matrix --noerror --labels 0 1 2
       ```

       More options run `python main.py plot -h` for help

   
4. **Review Help Information:**
   - If you're unsure about the available commands and their arguments, you can review the help information by running the script with the `-h` or `--help` option:

     ```bash
     python main.py -h
     ```

     This will provide information on the available commands and their respective arguments.

5. **Configurations:**
   - The script relies on configurations (`PrepConfig`, `ModelConfig`, `EmbeddingConfig`, `PlotConfig`) to define the settings for each phase. These configurations are created based on the provided command-line arguments.

6. **Check Results:**
   - Depending on the command, the script may generate outputs such as preprocessed data, trained models, or plots. Check the specified output folders for the results.

Remember to replace placeholder values such as `<input_file/directory>`, `<output_folder>`, `<train_file>`, `<test_file>`, and others with your actual file paths or values.


---

## References
* Chen Lin, Timothy Miller, Dmitriy Dligach, Steven Bethard, and Guergana Savova. *A BERT-based Universal Model for Both Within- and Cross-sentence Clinical Temporal Relation Extraction*. June 2019. [URL](https://www.aclweb.org/anthology/W19-1908)
* Chen Lin, Timothy Miller, Dmitriy Dligach, Farig Sadeque, Steven Bethard, and Guergana Savova. *A BERT-based One-Pass Multi-Task Model for Clinical Temporal Relation Extraction*. July 2020. [URL](https://aclanthology.org/2020.bionlp-1.7)



