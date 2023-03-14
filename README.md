# Medical concept PROBLEM: Polarity, Modality and Temporal Relations

This repository contains my scripts, result reports and visualization for my bachelor thesis "Medical concept PROBLEM: Polarity, Modality and Temporal Relations"

----------


## Examples


---

## Structure

```bash
.
├── result - Result from experiments, each consists of analysis, test and training reports
│   ├── output_mod - in Modality task
│   ├── output_pol - in Polarity task
│   ├── output_univ - in Temporal Relation task with "universal" model
│   │   ├── nosep - sentences in 1 instance are not marked through SEP, take context window size as the maximum length a pair can have
│   │   ├── sep - sentences in 1 instance are marked through SEP, take context window size as the maximum length a pair can have
│   │   ├── sep100 - sentences in 1 instance are marked through SEP, length a pair can have is 100 words left and right
│   │   └── sep60 - sentences in 1 instance are marked through SEP, length a pair can have is 60 words left and right
│   └── output_within - in Temporal Relation task in "within-cases"
├── scripts - scripts to run the whole experiments

└── visualization
```

---

## References

