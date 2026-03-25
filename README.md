# CSL 7640: Natural Language Understanding - Programming Assignment 2

## Overview
This assignment focuses on two distinct yet complementary areas of natural language processing: learning semantic representations through word embeddings and generating realistic character-level sequences using recurrent neural networks.

## Problem 1: Word Embeddings from IIT Jodhpur Data
**Objective:** Train and analyze Word2Vec models on institutional textual data to understand semantic relationships.

### Key Contributions
- Collected and preprocessed 460 documents from IIT Jodhpur sources
- Extracted 698,783 tokens with vocabulary size of 27,873 unique words
- Trained custom implementations of CBOW and Skip-gram architectures from scratch using NumPy
- Compared results against industry-standard Gensim baselines
- Performed semantic analysis including nearest neighbor searches and analogy tests
- Visualized embeddings using PCA and t-SNE projections

### Quick Start
```bash
cd problem1
bash run_experiment.sh  # Runs full pipeline: scraping, preprocessing, training, evaluation
```

## Problem 2: Character-Level Name Generation using RNN Variants
**Objective:** Design and compare sequence models for generating culturally realistic Indian names character-by-character.

### Key Contributions
- Generated 1,000 diverse Indian names using language models
- Implemented three distinct architectures from scratch:
  - **Vanilla RNN:** Simple unidirectional recurrent architecture
  - **BLSTM:** Bidirectional Long Short-Term Memory for contextual learning
  - **Attention RNN:** GRU-based model with attention mechanism
- Evaluated models on novelty (generation of unseen names) and diversity metrics
- Analyzed failure modes and generation quality across architectures

### Quick Start
```bash
cd problem2
python main.py --data data/TrainingNames.txt --samples 100  # Generate and evaluate names
```

## Project Structure
```
.
├── problem1/                    # Word embeddings implementation
│   ├── README.md               # Detailed documentation
│   ├── scraper.py              # PDF and HTML parsing
│   ├── preprocess.py           # Text normalization
│   ├── train.py                # Custom Word2Vec implementation
│   ├── evaluate.py             # Semantic analysis and visualization
│   ├── run_experiment.sh       # Automated pipeline
│   ├── experiment4.log         # Execution log with metrics
│   └── data/                   # Outputs and visualizations
│
├── problem2/                    # RNN-based name generation
│   ├── README.md               # Detailed documentation
│   ├── models.py               # Architecture definitions
│   ├── train.py                # Training pipeline
│   ├── generate.py             # Sampling engine
│   ├── evaluate.py             # Metrics computation
│   ├── main.py                 # Main evaluation script
│   ├── dataset.py              # Data loading utilities
│   └── outputs/                # Generated samples and reports
│
└── reports/                     # Final comprehensive report
    └── final_report.tex        # Combined LaTeX report
```

For detailed information on each problem, refer to the individual README files in `problem1/` and `problem2/` directories.
