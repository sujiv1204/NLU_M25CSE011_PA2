# Problem 1: Word Embeddings Generation and Evaluator

This project implements Custom Word2Vec architectures (Skip-gram and CBOW) from scratch. Using NumPy dense matrices, this project scrapes an academic institutional domain (IIT Jodhpur) to parse thousands of PDFs and localized HTML sources, trains custom word embeddings, and subsequently visualizes and compares the resulting semantic properties against industry-standard baselines via `gensim`.

---

## Directory Structure

```text
/data/sujiv1/sujiv/NLU/PA2/problem1/
├── README.md                # Project documentation and execution guide
├── report.tex               # LaTeX assignment report
├── scraper.py               # Web scraping utility parsing HTML and fetching PDFs via BeautifulSoup/pypdf
├── preprocess.py            # Text normalization, stopword filtering, and vocabulary indexing
├── train.py                 # Core NumPy model builder for Custom Scratch Skip-gram and CBOW frameworks
├── evaluate.py              # Statistical testing tools, analogy validators, and dimensionality reducers (PCA/t-SNE)
├── print_embedding.py       # Standalone utility to print the raw 300D embedding array for a specific word
├── run_experiment.sh        # Bash wrapper automating the total run sequence sequentially (with unbuffered logging)
├── experiment4.log          # Master execution trace from an active pipeline run
└── data/                    # Generated asset root
    ├── corpus.txt           # Cleaned flat text body extracted from the scraper
    ├── raw_corpus.txt       # Unfiltered text dumps prior to normalization
    ├── wordcloud.png        # Visualization generated from word frequencies mapping
    ├── word_freq_bar.png    # Top token occurrences graph
    ├── training_loss.png    # Line graph documenting epochs decay
    ├── pca_tsne_skipgram.png      # 2D visual projection of the scratch Skip-gram mapping
    ├── pca_tsne_cbow.png          # 2D visual projection of the scratch CBOW mapping
    ├── pca_tsne_gensim_sg.png     # 2D visual projection of the Gensim Skip-gram baseline mapping
    ├── pca_tsne_gensim_cbow.png   # 2D visual projection of the Gensim CBOW baseline mapping
    ├── models/              # Export directory for binary NumPy weights Arrays (`sg_W.npy`, `cbow_W.npy`, etc)
    └── pdfs/                # Temporary holding directory caching downloaded institutional PDFs
```

---

## Setup & Requirements

First, ensure you are operating within a modern Python 3 environment.

### 1. Install Dependencies
Install the required packages to run all scripts successfully. Ensure you have the scientific stack available:
```bash
pip install requests beautifulsoup4 pypdf numpy matplotlib seaborn wordcloud scikit-learn gensim nltk
```

*(Note: `nltk` might require you to run `import nltk; nltk.download('stopwords')` locally if the cache does not already exist).*

---

## Execution Guide

You can run the entire workflow sequentially with the provided bash automation script, or trigger each stage individually manually. **Ensure you execute standard scripts within the `problem1_copy/` directory.**

### Automated Execution (Recommended)

To run the entire scraper, processor, model trainer, and evaluator sequentially while actively tracking logs:

```bash
bash run_experiment.sh
```
This wrapper effectively ensures output buffers are displayed interactively in the terminal while synchronously pushing to `experiment.log` for debugging and extraction purposes later.

### Manual Execution Sequence

If you prefer to debug independent components, step through the scripts exactly in this order.

#### Step 1: Information Harvesting
```bash
python scraper.py
```
**Functionality:** Bootstraps a custom web crawler prioritizing academic curriculum (UG/PG overviews, research programs) under the `iitj.ac.in` domain structure. It navigates dynamically, downloading associated PDFs into `data/pdfs/` and caching extracted text bodies. It ceases elegantly once hitting the internally defined cap of ~500 documents.

#### Step 2: Semantic Standardization
```bash
python preprocess.py
```
**Functionality:** Normalizes the `raw_corpus.txt`. Systematically lowercases, strips external URLs, and purges standard English stopwords. Crucially, it leaves critical length $\geq 2$ domain acronyms (`"ug"`, `"pg"`, `"cs"`). The script subsequently builds frequency indices and yields `data/corpus.txt` along with data-exploratory visuals (`wordcloud.png`).

#### Step 3: Core Model Construction
```bash
python train.py --epochs 3 --embed_dim 300 --window 5 --neg_samples 5
```
**Functionality:** Begins iterating custom `NumPy` matrices across multiple baseline dimensions (e.g. 50, 100, 300) for both Continuous Bag-of-Words and Skip-gram architectures.
*   The system uses heavily vectorized negative sampling dot products instead of slow Pythonic `for`-loops to aggressively speed up the training computation over the large 690,000+ token text corpus. 
*   **Command Line Flags:** 
    *   `--epochs`: Cycles across the dataset matrix (Default: 3).
    *   `--embed_dim`: Latent parameter dimensions (Default: 300 for main matrix).
    *   `--window`: Context span proximity limit (Default: 5).
    *   `--neg_samples`: Sub-word noise pairs per valid calculation (Default: 5).
*   **Artifacts Generated:** `training_loss.png` displaying the decay, plus raw neural network weights inside `data/models/*.npy`.

#### Step 4: Visual Interpretations & Evaluation
```bash
python evaluate.py
```
**Functionality:** Formats the final qualitative evaluations against the trained arrays.
*   Boots up a competitive `gensim` baseline variant identically parameterized.
*   Searches nearest neighbors arrays (`student`, `phd`, `research`) tracking semantic groupings.
*   Calculates vector arithmetic analogy predictions mapping equations like $Vec(\texttt{ug}) : Vec(\texttt{btech}) :: Vec(\texttt{pg}) : Vec(\texttt{?})$. 
*   Shrinks 300-dimension matrices mapping arrays down into readable 2D planes specifically mapping specific groups (Academics vs Roles vs Departments) leveraging `PCA` against `t-SNE` via `matplotlib` image sets.

#### Step 5: Extracting Specific Word Embeddings
```bash
python print_embedding.py --word exams
```
**Functionality:** A targeted extraction utility designed specifically to load the compiled vocab/model weights and output the literal 300-dimensional continuous floating point vector for a given word directly to the console in comma-separated format. If omitted, `--word` defaults to `engineering`.