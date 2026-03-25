# Character-Level Name Generation using RNN Variants

This repository implements, trains, and evaluates three different character-level sequence models: **Vanilla RNN**, **BiLSTM**, and **Attention+RNN** using PyTorch. 
It explores how different architectures effectively learn to generate culturally realistic names character by character.

## Directory Structure (Where to find what)

```text
.
├── .gitignore                  # Prevents cache and report output files from being committed
├── data/
│   └── TrainingNames.txt       # The ground-truth dataset containing 1,000 target names
├── dataset.py                  # Data loading, dictionary mappings, and string-to-tensor processing
├── models.py                   # PyTorch architectures (`VanillaRNN`, `BLSTMModel`, `AttentionRNN`)
├── train.py                    # Independent training loop (includes loss tracking & model saving)
├── generate.py                 # Core text sampling engine utilizing multinomial probability 
├── evaluate.py                 # Quantitative metrics (Diversity/Novelty) and analytical plotting
├── utils.py                    # Aggregation logger to ensure clean CLI formatting
├── main.py                     # Main evaluation script: Loads weights, generates samples, exports metrics
├── generate_report.py          # Script extracting outputs into an Overleaf-ready LaTeX Report
├── README.md                   # This instruction overview
└── outputs/                    # Auto-generated directory upon executing scripts
    ├── *_weights.pt            # Pre-trained core weights for each architecture
    ├── generated_samples.txt   # Combined evaluation text output  
    ├── evaluation_metrics.png  # Bar charts assessing novelty vs diversity per model
    ├── length_distribution.png # Distribution density overlap (Real vs Generated sizes)
```

## Requirements & Installation

1. Make sure Python 3.8+ is installed on your operating system.
2. Install the core dependencies utilizing either pure `pip` or by adding them to your virtual environment:

```bash
pip install torch pandas matplotlib seaborn
```

*(Note: Ensure your `torch` distribution supports CUDA if you intend to run this on a GPU, although these lightweight Character-RNNs train completely fine on a standard CPU).*

---

## 1. How to Evaluate Models (Inference Mode)

If you have pre-trained checkpoints (the `.pt` files located securely inside the `outputs/` folder) and solely wish to **generate names**, execute the primary evaluation orchestrator safely:

```bash
python main.py --data data/TrainingNames.txt --samples 100
```

**What it does:** 
1. Avoids training computations natively.
2. Uses temperature scaling (`1.1`) ensuring high stochastic variance.
3. Automatically outputs all generated samples into ONE singular text file explicitly matching constraints: `outputs/generated_samples.txt`.
4. Produces two metric analyses saving dynamically bounding Diversity factors visually: `outputs/evaluation_metrics.png` and `outputs/length_distribution.png`.

---

## 2. How to Train Models

If you modify the architectures in `models.py` or want to re-train the parameters dynamically against the dataset from scratch:

```bash
python train.py --data data/TrainingNames.txt --epochs 60
```

**What it does:**
1. Maps Learning Rate plateaus securely and L2 Weight Decay ($1e-4$) avoiding 100% accuracy memorization locks.
2. Iterates over `Vanilla RNN`, `BLSTM`, and `Attention RNN` individually.
3. Saves resulting finalized PyTorch state dictionaries structurally straight into `outputs/vanilla_rnn_weights.pt`, `outputs/blstm_weights.pt`, and `outputs/attention_rnn_weights.pt`.
