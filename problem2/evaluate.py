import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_logger

logger = get_logger()

def evaluate(generated, training_set):
    """Computes basic algorithmic overlaps mapping diversity vs novelty."""
    gen_set = set(n.lower() for n in generated)
    novel = len(gen_set - training_set)
    novelty = novel / len(gen_set) if gen_set else 0
    diversity = len(gen_set) / len(generated) if generated else 0
    return novelty, diversity

def compute_metrics(generated_dict, names):
    """Parses model sets evaluating performance mapping structures."""
    logger.info("Computing Quantitative Evaluation Metrics (Novelty & Diversity)")
    names_set = set(n.lower() for n in names)
    
    eval_results = []
    for model_name, gen_names in generated_dict.items():
        nov, div = evaluate(gen_names, names_set)
        eval_results.append({'Model': model_name, 'Novelty Rate': nov, 'Diversity': div})
        logger.info(f"Metrics [{model_name}] -> Novelty: {nov:.2%} | Diversity: {div:.2%}")
        
    return pd.DataFrame(eval_results)

def plot_comparisons(df_eval, generated_dict, real_names):
    """Renders evaluation plots locally, strictly independent of internal training curves."""
    logger.info("Initializing visual evaluation charts...")
    os.makedirs("outputs", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Performance Comparison Bars
    df_eval.set_index('Model').plot(kind='bar', figsize=(10, 6), color=['#1f77b4', '#ff7f0e'])
    plt.title('Task-2 Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Score (Percentage)', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=0, fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('outputs/evaluation_metrics.png')
    plt.close()
    
    # 2. Model Distribution Density Maps (KDE Overlays)
    plt.figure(figsize=(10, 6))
    for name, gen in generated_dict.items():
        sns.kdeplot([len(n) for n in gen], label=name, bw_adjust=1.5)
    sns.kdeplot([len(n) for n in real_names], label='Real Data', linestyle='--', linewidth=2, color='black')
    
    plt.title('Distribution of Name Lengths')
    plt.xlabel('Character Count')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/length_distribution.png')
    plt.close()
    
    logger.info("Saved analytical evaluation visuals securely into `outputs/`")
