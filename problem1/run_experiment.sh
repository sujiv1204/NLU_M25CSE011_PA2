#!/bin/bash

# Main runner for Word2Vec experiments 
# Usage: ./run_experiment.sh

LOG_FILE="experiment4.log"

echo "Starting Word2Vec Preprocessing and Training Pipeline" | tee $LOG_FILE
date | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE

# 1. Scrape
echo "Running Scraper..." | tee -a $LOG_FILE
python3 -u scraper.py 2>&1 | tee -a $LOG_FILE

# 2. Preprocess
echo "Running Preprocessing..." | tee -a $LOG_FILE
python3 -u preprocess.py 2>&1 | tee -a $LOG_FILE

# 3. Train Skip-gram Configs
echo "Training Config sg1: Skip-gram (dim=50, window=3)..." | tee -a $LOG_FILE
python3 -u train.py --model skipgram --embed_dim 50 --window 3 --neg_samples 5 --epochs 3 --save_name sg1 2>&1 | tee -a $LOG_FILE

echo "Training Config sg2: Skip-gram (dim=300, window=5)..." | tee -a $LOG_FILE
python3 -u train.py --model skipgram --embed_dim 300 --window 5 --neg_samples 5 --epochs 3 --save_name sg2 2>&1 | tee -a $LOG_FILE

echo "Training Config sg3: Skip-gram (dim=100, window=5, neg=10)..." | tee -a $LOG_FILE
python3 -u train.py --model skipgram --embed_dim 100 --window 5 --neg_samples 10 --epochs 3 --save_name sg3 2>&1 | tee -a $LOG_FILE

# 4. Train CBOW Configs
echo "Training Config cbow1: CBOW (dim=50, window=3)..." | tee -a $LOG_FILE
python3 -u train.py --model cbow --embed_dim 50 --window 3 --neg_samples 5 --epochs 3 --save_name cbow1 2>&1 | tee -a $LOG_FILE

echo "Training Config cbow2: CBOW (dim=300, window=5)..." | tee -a $LOG_FILE
python3 -u train.py --model cbow --embed_dim 300 --window 5 --neg_samples 5 --epochs 3 --save_name cbow2 2>&1 | tee -a $LOG_FILE

echo "Training Config cbow3: CBOW (dim=100, window=5, neg=10)..." | tee -a $LOG_FILE
python3 -u train.py --model cbow --embed_dim 100 --window 5 --neg_samples 10 --epochs 3 --save_name cbow3 2>&1 | tee -a $LOG_FILE

# 5. Evaluate and Visualize
echo "Running Evaluation (PCA, t-SNE, Heatmaps, stats)..." | tee -a $LOG_FILE
python3 -u evaluate.py 2>&1 | tee -a $LOG_FILE

echo "----------------------------------------" | tee -a $LOG_FILE
echo "Pipeline execution finished successfully. Output detailed in experiment.log" | tee -a $LOG_FILE
echo "Check data/ directory for visualizations, corpus, and models." | tee -a $LOG_FILE
date | tee -a $LOG_FILE
