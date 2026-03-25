import os
import argparse
import torch
from utils import get_logger
from dataset import load_and_prepare_data
from models import VanillaRNN, BLSTMModel, AttentionRNN
from generate import generate_names
from evaluate import compute_metrics, plot_comparisons

def main():
    parser = argparse.ArgumentParser(description="Run Evaluation Models on Checkpoints")
    parser.add_argument("--data", type=str, default="data/TrainingNames.txt", help="Target input data structure path.")
    parser.add_argument("--samples", type=int, default=100, help="Amount of outputs generated per model.")
    args = parser.parse_args()

    logger = get_logger()
    logger.info("Commencing Evaluation Execution Block (Generation & Computation)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Dataset Dictionary Hook
    names, data, vocab, idx_to_char, vocab_size = load_and_prepare_data(args.data)
    
    # 2. Architectures Setup Hook mapping identical default configurations
    hidden_dim = 128
    num_layers = 1
    emb_dim = 64
    
    models = {
        'Vanilla RNN': VanillaRNN(vocab_size, emb_dim, hidden_dim, num_layers).to(device),
        'BLSTM': BLSTMModel(vocab_size, emb_dim, hidden_dim, num_layers).to(device),
        'Attention RNN': AttentionRNN(vocab_size, emb_dim, hidden_dim, num_layers).to(device)
    }
    
    generated_dict = {}
    combined_output_file = "outputs/generated_samples.txt"
    
    # 3. Read Checkpoints and Evaluate Synthetically Into Single Txt File
    with open(combined_output_file, "w", encoding="utf-8") as f_out:
        for name, model in models.items():
            weight_path = f"outputs/{name.replace(' ', '_').lower()}_weights.pt"
            
            if not os.path.exists(weight_path):
                logger.error(f"Cannot evaluate {name}; its weights file {weight_path} does not exist. (If expected, please run train.py first)")
                continue
                
            model.load_state_dict(torch.load(weight_path, map_location=device))
            logger.info(f"Memory Checkpoint mapped successfully for -> {name}")
            
            # Predict
            gen_names = generate_names(model, vocab, idx_to_char, device, n=args.samples, temp=1.1)
            generated_dict[name] = gen_names
            
            # Document iteratively into central trace
            f_out.write(f"--- {name} Generated Output Samples ---\n")
            f_out.write("\n".join(gen_names) + "\n\n")
            logger.info(f"Flushed generation stream successfully for {name}")

    # 4. Plot Standalone Metrics (Diversity/Realism vs Density mappings)
    if generated_dict:
        logger.info("Tracing evaluation logic against original target indices.")
        df_eval = compute_metrics(generated_dict, names)
        plot_comparisons(df_eval, generated_dict, names)
        logger.info(f"System evaluation exited securely updating {combined_output_file} and /outputs/ plotting graphs.")

if __name__ == '__main__':
    main()