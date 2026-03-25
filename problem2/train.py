import os
import argparse
import torch
import torch.nn as nn
from utils import get_logger
from dataset import load_and_prepare_data
from models import VanillaRNN, BLSTMModel, AttentionRNN, log_model_stats

logger = get_logger()

def train_model(model, data, vocab_pad_idx, vocab_size, device, epochs=60, batch_size=128, lr=0.003):
    """
    Iterates batches over the sequence generation model applying L2 Decay and Plateau Schedulers.
    """
    model = model.to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_pad_idx)
    
    n = data.size(0)
    loss_history = []
    acc_history = []
    
    for ep in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total_chars = 0
        batches = 0
        
        idx = torch.randperm(n)
        
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            batch = data[batch_idx].to(device)
            
            x = batch[:, :-1]
            y = batch[:, 1:]
            
            logits, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.reshape(-1))
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            
            total_loss += loss.item()
            batches += 1
            
            preds = logits.argmax(dim=-1)
            mask = (y != vocab_pad_idx)
            correct += (preds[mask] == y[mask]).sum().item()
            total_chars += mask.sum().item()
            
        epoch_loss = total_loss / batches
        epoch_acc = correct / total_chars
        loss_history.append(epoch_loss)
        acc_history.append(epoch_acc)
        
        scheduler.step(epoch_loss)
        
        if (ep + 1) % 10 == 0 or ep == 0:
            logger.info(f"Progress | Epoch {ep+1:02d}/{epochs} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2%} | Active LR: {opt.param_groups[0]['lr']:.6f}")
            
    return model, loss_history, acc_history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standalone Training Execution Pipeline")
    parser.add_argument("--data", type=str, default="data/TrainingNames.txt", help="Path to input data")
    parser.add_argument("--epochs", type=int, default=60, help="Epoch count")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    names, data, vocab, idx_to_char, vocab_size = load_and_prepare_data(args.data)
    vocab_pad_idx = vocab['<P>']

    hidden_dim = 128
    num_layers = 1
    emb_dim = 64
    hyper_lr = 0.003
    
    models = {
        'Vanilla RNN': VanillaRNN(vocab_size, emb_dim, hidden_dim, num_layers).to(device),
        'BLSTM': BLSTMModel(vocab_size, emb_dim, hidden_dim, num_layers).to(device),
        'Attention RNN': AttentionRNN(vocab_size, emb_dim, hidden_dim, num_layers).to(device)
    }

    for name, model in models.items():
        logger.info(f"Beginning optimization pass for model block -> {name}")
        model, _, _ = train_model(
            model=model, data=data, vocab_pad_idx=vocab_pad_idx, 
            vocab_size=vocab_size, device=device, epochs=args.epochs, lr=hyper_lr
        )
        torch.save(model.state_dict(), f"outputs/{name.replace(' ', '_').lower()}_weights.pt")
        logger.info(f"Persisted structural weights successfully for {name}")
