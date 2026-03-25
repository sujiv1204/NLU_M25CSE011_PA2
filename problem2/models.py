import torch
import torch.nn as nn
from utils import get_logger

logger = get_logger()

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        
        # Explicit dropouts deployed since num_layers=1 bypasses standard PyTorch RNN dropout
        self.embedding_dropout = nn.Dropout(dropout)
        self.rnn = nn.RNN(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, h=None):
        emb = self.embedding_dropout(self.embed(x))
        out, h = self.rnn(emb, h)
        out = self.output_dropout(out)
        return self.fc(out), h

class BLSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        self.blstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=num_layers, 
            batch_first=True, bidirectional=True
        )
        
        self.output_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, x, h=None):
        emb = self.embedding_dropout(self.embed(x))
        seq_len = x.size(1)
        
        # Fix for Data Leakage: Force autoregressive stepping during training
        if self.training and seq_len > 1:
            outputs = []
            for t in range(seq_len):
                out_t, _ = self.blstm(emb[:, :t+1, :])
                outputs.append(out_t[:, -1:, :])
            out = torch.cat(outputs, dim=1)
        else:
            out, h = self.blstm(emb, h)
            
        out = self.output_dropout(out)
        return self.fc(out), h

class AttentionRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        
        # Bahdanau Style Additive Attention parameters
        self.attn_w = nn.Linear(hidden_dim, hidden_dim)
        self.attn_u = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1)
        
        self.output_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, x, h=None):
        emb = self.embedding_dropout(self.embed(x))
        out, h = self.gru(emb, h)
        
        query = self.attn_w(out)
        keys = self.attn_u(out)
        scores = self.attn_v(torch.tanh(query.unsqueeze(2) + keys.unsqueeze(1))).squeeze(-1)
        
        # Establish strict causal dependency allowing only historical attention
        mask = torch.triu(torch.ones(scores.size(-2), scores.size(-1)), diagonal=1).bool().to(out.device)
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(attn_weights, out)
        
        combined = torch.cat([out, ctx], dim=-1)
        combined = self.output_dropout(combined)
        return self.fc(combined), h

def count_parameters(model):
    """Calculates and returns the total count of trainable gradients."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_model_stats(model, name, desc, hidden_dim, num_layers, lr):
    """Logs the network architecture topology and meta-parameter count cleanly."""
    params = count_parameters(model)
    size_mb = params * 4 / (1024**2)
    
    logger.info(f"--- Model Spec: {name} ---")
    logger.info(f"Description: {desc}")
    logger.info(f"Hyperparameters Set: Hidden Size={hidden_dim}, Layers={num_layers}, LR={lr}")
    logger.info(f"Trainable memory parameters: {params:,} (Estimated Memory Footprint: {size_mb:.4f} MB)")
    logger.info(f"Structural Pipeline Summary:\n{str(model)}\n")
