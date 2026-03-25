import os
import argparse
import pickle
import numpy as np
import random

def get_neg_sample_probs(word2idx, word_counts):
    # Word2Vec authors recommended raising word frequencies to the 3/4 power 
    # This purposefully dampens extremely frequent words (like 'the', 'is')
    # to increase the probability of sampling rarer, more meaningful words.
    vocab_size = len(word2idx)
    freqs = np.zeros(vocab_size)
    for word, idx in word2idx.items():
        freqs[idx] = word_counts[word]
    freqs = np.power(freqs, 0.75)
    return freqs / freqs.sum()

def get_neg_samples(exclude_idxs, k, neg_probs, vocab_size):
    # Retrieve k random noise words utilizing the weighted probability array
    return np.random.choice(vocab_size, size=k, p=neg_probs)

def sigmoid(x):
    # Standard activation function. 
    # Clipping bounds (-10 to 10) to prevent overflow/Runtime warnings in np.exp()
    x = np.clip(x, -10, 10)
    return 1.0 / (1.0 + np.exp(-x))

def train_skipgram(sentences, word2idx, neg_probs, embed_dim, window, neg_samples, epochs, lr):
    vocab_size = len(word2idx)
    W_in = (np.random.rand(vocab_size, embed_dim) - 0.5) / embed_dim
    W_out = np.zeros((vocab_size, embed_dim))
    
    print(f"Training skipgram: dim={embed_dim}, window={window}, neg={neg_samples}")
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        shuffled = sentences.copy()
        random.shuffle(shuffled)
        
        for sent in shuffled:
            indices = [word2idx[w] for w in sent if w in word2idx]
            if len(indices) < 2: continue
            
            for i, center_idx in enumerate(indices):
                start = max(0, i - window)
                end = min(len(indices), i + window + 1)
                center_vec = W_in[center_idx]
                
                for j in range(start, end):
                    if i == j: continue
                    context_idx = indices[j]
                    
                    # Positive
                    out_vec = W_out[context_idx]
                    dot = np.dot(center_vec, out_vec)
                    sig = sigmoid(dot)
                    
                    grad_out = (sig - 1) * center_vec
                    grad_in = (sig - 1) * out_vec
                    total_loss += -np.log(sig + 1e-10)
                    
                    # Negative
                    neg_idxs = get_neg_samples([context_idx], neg_samples, neg_probs, vocab_size)
                    for neg_idx in neg_idxs:
                        neg_vec = W_out[neg_idx]
                        dot_neg = np.dot(center_vec, neg_vec)
                        sig_neg = sigmoid(dot_neg)
                        
                        grad_out_neg = sig_neg * center_vec
                        grad_in += sig_neg * neg_vec
                        total_loss += -np.log(1 - sig_neg + 1e-10)
                        
                        W_out[neg_idx] -= lr * grad_out_neg
                        
                    W_out[context_idx] -= lr * grad_out
                    W_in[center_idx] -= lr * grad_in
                    count += 1
                    
        avg_loss = total_loss / max(count, 1)
        loss_history.append(avg_loss)
        print(f"  epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}", flush=True)
        lr *= 0.95
        
    return W_in, W_out, loss_history

def train_cbow(sentences, word2idx, neg_probs, embed_dim, window, neg_samples, epochs, lr):
    # CBOW tries to predict the center word by taking an average of all surrounding context tokens.
    # Because of this averaging, the gradients bleed across classes, which makes it perform slightly 
    # worse than Skip-gram for capturing nuanced local semantics in our tests.
    vocab_size = len(word2idx)
    W_in = (np.random.rand(vocab_size, embed_dim) - 0.5) / embed_dim
    W_out = np.zeros((vocab_size, embed_dim))
    
    print(f"Training CBOW: dim={embed_dim}, window={window}, neg={neg_samples}")
    loss_history = []
    
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        shuffled = sentences.copy()
        
        # Shuffling prevents the model from memorizing the specific document iteration order
        random.shuffle(shuffled)
        
        for sent in shuffled:
            indices = [word2idx[w] for w in sent if w in word2idx]
            if len(indices) < 2: continue
            
            for i, center_idx in enumerate(indices):
                # Dynamically set sliding window bounds protecting against indexing outside array limits
                start = max(0, i - window)
                end = min(len(indices), i + window + 1)
                context_idxs = [indices[j] for j in range(start, end) if j != i]
                if not context_idxs: continue
                
                # Average pooling operation over context vectors is mathematically what differentiates CBOW
                context_mean = np.mean(W_in[context_idxs], axis=0)
                center_out = W_out[center_idx]
                
                # Positive
                dot = np.dot(context_mean, center_out)
                sig = sigmoid(dot)
                
                grad_out = (sig - 1) * context_mean
                grad_in = (sig - 1) * center_out
                total_loss += -np.log(sig + 1e-10)
                
                # Negative
                neg_idxs = get_neg_samples([center_idx], neg_samples, neg_probs, vocab_size)
                for neg_idx in neg_idxs:
                    neg_vec = W_out[neg_idx]
                    dot_neg = np.dot(context_mean, neg_vec)
                    sig_neg = sigmoid(dot_neg)
                    
                    W_out[neg_idx] -= lr * sig_neg * context_mean
                    grad_in += sig_neg * neg_vec
                    total_loss += -np.log(1 - sig_neg + 1e-10)
                    
                W_out[center_idx] -= lr * grad_out
                grad_per_ctx = grad_in / len(context_idxs)
                for ctx_idx in context_idxs:
                    W_in[ctx_idx] -= lr * grad_per_ctx
                count += 1
                
        avg_loss = total_loss / max(count, 1)
        loss_history.append(avg_loss)
        print(f"  epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}", flush=True)
        lr *= 0.95
        
    return W_in, W_out, loss_history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["skipgram", "cbow"], required=True)
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--neg_samples", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--save_name", type=str, required=True)
    args = parser.parse_args()

    with open('data/processed/vocab.pkl', 'rb') as f:
        vocab_data = pickle.load(f)
    with open('data/processed/processed_docs.pkl', 'rb') as f:
        docs = pickle.load(f)

    word2idx = vocab_data['word2idx']
    word_counts = vocab_data['word_counts']
    neg_probs = get_neg_sample_probs(word2idx, word_counts)

    if args.model == "skipgram":
        W_in, W_out, loss = train_skipgram(docs, word2idx, neg_probs, args.embed_dim, args.window, args.neg_samples, args.epochs, args.lr)
    else:
        W_in, W_out, loss = train_cbow(docs, word2idx, neg_probs, args.embed_dim, args.window, args.neg_samples, args.epochs, args.lr)

    os.makedirs('data/models', exist_ok=True)
    
    np.save(f'data/models/{args.save_name}_W.npy', W_in)
    with open(f'data/models/{args.save_name}_loss.pkl', 'wb') as f:
        pickle.dump(loss, f)
        
    print(f"Finished. Saved to data/models/{args.save_name}_W.npy")

if __name__ == "__main__":
    main()
