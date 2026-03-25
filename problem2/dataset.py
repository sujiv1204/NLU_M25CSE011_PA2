import torch
from utils import get_logger

logger = get_logger()

def load_and_prepare_data(filepath='data/TrainingNames.txt'):
    """
    Loads raw text names, constructs a character-level vocabulary, 
    encodes the data into integers, and pads the resulting tensors.
    """
    logger.info("Task-0: Loading and preparing data from dataset.")
    
    with open(filepath, 'r') as f:
        names = [line.strip() for line in f if line.strip()]

    logger.info(f"Successfully loaded {len(names)} names.")
    logger.info(f"Data sample (first 5): {names[:5]}")

    # Build character vocabulary dynamically from the loaded dataset
    chars = sorted(set(''.join(names)))
    
    START = '<S>'
    END = '<E>'
    PAD = '<P>'

    vocab = {PAD: 0, START: 1, END: 2}
    for i, c in enumerate(chars):
        vocab[c] = i + 3

    idx_to_char = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    logger.info(f"Constructed vocabulary with size: {vocab_size}")

    def encode(name):
        return [vocab[START]] + [vocab[c] for c in name] + [vocab[END]]

    encoded_names = [encode(n) for n in names]
    max_len = max(len(e) for e in encoded_names)

    def pad(seq):
        return seq + [vocab[PAD]] * (max_len - len(seq))

    padded_names = [pad(e) for e in encoded_names]
    data_tensor = torch.tensor(padded_names, dtype=torch.long)

    logger.info(f"Generated padded data tensor of shape: {list(data_tensor.shape)}")
    
    return names, data_tensor, vocab, idx_to_char, vocab_size
