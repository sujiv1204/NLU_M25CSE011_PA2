import torch
from utils import get_logger

logger = get_logger()

def generate_names(model, vocab, idx_to_char, device, n=100, max_len=20, temp=1.1):
    """
    Steps an evaluated network natively up to max_len parameters using multinomial 
    temperature sampling to establish varied unique novelty names.
    """
    model.eval()
    generated = []
    
    START, END, PAD = '<S>', '<E>', '<P>'

    with torch.no_grad():
        for _ in range(n):
            seq = [vocab[START]]
            name = []
            
            for _ in range(max_len):
                x = torch.tensor([seq]).to(device)
                logits, _ = model(x)
                
                # Temperature shift governs structural conformity vs diverse creativity
                logits = logits[0, -1] / temp
                probs = torch.softmax(logits, dim=0)
                
                idx = torch.multinomial(probs, 1).item()
                
                if idx == vocab[END]:
                    break
                    
                if idx not in [vocab[PAD], vocab[START]]:
                    name.append(idx_to_char[idx])
                
                seq.append(idx)
            
            if name:
                generated.append(''.join(name))
                
    return generated
