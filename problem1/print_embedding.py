import pickle
import numpy as np
import argparse

def main():
    # Setup simple argument parsing to allow user choice if they want, default to 'engineering'
    parser = argparse.ArgumentParser(description="Print the 300D embedding for a specific word.")
    parser.add_argument('--word', type=str, default='engineering', help='The word to extract the vector for')
    args = parser.parse_args()

    target_word = args.word

    # 1. Load the mapped vocabulary dictionary
    try:
        with open('data/processed/vocab.pkl', 'rb') as f:
            vocab_data = pickle.load(f)
            word2idx = vocab_data['word2idx']
    except FileNotFoundError:
        print("Error: Vocabulary file not found. Ensure you have run preprocess.py first.")
        return

    # Check if the requested word exists in our scraped corpus
    if target_word not in word2idx:
        print(f"Error: The word '{target_word}' is not present in our filtered vocabulary.")
        return

    idx = word2idx[target_word]

    # 2. Load the trained 300-dimensional Skip-gram weight matrix (Configuration 2)
    # The 'sg2_W.npy' matrix serves as our center-word embedding lookup table
    try:
        W = np.load('data/models/sg2_W.npy')
    except FileNotFoundError:
        print("Error: Model weights not found. Ensure you have run train.py first.")
        return

    # Extract the full 300D spatial vector using the mapped index
    vector = W[idx]
    
    # 3. Format the array as a comma-separated string to match assignment requirements
    # Keeping 4 decimal places for cleanly formatted floating points
    formatted_vector = ", ".join([f"{val:.4f}" for val in vector])
    
    # Print the deliverable output precisely
    print(f"{target_word} - {formatted_vector}")

if __name__ == "__main__":
    main()
