import os
import re
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

stopwords = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these',
    'those', 'it', 'its', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'when',
    'where', 'why', 'how', 'any', 'if', 'then', 'what', 'which', 'who', 'whom',
    'their', 'them', 'they', 'we', 'us', 'our', 'you', 'your', 'he', 'she',
    'his', 'her', 'up', 'out', 'about', 'into', 'over', 'after', 'through'
}

def clean_and_tokenize(text):
    # Standardize casing first so 'Course' and 'course' hit the same vocabulary index
    text = text.lower()
    
    # Strip networking noise like hyper-links and email addresses scraped from staff directories
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove standalone numeric digits (we want semantic words, not random years or phone numbers)
    text = re.sub(r'\b\d+\b', '', text)
    
    # Replace punctuation and non-alpha characters with spaces to cleanly split tokens later
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    tokens = text.split()
    cleaned = []
    
    for tok in tokens:
        # Crucial length condition (length >= 2) explicitly configured to keep academic 
        # domain-acronyms like 'ug', 'pg', 'ai' intact while dropping leftover single-letter noise.
        if len(tok) >= 2 and tok.isalpha() and tok not in stopwords:
            cleaned.append(tok)
    return cleaned

def build_vocab(sentences, min_count=2):
    word_counts = Counter()
    for sent in sentences:
        word_counts.update(sent)
        
    word2idx = {}
    idx2word = {}
    idx = 0
    for word, count in word_counts.items():
        if count >= min_count:
            word2idx[word] = idx
            idx2word[idx] = word
            idx += 1
    return word2idx, idx2word, word_counts

def main():
    print("Starting Preprocessing")
    if not os.path.exists('data/raw_corpus.txt'):
        print("data/raw_corpus.txt not found. Run scraper.py first.")
        return

    with open('data/raw_corpus.txt', 'r', encoding='utf-8') as f:
        content = f.read()

    documents = content.split('\n\n')
    processed_docs = []
    for text in documents:
        if not text.strip(): continue
        tokens = clean_and_tokenize(text)
        if len(tokens) > 10:
            processed_docs.append(tokens)

    all_tokens = [tok for doc in processed_docs for tok in doc]
    word_freq = Counter(all_tokens)

    print("\nCorpus Statistics:")
    print(f"Number of documents: {len(processed_docs)}")
    print(f"Total tokens: {len(all_tokens)}")
    print(f"Vocabulary size: {len(word_freq)}")
    print("\nTop 20 words:")
    for word, count in word_freq.most_common(20):
        print(f"  {word}: {count}")

    os.makedirs('data', exist_ok=True)
    with open('data/corpus.txt', 'w', encoding='utf-8') as f:
        for doc in processed_docs:
            f.write(' '.join(doc) + '\n')
    print("Saved processed corpus to data/corpus.txt")

    # WordCloud
    wc = WordCloud(width=800, height=400, background_color='white', max_words=100)
    wc.generate_from_frequencies(word_freq)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Frequent Words in IIT Jodhpur Corpus')
    plt.savefig('data/wordcloud.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Bar plot
    top_words = word_freq.most_common(20)
    words, counts = zip(*top_words)
    plt.figure(figsize=(12, 6))
    plt.bar(words, counts, color='skyblue')
    plt.title('Top 20 Most Frequent Words in Corpus')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/word_freq_bar.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Build and save vocab
    os.makedirs('data/processed', exist_ok=True)
    word2idx, idx2word, word_counts_filtered = build_vocab(processed_docs, min_count=2)
    print(f"Filtered vocabulary size (min_count=2): {len(word2idx)}")
    
    with open('data/processed/vocab.pkl', 'wb') as f:
        pickle.dump({'word2idx': word2idx, 'idx2word': idx2word, 'word_counts': word_counts_filtered}, f)
        
    with open('data/processed/processed_docs.pkl', 'wb') as f:
        pickle.dump(processed_docs, f)

    print("Preprocessing completed and data saved.")

if __name__ == "__main__":
    main()
