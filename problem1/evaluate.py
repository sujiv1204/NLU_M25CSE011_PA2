import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models import Word2Vec as GensimW2V

# cosine similarity helper function
def find_similar(word, W, word2idx, idx2word, topn=5):
    if word not in word2idx:
        return []
    
    word_idx = word2idx[word]
    word_vec = W[word_idx]
    word_vec = word_vec / (np.linalg.norm(word_vec) + 1e-10)
    
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-10
    W_norm = W / norms
    
    sims = np.dot(W_norm, word_vec)
    sims[word_idx] = -1 
    
    top_idxs = np.argsort(sims)[-topn:][::-1]
    return [(idx2word[i], sims[i]) for i in top_idxs]

# analogy function for scratch models
def analogy(word_a, word_b, word_c, W, word2idx, idx2word, topn=5):
    for w in [word_a, word_b, word_c]:
        if w not in word2idx:
            return []
    
    vec_a = W[word2idx[word_a]]
    vec_b = W[word2idx[word_b]]
    vec_c = W[word2idx[word_c]]
    
    target = vec_b - vec_a + vec_c
    target = target / (np.linalg.norm(target) + 1e-10)
    
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-10
    W_norm = W / norms
    sims = np.dot(W_norm, target)
    
    for w in [word_a, word_b, word_c]:
        sims[word2idx[w]] = -1
    
    top_idxs = np.argsort(sims)[-topn:][::-1]
    return [(idx2word[i], sims[i]) for i in top_idxs]

# Visualization function for PCA and t-SNE comparison
def visualize_embeddings_comparison(W, word2idx, word_groups, title, save_path):
    words = []
    vecs = []
    labels = []
    
    for group_name, group_words in word_groups.items():
        for word in group_words:
            if word in word2idx:
                words.append(word)
                vecs.append(W[word2idx[word]])
                labels.append(group_name)
    
    if len(vecs) < 5:
        return
    
    vecs = np.array(vecs)
    
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(vecs)
    
    perp = min(5, len(vecs) - 5)
    if perp < 2: perp = 2
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    coords_tsne = tsne.fit_transform(vecs)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(title, fontsize=16)
    
    unique_labels = list(word_groups.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
    label2color = {l: colors[i] for i, l in enumerate(unique_labels)}
    
    ax1.set_title('PCA Projection')
    for i, (word, coord) in enumerate(zip(words, coords_pca)):
        color = label2color[labels[i]]
        ax1.scatter(coord[0], coord[1], c=[color], s=100, alpha=0.7)
        ax1.annotate(word, (coord[0], coord[1]), fontsize=9, xytext=(5, 5), textcoords='offset points')
                    
    ax2.set_title('t-SNE Projection')
    for i, (word, coord) in enumerate(zip(words, coords_tsne)):
        color = label2color[labels[i]]
        ax2.scatter(coord[0], coord[1], c=[color], s=100, alpha=0.7)
        ax2.annotate(word, (coord[0], coord[1]), fontsize=9, xytext=(5, 5), textcoords='offset points')
    
    for label, color in label2color.items():
        ax1.scatter([], [], c=[color], label=label, s=100)
    ax1.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# Main evaluation function
def main():
    print("\nLoading models for Evaluation")
    with open('data/processed/vocab.pkl', 'rb') as f:
        vocab_data = pickle.load(f)
    with open('data/processed/processed_docs.pkl', 'rb') as f:
        processed_docs = pickle.load(f)
        
    word2idx = vocab_data['word2idx']
    idx2word = vocab_data['idx2word']
    
    # Load Loss and Plot
    loss_data = {}
    models_to_plot = ['sg1', 'sg2', 'sg3', 'cbow1', 'cbow2', 'cbow3']
    for m in models_to_plot:
        try:
            with open(f'data/models/{m}_loss.pkl', 'rb') as f:
                loss_data[m] = pickle.load(f)
        except:
            loss_data[m] = []

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    if loss_data.get('sg1'): plt.plot(loss_data['sg1'], label='SG Dim=50, Neg=5')
    if loss_data.get('sg2'): plt.plot(loss_data['sg2'], label='SG Dim=300, Neg=5')
    if loss_data.get('sg3'): plt.plot(loss_data['sg3'], label='SG Dim=100, Neg=10')
    plt.title('Skip-gram Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if loss_data.get('sg1'): plt.legend()

    plt.subplot(1, 2, 2)
    if loss_data.get('cbow1'): plt.plot(loss_data['cbow1'], label='CBOW Dim=50, Neg=5')
    if loss_data.get('cbow2'): plt.plot(loss_data['cbow2'], label='CBOW Dim=300, Neg=5')
    if loss_data.get('cbow3'): plt.plot(loss_data['cbow3'], label='CBOW Dim=100, Neg=10')
    plt.title('CBOW Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if loss_data.get('cbow1'): plt.legend()
    plt.tight_layout()
    plt.savefig('data/training_loss.png', dpi=150)
    plt.close()

    # Load main optimized models (Configuration 2 - Dim:300, Window:5)
    # Why these two specifically? We hardcode sg2 and cbow2 here because 300 dimensions is the 
    # industry standard for Word2Vec baseline comparisons (e.g., standard GoogleNews vectors). 
    # Our smaller 50D models underfit complex concepts, and our high-neg setups were slower 
    # without gaining significant semantic accuracy, making Config 2 the best candidate for visual evaluation.
    try:
        W_sg2 = np.load('data/models/sg2_W.npy')
        W_cbow2 = np.load('data/models/cbow2_W.npy')
    except Exception as e:
        print("Model files not found. Run training first.")
        return

    # Train Gensim comparison for baseline
    print("\nTraining Gensim models for baseline comparison:")
    gensim_sg = GensimW2V(sentences=processed_docs, vector_size=300, window=5, min_count=2, sg=1, negative=5, epochs=5, workers=1)
    gensim_cbow = GensimW2V(sentences=processed_docs, vector_size=300, window=5, min_count=2, sg=0, negative=5, epochs=5, workers=1)
    gensim_sg.save('data/models/gensim_sg.model')
    gensim_cbow.save('data/models/gensim_cbow.model')
    print("trained gensim skipgram and cbow models")

    test_words = ['research', 'student', 'phd', 'exam']
    
    models = {
        "Scratch Skip-Gram": (W_sg2, "scratch"),
        "Scratch CBOW": (W_cbow2, "scratch"),
        "Gensim Skip-Gram": (gensim_sg, "gensim"),
        "Gensim CBOW": (gensim_cbow, "gensim")
    }

    print("\n" + "="*50)
    print("TASK 3: SEMANTIC ANALYSIS - NEAREST NEIGHBORS")
    print("="*50)
    
    for model_name, (model_obj, m_type) in models.items():
        print(f"\n--- {model_name} Nearest Neighbors ---")
        for word in test_words:
            if m_type == "scratch":
                res = find_similar(word, model_obj, word2idx, idx2word)
            else:
                try:
                    res = model_obj.wv.most_similar(word, topn=5)
                except KeyError:
                    res = []
            
            if res:
                formatted = [f"{w} ({s:.3f})" for w, s in res]
                print(f"{word.ljust(10)} -> {', '.join(formatted)}")
            else:
                print(f"{word.ljust(10)} -> [Not in Vocabulary]")

    print("\n" + "="*50)
    print("TASK 3: SEMANTIC ANALYSIS - ANALOGY TESTS")
    print("="*50)

    analogy_tests = [
        ('ug', 'btech', 'pg'),
        ('student', 'phd', 'faculty'),
        ('department', 'cse', 'school')
    ]

    for model_name, (model_obj, m_type) in models.items():
        print(f"\n--- Analogy Testing for {model_name} ---")
        for (wa, wb, wc) in analogy_tests:
            query_str = f"{wa} : {wb} :: {wc} : ?"
            if m_type == "scratch":
                res = analogy(wa, wb, wc, model_obj, word2idx, idx2word)
            else:
                try:
                    res = model_obj.wv.most_similar(positive=[wb, wc], negative=[wa], topn=5)
                except KeyError:
                    res = []
            
            if res:
                top_preds = [f"{w}({s:.3f})" for w, s in res[:3]]
                print(f"{query_str.ljust(35)} -> {', '.join(top_preds)}")
            else:
                print(f"{query_str.ljust(35)} -> [Words not found]")

    print("\n" + "="*50)
    print("TASK 4: VISUALIZATIONS GENERATION")
    print("="*50)
    word_groups = {
        'academic': ['research', 'thesis', 'phd', 'mtech', 'btech', 'program', 'course', 'credits', 'semester', 'examination'],
        'departments': ['engineering', 'science', 'mathematics', 'physics', 'chemistry', 'mechanical', 'electrical', 'civil'],
        'people': ['student', 'faculty', 'professor', 'staff', 'director', 'dean'],
        'activities': ['placement', 'internship', 'project', 'hostel', 'sports']
    }

    visualize_embeddings_comparison(W_sg2, word2idx, word_groups, 'PCA vs t-SNE: Skip-gram Model from Scratch', 'data/pca_tsne_skipgram.png')
    visualize_embeddings_comparison(W_cbow2, word2idx, word_groups, 'PCA vs t-SNE: CBOW Model from Scratch', 'data/pca_tsne_cbow.png')
    visualize_embeddings_comparison(gensim_sg.wv.vectors, gensim_sg.wv.key_to_index, word_groups, 'PCA vs t-SNE: Gensim Skip-gram Model', 'data/pca_tsne_gensim_sg.png')
    visualize_embeddings_comparison(gensim_cbow.wv.vectors, gensim_cbow.wv.key_to_index, word_groups, 'PCA vs t-SNE: Gensim CBOW Model', 'data/pca_tsne_gensim_cbow.png')

    print("Visualizations saved to data/ directory successfully.")

    print("\nASSIGNMENT DELIVERABLES SUMMARY:")
    corpus_path = 'data/corpus.txt'
    if os.path.exists(corpus_path):
        size_mb = os.path.getsize(corpus_path) / (1024 * 1024)
        print(f"P1: Size of corpus file -> {size_mb:.4f} MB")
    
    top10 = vocab_data['word_counts'].most_common(10)
    top10_fmt = ", ".join([f"{w}: {c}" for w, c in top10])
    print(f"P1: Top 10 frequent words -> {top10_fmt}")


if __name__ == "__main__":
    main()