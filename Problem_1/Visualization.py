import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# Load embeddings
with open('Problem_1/word_vectors_cbow.pkl', 'rb') as f:
    cbow = pickle.load(f)

with open('Problem_1/word_vectors_skipgram.pkl', 'rb') as f:
    skipgram = pickle.load(f)


# -------------------
# Select better words (important)
# -------------------
words = ['student','teacher','research','paper','learning',
         'algorithm','data','model','training','network',
         'exam','course','education','phd','system',
         'computer','science','analysis','design','theory']

words = [w for w in words if w in cbow and w in skipgram]


cbow_vecs = np.array([cbow[w] for w in words])
skip_vecs = np.array([skipgram[w] for w in words])


# -------------------
# t-SNE
# -------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=5)

cbow_2d = tsne.fit_transform(cbow_vecs)
skip_2d = tsne.fit_transform(skip_vecs)


# -------------------
# Plot
# -------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))


# CBOW
axes[0].scatter(cbow_2d[:, 0], cbow_2d[:, 1])
for i, word in enumerate(words):
    axes[0].text(cbow_2d[i, 0]+0.2, cbow_2d[i, 1]+0.2, word, fontsize=9)

axes[0].set_title("CBOW Embeddings (t-SNE)")
axes[0].grid()


# Skip-gram
axes[1].scatter(skip_2d[:, 0], skip_2d[:, 1])
for i, word in enumerate(words):
    axes[1].text(skip_2d[i, 0]+0.2, skip_2d[i, 1]+0.2, word, fontsize=9)

axes[1].set_title("Skip-gram Embeddings (t-SNE)")
axes[1].grid()


plt.tight_layout()
plt.savefig("Problem_1/tsne_comparison.png", dpi=300)
plt.show()