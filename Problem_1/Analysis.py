import numpy as np
import pickle


# -------------------
# Load embeddings
# -------------------
with open('Problem_1/word_vectors_cbow.pkl', 'rb') as f:
    cbow = pickle.load(f)

with open('Problem_1/word_vectors_skipgram.pkl', 'rb') as f:
    skipgram = pickle.load(f)


# -------------------
# Cosine Similarity
# -------------------
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# -------------------
# Nearest Neighbors
# -------------------
def get_neighbors(word, vectors, k=5):
    if word not in vectors:
        return []

    target = vectors[word]
    scores = []

    for w, vec in vectors.items():
        if w != word:
            sim = cosine_similarity(target, vec)
            scores.append((w, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


# -------------------
# Analogy Function
# -------------------
def analogy(a, b, c, vectors, k=3):
    if any(w not in vectors for w in [a, b, c]):
        return []

    target = vectors[c] + (vectors[b] - vectors[a])

    scores = []
    for w, vec in vectors.items():
        if w not in [a, b, c]:
            sim = cosine_similarity(target, vec)
            scores.append((w, sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


# -------------------
# Test Words
# -------------------
test_words = ['research', 'student', 'phd', 'exam']


# -------------------
# CBOW Results
# -------------------
print("\n=== CBOW Nearest Neighbors ===")
for word in test_words:
    neighbors = get_neighbors(word, cbow)
    print(f"\n{word}:")
    for w, s in neighbors:
        print(f"  {w:15s} {s:.4f}")


# -------------------
# Skip-gram Results
# -------------------
print("\n=== Skip-gram Nearest Neighbors ===")
for word in test_words:
    neighbors = get_neighbors(word, skipgram)
    print(f"\n{word}:")
    for w, s in neighbors:
        print(f"  {w:15s} {s:.4f}")


# -------------------
# Analogies
# -------------------
print("\n=== Analogies ===")

tests = [
    ('student', 'learning', 'teacher'),
    ('research', 'paper', 'study')
]

print("\nCBOW:")
for a, b, c in tests:
    print(f"\n{a} : {b} :: {c} : ?")
    for w, s in analogy(a, b, c, cbow):
        print(f"  {w} ({s:.4f})")


print("\nSkip-gram:")
for a, b, c in tests:
    print(f"\n{a} : {b} :: {c} : ?")
    for w, s in analogy(a, b, c, skipgram):
        print(f"  {w} ({s:.4f})")