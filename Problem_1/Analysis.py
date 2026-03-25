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
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)


# -------------------
# Nearest Neighbors
# -------------------
def get_neighbors(word, vectors, k=5):
    if word not in vectors:
        print(f"[Warning] '{word}' not in vocabulary")
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
    missing = [w for w in [a, b, c] if w not in vectors]
    if missing:
        print(f"[Warning] Missing words: {missing}")
        return []

    # normalize vectors (important improvement)
    vec_a = vectors[a] / (np.linalg.norm(vectors[a]) + 1e-10)
    vec_b = vectors[b] / (np.linalg.norm(vectors[b]) + 1e-10)
    vec_c = vectors[c] / (np.linalg.norm(vectors[c]) + 1e-10)

    target = vec_c + (vec_b - vec_a)

    scores = []
    for w, vec in vectors.items():
        if w not in [a, b, c]:
            vec_w = vec / (np.linalg.norm(vec) + 1e-10)
            sim = cosine_similarity(target, vec_w)
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
# Better Analogy Tests (more reliable)
# -------------------
print("\n=== Analogies ===")

tests = [
    ('phd', 'research', 'mtech'),
    ('student', 'students', 'teacher'),   # safer
    ('research', 'papers', 'study')       # safer
]


print("\nCBOW:")
for a, b, c in tests:
    print(f"\n{a} : {b} :: {c} : ?")
    results = analogy(a, b, c, cbow)
    if not results:
        print("  No result")
    for w, s in results:
        print(f"  {w} ({s:.4f})")


print("\nSkip-gram:")
for a, b, c in tests:
    print(f"\n{a} : {b} :: {c} : ?")
    results = analogy(a, b, c, skipgram)
    if not results:
        print("  No result")
    for w, s in results:
        print(f"  {w} ({s:.4f})")