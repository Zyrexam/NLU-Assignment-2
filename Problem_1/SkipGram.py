import numpy as np
import pickle
import random
from collections import Counter
from Problem_1.utils import plot_training_loss


# Load data
with open('Problem_1/tokenized_sentences.pkl', 'rb') as f:
    sentences = pickle.load(f)

with open('Problem_1/vocabulary.pkl', 'rb') as f:
    vocab = pickle.load(f)


# Convert to indices
indexed_sentences = []
for sent in sentences:
    temp = [vocab[w] for w in sent if w in vocab]
    if len(temp) > 2:
        indexed_sentences.append(temp)


# Build frequency
word_freq = Counter()
for sent in indexed_sentences:
    word_freq.update(sent)


# -------------------
# Model
# -------------------
class SkipGram:
    def __init__(self, vocab_size, dim=300, window=3, lr=0.025, neg_samples=10):
        self.W = np.random.randn(vocab_size, dim) * 0.01
        self.C = np.random.randn(vocab_size, dim) * 0.01

        self.window = window
        self.lr = lr
        self.neg_samples = neg_samples
        self.vocab_size = vocab_size

        # sampling distribution
        freq = np.ones(vocab_size)
        for i, c in word_freq.items():
            freq[i] = c
        freq = freq ** 0.75
        self.prob = freq / np.sum(freq)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    

    def get_negatives(self, target):
        negs = []
        while len(negs) < self.neg_samples:
            w = np.random.choice(self.vocab_size, p=self.prob)
            if w != target:
                negs.append(w)
        return negs

    def train_pair(self, target, context):
        loss = 0

        target_vec = self.W[target]
        context_vec = self.C[context]

        # positive
        score = np.dot(target_vec, context_vec)
        pred = self.sigmoid(score)
        loss += -np.log(pred + 1e-10)

        grad = pred - 1
        self.W[target] -= self.lr * grad * context_vec
        self.C[context] -= self.lr * grad * target_vec

        # negatives
        for neg in self.get_negatives(target):
            neg_vec = self.C[neg]

            score = np.dot(target_vec, neg_vec)
            pred = self.sigmoid(score)
            loss += -np.log(1 - pred + 1e-10)

            grad = pred
            self.W[target] -= self.lr * grad * neg_vec
            self.C[neg] -= self.lr * grad * target_vec

        return loss

    def train_sentence(self, sentence):
        loss = 0
        for i, target in enumerate(sentence):
            start = max(0, i - self.window)
            end = min(len(sentence), i + self.window + 1)

            for j in range(start, end):
                if j != i:
                    loss += self.train_pair(target, sentence[j])

        return loss / len(sentence)


# -------------------
# Training
# -------------------
model = SkipGram(len(vocab))

print("Training...")

epoch_losses = []
for epoch in range(3):
    random.shuffle(indexed_sentences)

    total_loss = 0
    for sent in indexed_sentences:
        total_loss += model.train_sentence(sent)

    epoch_loss = total_loss / len(indexed_sentences)
    epoch_losses.append(epoch_loss)

    print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
    model.lr *= 0.95

# Plot training curve
plot_training_loss(list(range(1, len(epoch_losses)+1)), epoch_losses, 'Problem_1/skipgram_training_loss.png')

# Save embeddings
reverse_vocab = {idx: word for word, idx in vocab.items()}

word_vectors = {
    reverse_vocab[i]: model.W[i]
    for i in range(len(vocab))
}

with open("Problem_1/word_vectors_skipgram.pkl", "wb") as f:
    pickle.dump(word_vectors, f)

print("Done!")