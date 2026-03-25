import numpy as np
import pickle
import random


# ---------------------------
# Load Data
# ---------------------------
with open('Problem_1/tokenized_sentences.pkl', 'rb') as f:
    sentences = pickle.load(f)

with open('Problem_1/vocabulary.pkl', 'rb') as f:
    vocab = pickle.load(f)


# Convert to indices
def convert_to_indices(sentences, vocab):
    indexed = []
    for sent in sentences:
        temp = [vocab[w] for w in sent if w in vocab]
        if len(temp) > 2:
            indexed.append(temp)
    return indexed


indexed_sentences = convert_to_indices(sentences, vocab)
print("Valid sentences:", len(indexed_sentences))


# ---------------------------
# CBOW Model
# ---------------------------
class CBOW:
    def __init__(self, vocab_size, embed_dim=100, window=5, lr=0.025):
        self.embed_dim = embed_dim
        self.window = window
        self.lr = lr

        self.W1 = np.random.randn(vocab_size, embed_dim) * 0.01
        self.W2 = np.random.randn(embed_dim, vocab_size) * 0.01

    def softmax(self, x):
        x = x - np.max(x)
        exp = np.exp(x)
        return exp / np.sum(exp)

    def train_sentence(self, sentence):
        loss = 0

        for i, target in enumerate(sentence):

            # context window
            start = max(0, i - self.window)
            end = min(len(sentence), i + self.window + 1)

            context = [sentence[j] for j in range(start, end) if j != i]
            if not context:
                continue

            # forward
            context_vec = np.mean(self.W1[context], axis=0)
            scores = np.dot(context_vec, self.W2)
            probs = self.softmax(scores)

            # loss
            loss += -np.log(probs[target] + 1e-10)

            # backward
            error = probs.copy()
            error[target] -= 1

            # gradients
            dW2 = np.outer(context_vec, error)
            dcontext = np.dot(self.W2, error) / len(context)

            # update
            self.W2 -= self.lr * dW2
            for w in context:
                self.W1[w] -= self.lr * dcontext

        return loss / len(sentence)


# ---------------------------
# Training
# ---------------------------
model = CBOW(len(vocab))

print("\nTraining...")

for epoch in range(5):

    random.shuffle(indexed_sentences)  

    total_loss = 0

    for i, sent in enumerate(indexed_sentences):
        total_loss += model.train_sentence(sent)

        if (i+1) % 100 == 0:
            print(f"Epoch {epoch+1} | {i+1} sentences")

    print(f"Epoch {epoch+1} Loss: {total_loss/len(indexed_sentences):.4f}")

    model.lr *= 0.95


# ---------------------------
# Save embeddings
# ---------------------------
reverse_vocab = {idx: word for word, idx in vocab.items()}

word_vectors = {
    reverse_vocab[i]: model.W1[i]
    for i in range(len(vocab))
}

with open("Problem_1/word_vectors_cbow.pkl", "wb") as f:
    pickle.dump(word_vectors, f)


print("\nTraining complete!")