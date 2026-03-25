import re
import pickle
from collections import Counter

with open('Problem_1/corpus.txt', 'r') as f:
    text = f.read()


all_words = []
tokenized_sentences = []

# Split into sentences
sentences = re.split(r'[.!?]+', text)
sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

print(f"Total sentences: {len(sentences)}")


for sentence in sentences:
    # Remove punctuation
    sentence = re.sub(r'[^\w\s]', ' ', sentence)
    # Split into words
    words = sentence.lower().split()
    # Remove short words
    words = [w for w in words if len(w) > 2]
    
    if len(words) > 2:  # Keep sentences with at least 3 words
        tokenized_sentences.append(words)
        all_words.extend(words)

print(f"Total valid sentences: {len(tokenized_sentences)}")
print(f"Total words: {len(all_words)}")


# Build vocabulary
word_freq = Counter(all_words)
# Keep words that appear at least 2 times
vocabulary = {word: idx for idx, (word, freq) in enumerate(
    sorted(word_freq.items(), key=lambda x: x[1], reverse=True)) 
    if freq >= 2}

print(f"Vocabulary size: {len(vocabulary)}")

# Save tokenized data

with open('Problem_1/tokenized_sentences.pkl', 'wb') as f:
    pickle.dump(tokenized_sentences, f)

with open('Problem_1/vocabulary.pkl', 'wb') as f:
    pickle.dump(vocabulary, f)

# Print top 20 words
print("\nTop 20 words:")
for word, freq in word_freq.most_common(20):
    print(f"  {word}: {freq}")