import PyPDF2
import re
import os
import pickle
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# ---------------------------
# Extract text from PDF
# ---------------------------
def extract_pdf_text(pdf_path):
    text = []
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

    return "\n".join(text)


# ---------------------------
# Clean text
# ---------------------------
def clean_text(text):
    text = re.sub(r'https?://\S+', '', text)      # URLs
    text = re.sub(r'\S+@\S+', '', text)           # Emails
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)    # Non-English
    text = re.sub(r'\s+', ' ', text)              # Extra spaces
    return text.strip()


# ---------------------------
# Preprocess (tokenization)
# ---------------------------
def preprocess(text):

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    tokenized_sentences = []
    all_words = []

    for sentence in sentences:
        sentence = re.sub(r'[^\w\s]', ' ', sentence)
        words = sentence.lower().split()
        words = [w for w in words if len(w) > 2]

        if len(words) > 2:
            tokenized_sentences.append(words)
            all_words.extend(words)

    return tokenized_sentences, all_words


# ---------------------------
# Build vocabulary
# ---------------------------
def build_vocab(all_words):
    word_freq = Counter(all_words)

    vocab = {
        word: idx
        for idx, (word, freq) in enumerate(
            sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        )
        if freq >= 2
    }

    return vocab, word_freq




text = open("Problem_1/corpus.txt", encoding="utf-8").read()

wc = WordCloud(
    width=1200,
    height=600,
    background_color='black',
    colormap='plasma',
    max_words=200,
    random_state=42
).generate(text)

plt.figure(figsize=(14, 7))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.savefig("real_wordcloud.png", dpi=150, bbox_inches='tight')
plt.close()


# ---------------------------
# MAIN PIPELINE
# ---------------------------
def main():

    pdf_dir = 'data'
    corpus = []

    # 1. Extract + clean
    for file in os.listdir(pdf_dir):
        if file.endswith('.pdf'):
            print(f"Processing {file}...")
            text = extract_pdf_text(os.path.join(pdf_dir, file))
            text = clean_text(text)
            corpus.append(text)

    all_text = "\n\n".join(corpus)

    # Save corpus
    with open('Problem_1/corpus.txt', 'w', encoding='utf-8') as f:
        f.write(all_text)

    # 2. Preprocess
    tokenized_sentences, all_words = preprocess(all_text)

    # 3. Build vocab
    vocab, word_freq = build_vocab(all_words)

    # 4. Save outputs
    with open('Problem_1/tokenized_sentences.pkl', 'wb') as f:
        pickle.dump(tokenized_sentences, f)

    with open('Problem_1/vocabulary.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    # 5. Stats
    print("\n=== STATS ===")
    print("Total sentences:", len(tokenized_sentences))
    print("Total words:", len(all_words))
    print("Vocabulary size:", len(vocab))

    print("\nTop 20 words:")
    for w, c in word_freq.most_common(20):
        print(f"{w}: {c}")


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    main()