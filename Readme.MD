# NLU Assignment 2

This project contains solutions for two problems related to Natural Language Understanding: Word Embeddings and Name Generation.

## 🛠️ Required Libraries

To run the scripts, you need to install the following libraries:

```bash
pip install numpy torch matplotlib scikit-learn PyPDF2 pandas
```

---

## 🔹 Problem 1: Word Embeddings (CBOW & Skip-gram)

This part involves extracting text from university documents to train word embeddings.

### 📊 Data Used
The model is trained on a corpus of **IIT Jodhpur academic documents** (PDFs), including:
- CSE Course Details & Curriculum
- Departmental Newsletters
- Faculty & People directories

### How to Run:

1. **Extraction & Text Cleaning:**
   Extract text from PDFs in the `data/` folder.
   ```bash
   python Problem_1/Extraction.py
   ```
2. **Preprocess Data:**
   Tokenize and build vocabulary.
   ```bash
   python Problem_1/Preprocess.py
   ```
3. **Train Models:**
   - Train CBOW:
     ```bash
     python Problem_1/CBow.py
     ```
   - Train Skip-gram:
     ```bash
     python Problem_1/SkipGram.py
     ```
4. **Analyze Results:**
   Check nearest neighbors and word analogies.
   ```bash
   python Problem_1/Analysis.py
   ```
5. **Visualization:**
   Generate PCA and t-SNE plots for the embeddings.
   ```bash
   python Problem_1/PCA_Visualization.py
   ```

---

## 🔹 Problem 2: Name Generator

This part builds character-level sequence models to generate Indian names.

### 📊 Data Used
The training data consists of **1,000 unique Indian names** sampled from datasets of Indian male and female names.

### How to Run:

1. **Name Extraction:**
   Fetch and prepare the name dataset from online sources.
  **Train & Evaluate:**
   Train all three models (Vanilla RNN, BLSTM, and RNN with Attention) and save results.
   ```bash
   python Problem_2/name_generator.py
   ```

Results (generated samples and metrics for all models) will be saved in the `Problem_2/results/` directory.

---

**Author:** Mohit Kumar
