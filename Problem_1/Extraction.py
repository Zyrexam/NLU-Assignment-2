
import PyPDF2
import re
import os

def extract_pdf_text(pdf_path):
    """Extract text from a single PDF file"""
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


def clean_text(text):
    """Clean extracted text"""
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\d{4}-\d{4}', '', text)
    
    # Remove non-English characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Clean multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Clean multiple line breaks
    text = re.sub(r'\n\n+', '\n\n', text)
    
    return text.strip()


def main():
    pdf_dir = 'data'
    corpus = []
    
    # Extract from all PDFs
    for filename in sorted(os.listdir(pdf_dir)):
        if filename.endswith('.pdf'):
            filepath = os.path.join(pdf_dir, filename)
            print(f"Processing {filename}...")
            
            text = extract_pdf_text(filepath)
            text = clean_text(text)
            
            if text:
                corpus.append(text)
    
    # Combine all text
    all_text = "\n\n".join(corpus)
    
    # Save corpus
    output_path = 'Problem_1/corpus.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(all_text)
    
    # Print statistics
    print(f"\n{'='*50}")
    print(f"Corpus created: {output_path}")
    print(f"Total characters: {len(all_text):,}")
    print(f"Total words: {len(all_text.split()):,}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()