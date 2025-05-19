import os
import pdfplumber

def extract_passages_from_pdfs(pdf_folder, passage_len=300):
    passages = []
    sources = []
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith('.pdf'):
            with pdfplumber.open(os.path.join(pdf_folder, filename)) as pdf:
                text = ''
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
                for i in range(0, len(text), passage_len):
                    passage = text[i:i+passage_len]
                    if passage.strip():
                        passages.append(passage)
                        sources.append(filename)
    return passages, sources
