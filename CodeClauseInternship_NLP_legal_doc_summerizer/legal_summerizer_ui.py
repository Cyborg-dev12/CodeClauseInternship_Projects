import streamlit as st
import spacy
import torch
import pdfplumber
import re
import json
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

nlp = spacy.load("en_core_web_sm")

LEGAL_STOP_WORDS = ["hereby", "thereof", "therein", "whereas", "hereto"]

def preprocess_text(text):
    sentences = sent_tokenize(text)
    return sentences


def clean_text(text):
    legal_terms = ["hereby", "thereof", "whereas", "aforementioned", "thereto"]
    pattern = r'\b(?:' + '|'.join(legal_terms) + r')\b'
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', cleaned_text).strip()

def extract_clauses(text, keywords, regex_patterns):
    clauses = []
    for sentence in preprocess_text(text):
        cleaned_sentence = clean_text(sentence)
        if any(keyword in cleaned_sentence.lower() for keyword in keywords) or any(re.search(pattern, cleaned_sentence) for pattern in regex_patterns):
            clauses.append(sentence)
    return clauses

def summarize_text(text, max_length=130, min_length=30):
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error in summarization: {e}")
        return text

def extract_named_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def extract_and_summarize(text):
    keywords = ["liability", "termination", "confidentiality", "dispute", "governing law"]
    regex_patterns = [r"\bGoverning Law\b", r"\bLiability\b", r"\bTermination\b"]

    clauses = extract_clauses(text, keywords, regex_patterns)

    summarized_clauses = []
    for clause in clauses:
        summary = summarize_text(clause)
        entities = extract_named_entities(clause)
        summarized_clauses.append({
            "clause": clause,
            "summary": summary,
            "entities": entities
        })
    
    return summarized_clauses

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

st.title("Legal Document Summarizer")

uploaded_file = st.file_uploader("Upload a legal document (PDF or Text)", type=["pdf", "txt"])

if uploaded_file is not None:

    if uploaded_file.type == "application/pdf":
        st.write("Analyzing PDF document...")
        document_text = extract_text_from_pdf(uploaded_file)
    else:
        document_text = uploaded_file.read().decode("utf-8")

    st.write("**Extracted Text from the Document:**")
    st.write(document_text)

    st.write("**Summarized Clauses and Key Information:**")
    summarized_data = extract_and_summarize(document_text)

    for item in summarized_data:
        st.subheader("Clause:")
        st.write(item['clause'])
        st.subheader("Summary:")
        st.write(item['summary'])
        st.subheader("Entities:")
        st.write(item['entities'])


    export_option = st.radio("Export Summary", ["None", "JSON", "HTML"])
    if export_option == "JSON":
        filename = "legal_summary.json"
        with open(filename, "w") as f:
            json.dump(summarized_data, f, indent=4)
        st.write(f"Summary saved to {filename}")
    elif export_option == "HTML":
        filename = "legal_summary.html"
        with open(filename, "w") as f:
            f.write("<html><body><h1>Legal Document Summary</h1>")
            for item in summarized_data:
                f.write(f"<h2>Clause:</h2><p>{item['clause']}</p>")
                f.write(f"<h2>Summary:</h2><p>{item['summary']}</p>")
                f.write(f"<h3>Entities:</h3><p>{item['entities']}</p><hr>")
            f.write("</body></html>")
        st.write(f"Summary saved to {filename}")

