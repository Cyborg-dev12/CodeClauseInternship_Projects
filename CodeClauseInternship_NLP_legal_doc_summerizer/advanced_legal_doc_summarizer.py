import spacy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import nltk
import re
import json
from nltk.tokenize import sent_tokenize

tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
model = AutoModelForSeq2SeqLM.from_pretrained("./fine_tuned_model")
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

nlp = spacy.load("en_core_web_sm")

LEGAL_STOP_WORDS = ["hereby", "thereof", "therein", "whereas", "hereto"]

def preprocess_text(text):
    sentences = sent_tokenize(text)
    return sentences

import re

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
    entities = []
    for ent in doc.ents:
        if ent.label_ in ["GPE", "ORG", "LAW", "DATE", "MONEY"]:
            if "State of" in ent.text:
                entities.append(("California", ent.label_))  
            else:
                entities.append((ent.text, ent.label_))
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

def export_to_json(summarized_data, filename="legal_summary.json"):
    with open(filename, "w") as file:
        json.dump(summarized_data, file, indent=4)
    print(f"Summary saved to {filename}")

def export_to_html(summarized_data, filename="legal_summary.html"):
    with open(filename, "w") as file:
        file.write("<html><body><h1>Legal Document Summary</h1>")
        for item in summarized_data:
            file.write(f"<h2>Clause:</h2><p>{item['clause']}</p>")
            file.write(f"<h2>Summary:</h2><p>{item['summary']}</p>")
            file.write(f"<h3>Entities:</h3><p>{item['entities']}</p><hr>")
        file.write("</body></html>")
    print(f"Summary saved to {filename}")
def sliding_window_summarization(text, window_size=1024, stride=512, max_length=130, min_length=30):
    summaries = []
    for i in range(0, len(text), stride):
        window_text = text[i:i+window_size]
        summary = summarize_text(window_text, max_length=max_length, min_length=min_length)
        summaries.append(summary)
    

    final_summary = summarize_text(" ".join(summaries), max_length=max_length, min_length=min_length)
    return final_summary

if __name__ == "__main__":
    sample_text = """
    This agreement shall be governed by the laws of the State of California. The parties agree to the termination of services upon breach of contract.
    Confidentiality shall be maintained by both parties for all proprietary information.
    """
    
    summarized_data = extract_and_summarize(sample_text)
    

    print(json.dumps(summarized_data, indent=4))  
    
    export_to_json(summarized_data)  
    export_to_html(summarized_data)  

