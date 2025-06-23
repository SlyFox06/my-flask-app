headers = {"Authorization": "Bearer RKSxqgz6mYarlWoUJkIPvE7NG3TtMLj5"}

import os
import requests
import json
import nltk
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from werkzeug.utils import secure_filename
from docx import Document
from io import BytesIO
from pdfminer.high_level import extract_text as extract_pdf_text

# Download required NLTK data with error handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Additional download for punkt_tab if needed
try:
    nltk.data.find('tokenizers/punkt_tab/english')
except LookupError:
    try:
        nltk.download('punkt_tab')
    except:
        print("Couldn't download punkt_tab specifically, trying full punkt download")
        nltk.download('punkt')

STOPWORDS = set(stopwords.words('english'))

app = Flask(__name__)
UPLOAD_FOLDER = 'user_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------
# TEXT PREPROCESSING
# -----------------------
def preprocess(text):
    try:
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum() and word not in STOPWORDS]
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return text.lower()  # Fallback to simple lowercase if tokenization fails

# -----------------------
# FILE PARSERS
# -----------------------
def extract_text(file, filename):
    try:
        if filename.endswith('.pdf'):
            return extract_pdf_text(file)
        elif filename.endswith('.docx'):
            document = Document(file)
            return '\n'.join([p.text for p in document.paragraphs])
        else:
            return file.read().decode('utf-8')
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

# -----------------------
# API FETCHERS
# -----------------------
def fetch_core_papers(query):
    try:
        url = f"https://api.core.ac.uk/v3/search/works?q={query}"
        headers = {"Authorization": "Bearer N0A3kK2vdcViuWSqPmBnbEQMGZae9rUCexir"}
        response = requests.get(url, headers=headers)
        papers = []
        if response.status_code == 200:
            results = response.json().get('results', [])
            for paper in results[:5]:
                papers.append({
                    "title": paper.get('title', ''),
                    "authors": [a.get('name', '') for a in paper.get('authors', [])],
                    "abstract": paper.get('abstract', '')
                })
        return papers
    except Exception as e:
        print(f"Error fetching CORE papers: {e}")
        return []

def fetch_semantic_papers(query):
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=5&fields=title,authors,abstract"
        response = requests.get(url)
        papers = []
        if response.status_code == 200:
            results = response.json().get('data', [])
            for paper in results:
                papers.append({
                    "title": paper.get('title', ''),
                    "authors": [a.get('name') for a in paper.get('authors', [])],
                    "abstract": paper.get('abstract', '')
                })
        return papers
    except Exception as e:
        print(f"Error fetching Semantic Scholar papers: {e}")
        return []

# -----------------------
# MAIN ROUTE
# -----------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    similarity_score = None
    credit_status = None
    match_title = ""
    match_authors = ""
    error_message = None

    if request.method == 'POST':
        try:
            uploaded_file = request.files['document']
            credit_info = request.form['credit'].lower()
            query = request.form.get('query', 'artificial intelligence')

            if uploaded_file:
                filename = secure_filename(uploaded_file.filename)
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                uploaded_file.save(file_path)

                with open(file_path, 'rb') as f:
                    user_text = extract_text(f, filename)

                if not user_text.strip():
                    error_message = "Could not extract text from the uploaded file"
                else:
                    user_text_clean = preprocess(user_text)
                    user_embedding = model.encode(user_text_clean, convert_to_tensor=True)

                    # Get papers from both APIs
                    all_papers = fetch_core_papers(query) + fetch_semantic_papers(query)

                    best_score = 0
                    for paper in all_papers:
                        if paper['abstract']:  # Only process papers with abstracts
                            paper_text = preprocess(paper['abstract'])
                            paper_embedding = model.encode(paper_text, convert_to_tensor=True)
                            score = util.cos_sim(user_embedding, paper_embedding).item()
                            if score > best_score:
                                best_score = score
                                match_title = paper['title']
                                match_authors = ', '.join(paper['authors'])

                    similarity_score = round(best_score * 100, 2) if best_score > 0 else 0
                    credit_status = 'Given' if credit_info in user_text.lower() else 'Not Given'

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(error_message)

    return render_template('index.html',
                         similarity_score=similarity_score,
                         credit_status=credit_status,
                         match_title=match_title,
                         match_authors=match_authors,
                         error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)