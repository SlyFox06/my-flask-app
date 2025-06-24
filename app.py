import os
import requests
import nltk
import logging
from flask import Flask, render_template, request, flash, redirect
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from werkzeug.utils import secure_filename
from docx import Document
import fitz  # PyMuPDF
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from dotenv import load_dotenv
import time

# Initialize logging
logging.basicConfig(filename='app.log', level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env variables
load_dotenv()

# NLTK setup
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')

# Rate limiting
limiter = Limiter(key_func=get_remote_address, default_limits=["200/day", "50/hour"])
limiter.init_app(app)


# Caching
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

# Configuration
UPLOAD_FOLDER = 'user_files'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'rtf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Load model and stopwords
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
STOPWORDS = set(stopwords.words('english'))

# -----------------------
# Helper Functions
# -----------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess(text):
    try:
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum() and word not in STOPWORDS]
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return text.lower()

def extract_text(file, filename):
    try:
        extension = filename.rsplit('.', 1)[1].lower()
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save file
        file.seek(0)
        with open(file_path, 'wb') as f:
            f.write(file.read())

        if extension == 'pdf':
            doc = fitz.open(file_path)
            text = ''
            for page in doc:
                text += page.get_text()
            doc.close()
            os.remove(file_path)
            return text

        elif extension == 'docx':
            doc = Document(file_path)
            text = '\n'.join([p.text for p in doc.paragraphs if p.text])
            os.remove(file_path)
            return text

        elif extension in ['txt', 'rtf']:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            os.remove(file_path)
            return text

        logger.warning("Unsupported file extension.")
        return ""
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""

# -----------------------
# API Functions
# -----------------------
@cache.memoize(timeout=3600)
def fetch_core_papers(query):
    try:
        url = f"https://api.core.ac.uk/v3/search/works?q={query}"
        headers = {"Authorization": f"Bearer {os.getenv('CORE_API_KEY')}"}
        response = requests.get(url, headers=headers, timeout=10)
        papers = []
        if response.status_code == 200:
            results = response.json().get('results', [])
            for paper in results[:5]:
                papers.append({
                    "title": paper.get('title', ''),
                    "authors": [a.get('name', '') for a in paper.get('authors', [])],
                    "abstract": paper.get('abstract', ''),
                    "source": "CORE"
                })
        return papers
    except Exception as e:
        logger.error(f"CORE fetch error: {e}")
        return []

@cache.memoize(timeout=3600)
def fetch_semantic_papers(query):
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=5&fields=title,authors,abstract"
        response = requests.get(url, timeout=10)
        papers = []
        if response.status_code == 200:
            results = response.json().get('data', [])
            for paper in results:
                papers.append({
                    "title": paper.get('title', ''),
                    "authors": [a.get('name') for a in paper.get('authors', [])],
                    "abstract": paper.get('abstract', ''),
                    "source": "Semantic Scholar"
                })
        return papers
    except Exception as e:
        logger.error(f"Semantic Scholar fetch error: {e}")
        return []

# -----------------------
# Main Route
# -----------------------
@app.route('/', methods=['GET', 'POST'])
@limiter.limit("10/minute")
def index():
    if request.method == 'POST':
        if 'document' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        uploaded_file = request.files['document']
        if uploaded_file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if not allowed_file(uploaded_file.filename):
            flash('Unsupported file type.', 'error')
            return redirect(request.url)

        try:
            if request.content_length > MAX_FILE_SIZE:
                flash(f'File too large (max {MAX_FILE_SIZE//(1024*1024)}MB)', 'error')
                return redirect(request.url)

            credit_input = request.form.get('credit', '').lower()
            query = request.form.get('query', 'artificial intelligence')
            filename = secure_filename(uploaded_file.filename)

            extracted_text = extract_text(uploaded_file, filename)

            if not extracted_text.strip():
                flash("Could not extract text from file.", "error")
                return redirect(request.url)

            # Process
            time.sleep(1)
            cleaned_text = preprocess(extracted_text)
            user_embed = model.encode(cleaned_text, convert_to_tensor=True)

            papers = fetch_core_papers(query) + fetch_semantic_papers(query)

            best_score = 0
            best_match = None
            results = []

            for paper in papers:
                if paper['abstract']:
                    paper_clean = preprocess(paper['abstract'])
                    paper_embed = model.encode(paper_clean, convert_to_tensor=True)
                    score = util.cos_sim(user_embed, paper_embed).item()
                    results.append({
                        'title': paper['title'],
                        'authors': ', '.join(paper['authors']),
                        'source': paper['source'],
                        'score': round(score * 100, 2)
                    })
                    if score > best_score:
                        best_score = score
                        best_match = paper

            similarity = round(best_score * 100, 2)
            credit_status = 'Given' if credit_input in extracted_text.lower() else 'Not Given'

            return render_template('index.html',
                                   similarity_score=similarity,
                                   credit_status=credit_status,
                                   match_title=best_match['title'] if best_match else '',
                                   match_authors=', '.join(best_match['authors']) if best_match else '',
                                   matches=results[:5],
                                   query=query,
                                   credit_info=credit_input)

        except Exception as e:
            logger.error(f"Processing error: {e}")
            flash(f"An error occurred: {e}", "error")
            return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', 'False') == 'True')
