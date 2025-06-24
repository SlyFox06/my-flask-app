import os
import requests
import nltk
import logging
from flask import Flask, render_template, request, flash, redirect, url_for
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

# Load environment variables
load_dotenv()

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key')

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Caching
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})

# Configuration
UPLOAD_FOLDER = 'user_files'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'rtf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Load model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
STOPWORDS = set(stopwords.words('english'))

# -----------------------
# HELPER FUNCTIONS
# -----------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess(text):
    try:
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum() and word not in STOPWORDS]
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return text.lower()

def extract_text(file, filename):
    try:
        extension = filename.rsplit('.', 1)[1].lower()
        file.seek(0)

        if extension == 'pdf':
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(temp_path, 'wb') as temp_pdf:
                temp_pdf.write(file.read())
            doc = fitz.open(temp_path)
            text = ''
            for page in doc:
                text += page.get_text()
            doc.close()
            os.remove(temp_path)
            return text

        elif extension == 'docx':
            try:
                doc = Document(file)
                return '\n'.join([para.text for para in doc.paragraphs if para.text])
            except Exception as e:
                logger.error(f"DOCX extraction failed: {e}")
                return ""

        elif extension in ['txt', 'rtf']:
            try:
                return file.read().decode('utf-8', errors='replace')
            except Exception as e:
                logger.error(f"TXT/RTF extraction failed: {e}")
                return ""

        logger.warning(f"Unsupported file extension: {extension}")
        return ""
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {e}")
        return ""

# -----------------------
# API FETCHERS WITH CACHING
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
        logger.error(f"Error fetching CORE papers: {e}")
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
        logger.error(f"Error fetching Semantic Scholar papers: {e}")
        return []

# -----------------------
# MAIN ROUTES
# -----------------------
@app.route('/', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
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
            flash('Invalid file type. Allowed: PDF, DOCX, TXT, RTF', 'error')
            return redirect(request.url)

        try:
            # Check file size
            file_size = request.content_length
            if file_size > MAX_FILE_SIZE:
                flash(f'File too large. Maximum size is {MAX_FILE_SIZE//(1024*1024)}MB', 'error')
                return redirect(request.url)

            credit_info = request.form['credit'].lower()
            query = request.form.get('query', 'artificial intelligence')
            filename = secure_filename(uploaded_file.filename)

            with uploaded_file.stream as file_stream:
                user_text = extract_text(file_stream, filename)

            if not user_text.strip():
                flash('Could not extract text. The file may be empty, corrupted, or in an unsupported format.', 'error')
                return redirect(request.url)

            time.sleep(1)  # Simulate processing
            user_text_clean = preprocess(user_text)
            user_embedding = model.encode(user_text_clean, convert_to_tensor=True)

            all_papers = fetch_core_papers(query) + fetch_semantic_papers(query)

            best_score = 0
            best_match = None
            matches = []

            for paper in all_papers:
                if paper['abstract']:
                    paper_text = preprocess(paper['abstract'])
                    paper_embedding = model.encode(paper_text, convert_to_tensor=True)
                    score = util.cos_sim(user_embedding, paper_embedding).item()
                    matches.append({
                        'title': paper['title'],
                        'authors': ', '.join(paper['authors']),
                        'source': paper.get('source', 'Unknown'),
                        'score': round(score * 100, 2)
                    })
                    if score > best_score:
                        best_score = score
                        best_match = paper

            similarity_score = round(best_score * 100, 2) if best_score > 0 else 0
            credit_status = 'Given' if credit_info in user_text.lower() else 'Not Given'

            return render_template('index.html',
                                   similarity_score=similarity_score,
                                   credit_status=credit_status,
                                   match_title=best_match['title'] if best_match else '',
                                   match_authors=', '.join(best_match['authors']) if best_match else '',
                                   matches=matches[:5],
                                   query=query,
                                   credit_info=credit_info)

        except Exception as e:
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            flash(f'An error occurred during processing: {str(e)}', 'error')
            return redirect(request.url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG', 'False') == 'True')
