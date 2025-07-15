import os
import re
import PyPDF2

CACHE_DIR = "pdf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class Paper:
    def __init__(self, title, authors, abstract, link, pdf_path=None, full_text=None):
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.link = link
        self.pdf_path = pdf_path
        self.full_text = full_text

    @property
    def arxiv_id(self):
        return extract_arxiv_id(self.link)

    def download_pdf(self):
        self.pdf_path = download_pdf(self.arxiv_id)

    def extract_text(self):
        if self.pdf_path:
            self.full_text = extract_text_from_pdf(self.pdf_path)

def extract_arxiv_id(link):
    match = re.search(r'arxiv.org/(abs|pdf)/([\w.]+)', link)
    return match.group(2) if match else None

def download_pdf(arxiv_id):
    pdf_path = os.path.join(CACHE_DIR, f"{arxiv_id}.pdf")
    if os.path.exists(pdf_path):
        return pdf_path
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        import requests
        response = requests.get(pdf_url, timeout=10)
        if response.status_code == 200:
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            return pdf_path
        else:
            return None
    except Exception:
        return None

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception:
        return "" 