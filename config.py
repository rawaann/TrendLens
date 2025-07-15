import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "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")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
ARXIV_MAX_RESULTS = 5
PDF_CHUNK_SIZE = 500
PDF_CHUNK_OVERLAP = 100
CACHE_DIR = "pdf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

try:
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.3)
except:
    llm = OllamaLLM(model="llama2", temperature=0.3) 