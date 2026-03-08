"""
Central configuration for the RAG system.
Edit PDF_FOLDERS to add or remove document sources.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # project root

PDF_FOLDERS = [
    ROOT / "annual_results",
    ROOT / "quarterly_results",
    ROOT / "financial_statements",
    ROOT / "rpt_half_year",
]

INDEX_DIR = Path(__file__).resolve().parent / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ── Embeddings ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000       # characters per chunk
CHUNK_OVERLAP = 150     # overlap between consecutive chunks

# ── Retrieval ──────────────────────────────────────────────────────────────────
BM25_TOP_K = 7          # candidates from BM25 keyword search
VECTOR_TOP_K = 7        # candidates from FAISS vector search
FINAL_TOP_K = 5         # final chunks passed to the LLM

# ── LLM ───────────────────────────────────────────────────────────────────────
# Set your OpenAI API key:
#   export OPENAI_API_KEY="your-key"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = "gpt-4o-mini"
