"""
Ingest pipeline: load PDFs → chunk → build FAISS + BM25 indices → save to disk.

Run this ONCE (or whenever you add new PDFs):
    python src/ingest.py
"""

import pickle
import re
import sys
from pathlib import Path
from typing import List

import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PDF_FOLDERS, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, INDEX_DIR


# ── Helpers ────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def extract_tables_as_text(page) -> str:
    """Extract tables from a pdfplumber page and convert to readable text."""
    tables = page.extract_tables()
    if not tables:
        return ""
    lines = []
    for table in tables:
        for row in table:
            if row:
                cleaned = [str(cell).strip() if cell else "" for cell in row]
                lines.append(" | ".join(cleaned))
        lines.append("")  # blank line between tables
    return "\n".join(lines)


# ── PDF Loading ────────────────────────────────────────────────────────────────

def load_pdf(pdf_path: Path) -> List[Document]:
    """
    Load a single PDF with pdfplumber.
    - Extracts plain text AND tables from each page.
    - Attaches metadata: source filename, folder name, page number.
    """
    docs = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                plain_text = page.extract_text() or ""
                table_text = extract_tables_as_text(page)

                # Merge: plain text first, then table content if present
                if table_text:
                    content = plain_text + "\n\n[TABLES]\n" + table_text
                else:
                    content = plain_text

                content = content.strip()
                if not content:
                    continue

                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source":  pdf_path.name,
                        "folder":  pdf_path.parent.name,
                        "page":    page_num,
                    },
                ))
    except Exception as exc:
        print(f"  [WARN] Skipping {pdf_path.name}: {exc}")

    return docs


def load_all_pdfs() -> List[Document]:
    """Walk every configured folder and load all PDFs."""
    all_docs: List[Document] = []
    for folder in PDF_FOLDERS:
        if not folder.exists():
            print(f"[WARN] Folder not found, skipping: {folder}")
            continue

        pdf_files = sorted(folder.glob("*.pdf"))
        print(f"\n  {folder.name}/  ({len(pdf_files)} PDFs)")
        for pdf_path in pdf_files:
            pages = load_pdf(pdf_path)
            print(f"    → {pdf_path.name}  ({len(pages)} pages)")
            all_docs.extend(pages)

    return all_docs


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Split page-level documents into smaller overlapping chunks.
    Metadata (source, folder, page) is preserved on every chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


# ── Index Building ─────────────────────────────────────────────────────────────

def build_index(chunks: List[Document]) -> None:
    """Build FAISS (vector) and BM25 (keyword) indices and save both to disk."""

    texts     = [c.page_content for c in chunks]
    metadatas = [c.metadata      for c in chunks]

    # ── FAISS ──────────────────────────────────────────────────────────────────
    print("\n  Building FAISS vector index…")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vectorstore.save_local(str(INDEX_DIR / "faiss_index"))
    print("  ✓ FAISS index saved")

    # ── BM25 ───────────────────────────────────────────────────────────────────
    print("  Building BM25 keyword index…")
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "texts": texts, "metadatas": metadatas}, f)
    print("  ✓ BM25 index saved")


# ── Entry Point ────────────────────────────────────────────────────────────────

def run_ingest() -> None:
    print("=" * 60)
    print("RAG INGEST PIPELINE")
    print("=" * 60)
    print("\nLoading PDFs from:")

    docs = load_all_pdfs()
    if not docs:
        print("\n[ERROR] No documents loaded. Check PDF_FOLDERS in config.py.")
        sys.exit(1)

    print(f"\nTotal pages loaded : {len(docs)}")

    chunks = chunk_documents(docs)
    print(f"Total chunks created: {len(chunks)}")

    build_index(chunks)

    print("\n✓ Ingestion complete!")
    print("  Run `python src/main.py` to start querying.\n")


if __name__ == "__main__":
    run_ingest()
