"""
Hybrid retriever: BM25 keyword search + FAISS vector search, fused with RRF.

Loaded once and reused across queries. Indices must be built first by running:
    python src/ingest.py
"""

import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EMBEDDING_MODEL, INDEX_DIR, BM25_TOP_K, VECTOR_TOP_K, FINAL_TOP_K


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace (matches ingest.py)."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class HybridRetriever:
    """
    Combines BM25 (keyword) and FAISS (semantic) search using
    Reciprocal Rank Fusion (RRF) to produce a single ranked list.

        RRF score = Σ  1 / (60 + rank_i)
    """

    def __init__(self):
        faiss_path = INDEX_DIR / "faiss_index"
        bm25_path  = INDEX_DIR / "bm25.pkl"

        if not faiss_path.exists() or not bm25_path.exists():
            raise FileNotFoundError(
                "Index files not found. Run `python src/ingest.py` first."
            )

        print("Loading embedding model…")
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        print("Loading FAISS index…")
        self._vectorstore = FAISS.load_local(
            str(faiss_path),
            self._embeddings,
            allow_dangerous_deserialization=True,
        )

        print("Loading BM25 index…")
        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
        self._bm25      = data["bm25"]
        self._texts     = data["texts"]
        self._metadatas = data["metadatas"]

        print("Retriever ready.\n")

    def search(self, query: str) -> List[Document]:
        """Return the top FINAL_TOP_K chunks most relevant to the query."""

        # ── BM25 search ────────────────────────────────────────────────────────
        tokens = tokenize(query)
        bm25_scores = self._bm25.get_scores(tokens)
        top_bm25_idx = np.argsort(bm25_scores)[-BM25_TOP_K:][::-1]

        bm25_hits = [
            (self._texts[i], self._metadatas[i], rank)
            for rank, i in enumerate(top_bm25_idx)
        ]

        # ── FAISS vector search ────────────────────────────────────────────────
        vector_docs = self._vectorstore.similarity_search(query, k=VECTOR_TOP_K)
        vector_hits = [
            (doc.page_content, doc.metadata, rank)
            for rank, doc in enumerate(vector_docs)
        ]

        # ── Reciprocal Rank Fusion ─────────────────────────────────────────────
        # Use the first 120 characters as a stable dedup key (avoids exact-dup issues)
        rrf_scores: defaultdict[str, float] = defaultdict(float)
        doc_store:  dict[str, tuple]        = {}

        for text, metadata, rank in bm25_hits:
            key = text[:120]
            rrf_scores[key] += 1.0 / (60 + rank)
            doc_store[key]   = (text, metadata)

        for text, metadata, rank in vector_hits:
            key = text[:120]
            rrf_scores[key] += 1.0 / (60 + rank)
            doc_store[key]   = (text, metadata)

        ranked = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

        return [
            Document(page_content=doc_store[k][0], metadata=doc_store[k][1])
            for k in ranked[:FINAL_TOP_K]
        ]
