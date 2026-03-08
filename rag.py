"""
RAG pipeline: retrieve relevant chunks → format context → call LLM → return answer.

Requires OPENAI_API_KEY to be set:
    export OPENAI_API_KEY="your-key"
"""

import sys
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OPENAI_API_KEY, OPENAI_MODEL
from retriever import HybridRetriever


# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial analyst assistant for Best Agrolife Limited and its subsidiaries.

Answer the user's question using the context excerpts provided below.

Guidelines:
- For specific factual questions (numbers, dates, names), extract and quote the value directly from the context.
- For summary or overview questions, synthesize the key points from all the excerpts into a clear answer.
- Preserve financial units exactly as they appear (Lacs, Crores, ₹, %).
- At the end of your answer, list the source documents and pages you used.
- Only say "I could not find this information in the available documents" if the excerpts contain genuinely nothing relevant to the question.
"""

HUMAN_PROMPT = """Context:
{context}

Question: {question}"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_context(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        folder = doc.metadata.get("folder", "")
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        header = f"[Excerpt {i} — {folder}/{source}, page {page}]"
        parts.append(f"{header}\n{doc.page_content}")
    return ("\n\n" + "—" * 60 + "\n\n").join(parts)


def _get_llm():
    """Return a ChatOpenAI instance using OPENAI_API_KEY."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "\nOPENAI_API_KEY is not set.\n"
            "  export OPENAI_API_KEY='your-key'\n"
        )
    from langchain_openai import ChatOpenAI
    print(f"Using OpenAI ({OPENAI_MODEL})")
    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )


# ── RAG Pipeline ───────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm       = _get_llm()
        self.prompt    = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human",  HUMAN_PROMPT),
        ])

    def query(self, question: str) -> dict:
        """
        Run the full RAG pipeline for a question.

        Returns:
            {
                "answer":  str,
                "sources": [{"file": str, "folder": str, "page": int}, …],
                "context": str   # raw chunks passed to the LLM
            }
        """
        docs    = self.retriever.search(question)
        context = _format_context(docs)

        chain    = self.prompt | self.llm
        response = chain.invoke({"context": context, "question": question})

        return {
            "answer":  response.content,
            "sources": [
                {
                    "file":   d.metadata.get("source"),
                    "folder": d.metadata.get("folder"),
                    "page":   d.metadata.get("page"),
                }
                for d in docs
            ],
            "context": context,
        }
