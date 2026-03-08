"""
Interactive RAG query interface.

Usage:
    python src/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import RAGPipeline


def print_separator(char="─", width=60):
    print(char * width)


def main():
    print_separator("=")
    print("  Financial Document RAG — Best Agrolife Group")
    print_separator("=")
    print("  Type your question and press Enter.")
    print("  Commands: 'sources' = toggle source display | 'quit' = exit")
    print_separator("=")
    print()

    try:
        pipeline = RAGPipeline()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    show_sources = True

    while True:
        try:
            question = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if question.lower() == "sources":
            show_sources = not show_sources
            print(f"  Source display: {'ON' if show_sources else 'OFF'}\n")
            continue

        print()
        try:
            result = pipeline.query(question)
        except Exception as exc:
            print(f"[ERROR] {exc}\n")
            continue

        print_separator()
        print(result["answer"])
        print_separator()

        if show_sources:
            print("Sources used:")
            seen = set()
            for s in result["sources"]:
                key = f"{s['folder']}/{s['file']} (page {s['page']})"
                if key not in seen:
                    print(f"  • {key}")
                    seen.add(key)

        print()


if __name__ == "__main__":
    main()
