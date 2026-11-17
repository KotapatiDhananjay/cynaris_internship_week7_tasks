
from pathlib import Path

from haystack.document_stores.in_memory import InMemoryDocumentStore

from src.create_pdfs import create_pdfs
from src.indexing import index_pdfs
from src.evaluation import run_evaluation


def main():

    print("=" * 55)
    print("W7D2 - HAYSTACK RETRIEVAL")
    print("BM25 VS DENSE RETRIEVAL")
    print("=" * 55)

    data_dir = Path("data")

    if not list(data_dir.glob("*.pdf")):
        print("\nCreating sample PDFs...")
        create_pdfs()

    store = InMemoryDocumentStore(
        embedding_similarity_function="cosine"
    )

    print("\nSTEP 1: INDEXING")
    index_pdfs(data_dir, store)

    print("\nSTEP 2: RETRIEVAL AND EVALUATION")
    run_evaluation(store)

    print("\nW7D2 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
