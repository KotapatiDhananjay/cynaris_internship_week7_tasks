
from pathlib import Path

from haystack.document_stores.in_memory import InMemoryDocumentStore

from src.create_sample_pdfs import create_pdfs
from src.indexing import index_documents
from src.evaluation import evaluate


def main():

    print("=" * 60)
    print("W7D1 - HAYSTACK PIPELINE ARCHITECTURE")
    print("=" * 60)

    data_dir = Path("data")

    pdf_files = list(data_dir.glob("*.pdf"))

    if len(pdf_files) == 0:
        print("\nCreating five sample PDF documents...")
        create_pdfs()

    if len(list(data_dir.glob("*.pdf"))) != 5:
        raise ValueError("Please provide exactly five PDF documents.")

    # Create document store
    document_store = InMemoryDocumentStore(
        embedding_similarity_function="cosine"
    )

    # Index documents
    print("\nSTEP 1: INDEXING DOCUMENTS")

    index_documents(
        data_dir=data_dir,
        document_store=document_store
    )

    # Run evaluation
    print("\nSTEP 2: BM25 AND DENSE RETRIEVAL")

    evaluate(
        document_store=document_store,
        output_dir="outputs"
    )

    print("\n" + "=" * 60)
    print("W7D1 TASK COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
