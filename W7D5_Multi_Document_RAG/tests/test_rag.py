
from src.ingestion import load_documents


def test_documents_are_loaded():
    documents = load_documents()

    assert len(documents) >= 3
    assert all(document.page_content.strip() for document in documents)


def test_document_source_metadata():
    documents = load_documents()

    for document in documents:
        assert "source" in document.metadata
        assert document.metadata["source"]


def test_document_content():
    documents = load_documents()

    combined_text = " ".join(
        document.page_content.lower()
        for document in documents
    )

    assert "artificial intelligence" in combined_text
    assert "machine learning" in combined_text
    assert "retrieval-augmented generation" in combined_text
