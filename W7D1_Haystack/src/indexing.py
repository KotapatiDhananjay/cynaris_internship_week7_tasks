
from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import (
    DocumentCleaner,
    DocumentSplitter
)
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_indexing_pipeline(document_store):

    pipeline = Pipeline()

    pipeline.add_component(
        "converter",
        PyPDFToDocument()
    )

    pipeline.add_component(
        "cleaner",
        DocumentCleaner()
    )

    pipeline.add_component(
        "splitter",
        DocumentSplitter(
            split_by="word",
            split_length=150,
            split_overlap=20
        )
    )

    pipeline.add_component(
        "embedder",
        SentenceTransformersDocumentEmbedder(
            model=MODEL_NAME
        )
    )

    pipeline.add_component(
        "writer",
        DocumentWriter(
            document_store=document_store
        )
    )

    pipeline.connect(
        "converter.documents",
        "cleaner.documents"
    )

    pipeline.connect(
        "cleaner.documents",
        "splitter.documents"
    )

    pipeline.connect(
        "splitter.documents",
        "embedder.documents"
    )

    pipeline.connect(
        "embedder.documents",
        "writer.documents"
    )

    return pipeline


def index_documents(data_dir, document_store):

    pdf_files = sorted(Path(data_dir).glob("*.pdf"))

    if len(pdf_files) != 5:
        raise ValueError(
            f"Expected 5 PDF documents, found {len(pdf_files)}"
        )

    pipeline = build_indexing_pipeline(document_store)

    result = pipeline.run({
        "converter": {
            "sources": pdf_files
        }
    })

    print("PDF indexing completed.")
    print("PDF documents:", len(pdf_files))
    print("Document chunks:", document_store.count_documents())

    return result
