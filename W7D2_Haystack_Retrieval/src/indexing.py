
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

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_indexing_pipeline(store):

    pipeline = Pipeline()

    pipeline.add_component("converter", PyPDFToDocument())
    pipeline.add_component("cleaner", DocumentCleaner())

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
        SentenceTransformersDocumentEmbedder(model=MODEL)
    )

    pipeline.add_component(
        "writer",
        DocumentWriter(document_store=store)
    )

    pipeline.connect("converter.documents", "cleaner.documents")
    pipeline.connect("cleaner.documents", "splitter.documents")
    pipeline.connect("splitter.documents", "embedder.documents")
    pipeline.connect("embedder.documents", "writer.documents")

    return pipeline


def index_pdfs(data_dir, store):

    files = sorted(Path(data_dir).glob("*.pdf"))

    if len(files) != 5:
        raise ValueError(f"Expected 5 PDFs, found {len(files)}")

    pipeline = build_indexing_pipeline(store)

    pipeline.run({
        "converter": {
            "sources": files
        }
    })

    print("\nPDF indexing completed.")
    print("PDF files:", len(files))
    print("Indexed chunks:", store.count_documents())

    return store
