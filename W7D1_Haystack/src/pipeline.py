
from haystack import Pipeline

from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever
)

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersTextEmbedder
)

from src.reader import Reader


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_bm25_pipeline(document_store):

    pipeline = Pipeline()

    pipeline.add_component(
        "retriever",
        InMemoryBM25Retriever(
            document_store=document_store,
            top_k=3
        )
    )

    pipeline.add_component(
        "reader",
        Reader()
    )

    pipeline.connect(
        "retriever.documents",
        "reader.documents"
    )

    return pipeline


def build_dense_pipeline(document_store):

    pipeline = Pipeline()

    pipeline.add_component(
        "text_embedder",
        SentenceTransformersTextEmbedder(
            model=MODEL_NAME
        )
    )

    pipeline.add_component(
        "retriever",
        InMemoryEmbeddingRetriever(
            document_store=document_store,
            top_k=3
        )
    )

    pipeline.add_component(
        "reader",
        Reader()
    )

    pipeline.connect(
        "text_embedder.embedding",
        "retriever.query_embedding"
    )

    pipeline.connect(
        "retriever.documents",
        "reader.documents"
    )

    return pipeline


def search_bm25(pipeline, question):

    result = pipeline.run({
        "retriever": {
            "query": question
        }
    })

    return result["reader"]["results"]


def search_dense(pipeline, question):

    result = pipeline.run({
        "text_embedder": {
            "text": question
        }
    })

    return result["reader"]["results"]
