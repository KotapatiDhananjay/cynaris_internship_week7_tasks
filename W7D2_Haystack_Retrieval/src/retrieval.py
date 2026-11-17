
from haystack import Pipeline

from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever
)

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersTextEmbedder
)

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_bm25(store):

    pipeline = Pipeline()

    pipeline.add_component(
        "retriever",
        InMemoryBM25Retriever(
            document_store=store,
            top_k=3
        )
    )

    return pipeline


def build_dense(store):

    pipeline = Pipeline()

    pipeline.add_component(
        "embedder",
        SentenceTransformersTextEmbedder(model=MODEL)
    )

    pipeline.add_component(
        "retriever",
        InMemoryEmbeddingRetriever(
            document_store=store,
            top_k=3
        )
    )

    pipeline.connect(
        "embedder.embedding",
        "retriever.query_embedding"
    )

    return pipeline


def retrieve_bm25(pipeline, question):

    result = pipeline.run({
        "retriever": {
            "query": question
        }
    })

    return result["retriever"]["documents"]


def retrieve_dense(pipeline, question):

    result = pipeline.run({
        "embedder": {
            "text": question
        }
    })

    return result["retriever"]["documents"]


def display_results(method, documents):

    print(f"\n{method} RESULTS")

    for index, document in enumerate(documents, start=1):

        print(f"\nRank: {index}")
        print("Source:", document.meta.get("file_path", "Unknown"))
        print("Score:", document.score)
        print("Content:", document.content[:300])
