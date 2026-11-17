
import time
from pathlib import Path

import chromadb
import pandas as pd

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CHROMA_DIR = BASE_DIR / "chroma_db"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

QUESTIONS = [
    "What is artificial intelligence?",
    "What are the applications of AI?",
    "What is machine learning?",
    "What are the three types of machine learning?",
    "What is natural language processing?",
    "List common NLP tasks.",
    "What is a vector database?",
    "What are embeddings?",
    "What is retrieval augmented generation?",
    "What are the main stages of a RAG system?"
]


# --------------------------------------------------
# STEP 1: CREATE SAMPLE TEXT DOCUMENTS
# --------------------------------------------------

def create_sample_documents():

    sample_documents = {
        "artificial_intelligence.txt": """
ARTIFICIAL INTELLIGENCE

Artificial Intelligence (AI) is a branch of computer science
that develops machines and software capable of performing
tasks that normally require human intelligence.

Examples of AI applications include virtual assistants,
recommendation systems, image recognition, speech recognition
and autonomous vehicles.

Machine Learning is a subset of Artificial Intelligence.
Deep Learning is a subset of Machine Learning that uses
neural networks with multiple layers.
""",

        "machine_learning.txt": """
MACHINE LEARNING

Machine Learning (ML) is a branch of Artificial Intelligence
that allows computers to learn patterns from data and make
predictions without being explicitly programmed for every task.

The three main types of Machine Learning are supervised
learning, unsupervised learning and reinforcement learning.

Supervised learning uses labelled data.
Unsupervised learning identifies patterns in unlabelled data.
Reinforcement learning learns through rewards and penalties.

Common applications include fraud detection, spam filtering
and recommendation systems.
""",

        "natural_language_processing.txt": """
NATURAL LANGUAGE PROCESSING

Natural Language Processing (NLP) is a field of Artificial
Intelligence that helps computers understand, process and
generate human language.

Common NLP tasks include tokenization, stemming,
lemmatization, sentiment analysis, text classification
and machine translation.

NLP is used in chatbots, virtual assistants, language
translation applications and email spam detection.
""",

        "vector_databases.txt": """
VECTOR DATABASES AND EMBEDDINGS

A vector database stores information as numerical vectors
called embeddings.

Embeddings represent the meaning of text, images or other
data in a numerical format. Similar items have vectors
that are close together in vector space.

Vector databases support semantic search by finding
information based on meaning rather than only exact
keyword matches.

Examples include ChromaDB, Pinecone and Weaviate.
""",

        "retrieval_augmented_generation.txt": """
RETRIEVAL AUGMENTED GENERATION

Retrieval Augmented Generation (RAG) combines information
retrieval with Large Language Models.

A RAG system first searches a document collection for
relevant information. It then provides the retrieved context
to a language model, which generates an answer based on
that information.

The main stages of RAG are document loading, text splitting,
embedding generation, vector storage, retrieval and
answer generation.

RAG helps language models answer questions using external
or organization-specific documents.
"""
    }

    for filename, content in sample_documents.items():
        file_path = DATA_DIR / filename
        file_path.write_text(content.strip(), encoding="utf-8")
        print(f"Created {filename}")

    print("\nFive sample text documents created.")


# --------------------------------------------------
# STEP 2: CONFIGURE OLLAMA MODELS
# --------------------------------------------------

def configure_models():

    print("\nConfiguring Ollama models...")

    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        request_timeout=120.0
    )

    Settings.llm = Ollama(
        model="qwen2.5:3b",
        request_timeout=300.0,
        temperature=0,
        context_window=2048,
        additional_kwargs={
            "num_ctx": 2048,
            "num_predict": 150
        }
    )

    Settings.node_parser = SentenceSplitter(
        chunk_size=200,
        chunk_overlap=30
    )

    print("Ollama models configured successfully.")


# --------------------------------------------------
# STEP 3: LOAD TEXT DOCUMENTS
# --------------------------------------------------

def load_documents():

    print("\nLoading text documents...")

    documents = SimpleDirectoryReader(
        input_dir=str(DATA_DIR),
        required_exts=[".txt"]
    ).load_data()

    print(f"Documents loaded: {len(documents)}")

    return documents


# --------------------------------------------------
# STEP 4: BUILD IN-MEMORY VECTOR INDEX
# --------------------------------------------------

def build_memory_index(documents):

    print("\nBuilding in-memory VectorStoreIndex...")

    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )

    print("In-memory index created.")

    return index


# --------------------------------------------------
# STEP 5: BUILD CHROMADB VECTOR INDEX
# --------------------------------------------------

def build_chroma_index(documents):

    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection_name = "w7d3_documents"

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name
    )

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    print("Indexing documents in ChromaDB...")

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    print("ChromaDB indexing completed.")

    return index


# --------------------------------------------------
# STEP 6: CREATE QUERY ENGINE
# --------------------------------------------------

def create_query_engine(index):

    return index.as_query_engine(
        similarity_top_k=2,
        response_mode="compact"
    )


# --------------------------------------------------
# STEP 7: RUN AND EVALUATE QUERIES
# --------------------------------------------------

def run_queries(query_engine, queries, index_name):

    results = []

    print("\n" + "=" * 65)
    print(f"RUNNING QUERIES: {index_name}")
    print("=" * 65)

    # Warm-up query to reduce first-query model-loading impact.
    print("\nWarming up the language model...")

    query_engine.query(
        "What is artificial intelligence?"
    )

    for number, question in enumerate(queries, start=1):

        print("\n" + "-" * 65)
        print(f"QUESTION {number}: {question}")

        start_time = time.perf_counter()

        response = query_engine.query(question)

        latency = time.perf_counter() - start_time

        answer = str(response).strip()

        source_files = []

        for source in response.source_nodes:

            filename = source.node.metadata.get(
                "file_name",
                source.node.metadata.get(
                    "file_path",
                    "Unknown"
                )
            )

            source_files.append(Path(filename).name)

        source_files = list(dict.fromkeys(source_files))

        print("\nANSWER:")
        print(answer)

        print("\nSOURCE DOCUMENTS:")
        print(", ".join(source_files) if source_files else "No source found")

        print(f"\nLATENCY: {latency:.3f} seconds")

        results.append({
            "question_number": number,
            "question": question,
            "answer": answer,
            "source_documents": ", ".join(source_files),
            "latency_seconds": round(latency, 3)
        })

    return results


# --------------------------------------------------
# STEP 8: SAVE RESULTS
# --------------------------------------------------

def save_results(memory_results, chroma_results):

    memory_df = pd.DataFrame(memory_results)
    chroma_df = pd.DataFrame(chroma_results)

    memory_df.to_csv(
        OUTPUT_DIR / "memory_results.csv",
        index=False
    )

    chroma_df.to_csv(
        OUTPUT_DIR / "chroma_results.csv",
        index=False
    )

    comparison = pd.DataFrame({
        "question_number": range(1, len(QUESTIONS) + 1),
        "question": QUESTIONS,
        "memory_latency_seconds": memory_df["latency_seconds"],
        "chroma_latency_seconds": chroma_df["latency_seconds"]
    })

    comparison["difference_seconds"] = (
        comparison["chroma_latency_seconds"]
        - comparison["memory_latency_seconds"]
    )

    comparison.to_csv(
        OUTPUT_DIR / "latency_comparison.csv",
        index=False
    )

    print("\nAll CSV files saved successfully.")

    return memory_df, chroma_df, comparison


# --------------------------------------------------
# STEP 9: DISPLAY LATENCY COMPARISON
# --------------------------------------------------

def display_comparison(memory_df, chroma_df, comparison):

    memory_average = memory_df["latency_seconds"].mean()
    chroma_average = chroma_df["latency_seconds"].mean()

    print("\n" + "=" * 65)
    print("FINAL LATENCY COMPARISON")
    print("=" * 65)

    print(
        f"In-memory average latency: "
        f"{memory_average:.3f} seconds"
    )

    print(
        f"ChromaDB average latency: "
        f"{chroma_average:.3f} seconds"
    )

    print("\nQUESTION-WISE COMPARISON:")

    print(
        comparison[
            [
                "question_number",
                "memory_latency_seconds",
                "chroma_latency_seconds"
            ]
        ].to_string(index=False)
    )

    if memory_average < chroma_average:
        print("\nIn-memory index was faster in this run.")

    elif chroma_average < memory_average:
        print("\nChromaDB was faster in this run.")

    else:
        print("\nBoth indexes had equal average latency.")

    print("\nGenerated output files:")
    print("1. outputs/memory_results.csv")
    print("2. outputs/chroma_results.csv")
    print("3. outputs/latency_comparison.csv")


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    print("=" * 65)
    print("W7D3 - LLAMAINDEX DOCUMENT INDEXING AND QUERYING")
    print("=" * 65)

    try:

        print("\nSTEP 1: CREATING DOCUMENTS")
        create_sample_documents()

        print("\nSTEP 2: CONFIGURING MODELS")
        configure_models()

        print("\nSTEP 3: LOADING DOCUMENTS")
        documents = load_documents()

        print("\nSTEP 4: BUILDING IN-MEMORY INDEX")
        memory_index = build_memory_index(documents)

        print("\nSTEP 5: BUILDING CHROMADB INDEX")
        chroma_index = build_chroma_index(documents)

        print("\nSTEP 6: CREATING QUERY ENGINES")
        memory_query_engine = create_query_engine(memory_index)
        chroma_query_engine = create_query_engine(chroma_index)

        print("\nSTEP 7: RUNNING 10 QUERIES ON IN-MEMORY INDEX")

        memory_results = run_queries(
            memory_query_engine,
            QUESTIONS,
            "IN-MEMORY VECTOR INDEX"
        )

        print("\nSTEP 8: RUNNING 10 QUERIES ON CHROMADB")

        chroma_results = run_queries(
            chroma_query_engine,
            QUESTIONS,
            "CHROMADB VECTOR INDEX"
        )

        print("\nSTEP 9: SAVING RESULTS")

        memory_df, chroma_df, comparison = save_results(
            memory_results,
            chroma_results
        )

        print("\nSTEP 10: DISPLAYING COMPARISON")

        display_comparison(
            memory_df,
            chroma_df,
            comparison
        )

        print("\n" + "=" * 65)
        print("W7D3 EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 65)

    except Exception as error:

        print("\n" + "=" * 65)
        print("ERROR DURING EXECUTION")
        print("=" * 65)

        print(type(error).__name__, ":", error)

        raise


if __name__ == "__main__":
    main()
