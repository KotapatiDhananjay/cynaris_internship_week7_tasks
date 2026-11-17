
# W7D1 - Haystack Pipeline Architecture

## Objective
Build a document retrieval pipeline using Haystack.

## Technologies
- Python
- Haystack 2.x
- BM25 Retriever
- Dense Retriever
- Sentence Transformers
- PyPDF
- ReportLab

## Architecture

PDF Documents
      |
PDF Converter
      |
Document Cleaner
      |
Document Splitter
      |
Document Embedder
      |
InMemoryDocumentStore
      |
      |---------------------|
      |                     |
BM25 Retriever       Dense Retriever
      |                     |
      |---------------------|
                |
              Reader
                |
         Retrieved Results

## Features
1. Index five PDF documents.
2. Convert and clean PDF content.
3. Split documents into chunks.
4. Generate document embeddings.
5. Retrieve using BM25.
6. Retrieve using DenseRetriever.
7. Run ten questions.
8. Manually evaluate relevance.
9. Calculate Precision@3.

## Installation

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt

## Execution

    python main.py

## Output

- outputs/retrieval_results.csv
- outputs/precision_comparison.csv

## Evaluation

Precision@3 = Relevant documents in top 3 / 3

Both retrievers use the same ten questions and the same
document collection.

## Limitations

- In-memory document storage.
- Manual relevance evaluation.
- Embedding model requires initial download.
- No generative LLM answer generation.

## Future Improvements

- Integrate Ragas for automated RAG evaluation.
- Add persistent vector storage.
- Add a generative question-answering component.
- Deploy the pipeline using FastAPI.
