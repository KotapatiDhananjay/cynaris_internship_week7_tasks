
# W7D2 - Haystack Retrieval

## Objective
Compare BM25 and Dense Retrieval using Haystack.

## Technologies
- Python
- Haystack 2.x
- BM25 Retriever
- Dense Retriever
- Sentence Transformers
- PyPDF
- ReportLab

## Pipeline
PDF -> Converter -> Cleaner -> Splitter
-> Document Embedder -> Document Store

Document Store -> BM25 Retriever -> Reader

Document Store -> Dense Retriever -> Reader

## Tasks
1. Generate five PDF documents.
2. Index PDF documents.
3. Run ten questions.
4. Retrieve top three documents using BM25.
5. Retrieve top three documents using Dense Retrieval.
6. Manually label relevance.
7. Compare Precision@3.

## Run

    pip install -r requirements.txt
    python main.py

## Outputs
- outputs/retrieval_results.csv
- outputs/precision_comparison.csv

## Precision@3

Relevant documents in top 3 / 3

## Future Improvements
- Integrate Ragas.
- Use persistent vector databases.
- Add LLM-based answer generation.
- Deploy using FastAPI.
    