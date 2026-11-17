
# W7D5 - Multi-Document RAG System

## Overview

A multi-document Retrieval-Augmented Generation system built using
CrewAI, LangGraph, ChromaDB, Ollama, MLflow and Ragas.

## Technologies

- Python
- CrewAI
- LangGraph
- ChromaDB
- Ollama
- Qwen 2.5 3B
- Nomic Embed Text
- MLflow
- Ragas
- Pytest

## Architecture

1. Load multiple PDF and TXT documents.
2. Split documents into smaller chunks.
3. Generate embeddings using Ollama.
4. Store embeddings in ChromaDB.
5. Retrieve relevant chunks for a question.
6. Use a CrewAI agent to generate a grounded answer.
7. Orchestrate retrieval and generation using LangGraph.
8. Track execution metrics using MLflow.
9. Evaluate RAG quality using Ragas.

## Setup

Create and activate a virtual environment.

Install dependencies:

    pip install -r requirements.txt

Make sure Ollama is running and these models are available:

    ollama pull qwen2.5:3b
    ollama pull nomic-embed-text

## Run

    python main.py

## Evaluate

    python main.py --evaluate

## Tests

    pytest -v

## MLflow

    mlflow ui

Open the local MLflow tracking page shown in the terminal.

## Output

- outputs/query_results.json
- outputs/ragas_evaluation.txt
- MLflow experiment logs
