
import json
import time
import argparse
from pathlib import Path

import mlflow

from src.workflow import ask_question


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Use SQLite instead of the deprecated file-store backend.
DATABASE_PATH = BASE_DIR / "mlflow.db"
mlflow.set_tracking_uri(f"sqlite:///{DATABASE_PATH.as_posix()}")

mlflow.set_experiment("W7D5_Multi_Document_RAG")


def run_query(question):
    """Run one RAG query and track its metrics."""

    start_time = time.time()

    with mlflow.start_run():
        mlflow.log_param("llm_model", "qwen2.5:3b")
        mlflow.log_param("embedding_model", "nomic-embed-text")
        mlflow.log_param("retrieval_top_k", 3)
        mlflow.log_param("question", question)

        result = ask_question(question)

        latency = time.time() - start_time

        mlflow.log_metric("latency_seconds", latency)
        mlflow.log_metric(
            "retrieved_source_count",
            len(result["sources"])
        )

        mlflow.log_text(result["answer"], "answer.txt")

        record = {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "latency_seconds": round(latency, 2)
        }

        output_file = OUTPUT_DIR / "query_results.json"

        existing = []
        if output_file.exists():
            existing = json.loads(
                output_file.read_text(encoding="utf-8")
            )

        existing.append(record)

        output_file.write_text(
            json.dumps(existing, indent=4, ensure_ascii=False),
            encoding="utf-8"
        )

    return result, latency


def run_evaluation():
    """Run Ragas evaluation using sample questions and reference answers."""

    from src.evaluation import evaluate_rag
    from src.ingestion import create_vector_store

    test_cases = [
        {
            "question": "What is artificial intelligence?",
            "reference": (
                "Artificial intelligence is a branch of computer science "
                "that creates machines capable of tasks requiring human intelligence."
            )
        },
        {
            "question": "What are the three types of machine learning?",
            "reference": (
                "The three types are supervised learning, "
                "unsupervised learning, and reinforcement learning."
            )
        },
        {
            "question": "What is retrieval-augmented generation?",
            "reference": (
                "RAG combines information retrieval with generative AI. "
                "It retrieves relevant information and gives it to a language model."
            )
        }
    ]

    vector_store = create_vector_store()
    samples = []

    for case in test_cases:
        print(f"\nEvaluating: {case['question']}")

        result, _ = run_query(case["question"])

        docs = vector_store.similarity_search(
            case["question"],
            k=3
        )

        samples.append({
            "user_input": case["question"],
            "retrieved_contexts": [
                doc.page_content for doc in docs
            ],
            "response": result["answer"],
            "reference": case["reference"]
        })

    print("\nRunning Ragas evaluation...")

    scores = evaluate_rag(samples)

    print("\nRAGAS EVALUATION RESULTS")
    print(scores)

    score_file = OUTPUT_DIR / "ragas_evaluation.txt"
    score_file.write_text(str(scores), encoding="utf-8")

    print(f"\nEvaluation saved to: {score_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Document RAG System"
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run Ragas evaluation"
    )

    args = parser.parse_args()

    print("\nMULTI-DOCUMENT RAG SYSTEM")
    print("=" * 45)

    if args.evaluate:
        try:
            run_evaluation()
        except Exception as error:
            print(f"\nEvaluation failed: {error}")
        return

    print("Type 'exit' to stop.\n")

    while True:
        question = input("Enter your question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            result, latency = run_query(question)

            print("\nANSWER:")
            print(result["answer"])

            print("\nSOURCES:")
            for source in result["sources"]:
                print("-", source)

            print(f"\nLatency: {latency:.2f} seconds")
            print("-" * 45)

        except Exception as error:
            print(f"\nError: {error}")
            print("Check Ollama, the models, and the application configuration.")


if __name__ == "__main__":
    main()
