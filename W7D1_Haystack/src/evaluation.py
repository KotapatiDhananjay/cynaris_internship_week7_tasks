
import csv
from pathlib import Path

from src.pipeline import search_bm25, search_dense


QUESTIONS = (
    "What is artificial intelligence?",
    "What are the main categories of machine learning?",
    "What is supervised learning?",
    "What are the common tasks in natural language processing?",
    "Explain tokenization and sentiment analysis.",
    "How does the BM25 ranking algorithm work?",
    "What is dense retrieval?",
    "What is Retrieval Augmented Generation?",
    "Explain the typical workflow of a RAG system.",
    "Why is retrieval quality important in RAG?"
)


def evaluate(document_store, output_dir="outputs"):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bm25_pipeline = __import__(
        "src.pipeline", fromlist=["build_bm25_pipeline"]
    ).build_bm25_pipeline(document_store)

    dense_pipeline = __import__(
        "src.pipeline", fromlist=["build_dense_pipeline"]
    ).build_dense_pipeline(document_store)

    rows = []

    for question_id, question in enumerate(QUESTIONS, start=1):

        print("\n" + "=" * 65)
        print(f"QUESTION {question_id}: {question}")

        for method, pipeline, search_function in (
            ("BM25", bm25_pipeline, search_bm25),
            ("Dense", dense_pipeline, search_dense)
        ):

            print(f"\n--- {method} RETRIEVAL ---")

            results = search_function(pipeline, question)

            for rank, result in enumerate(results, start=1):

                print(f"\nRank: {rank}")
                print(f"Source: {result['source']}")
                print(f"Score: {result['score']}")
                print(f"Content: {result['answer'][:250]}")

                while True:
                    relevance = input(
                        "Is this result relevant? (1=Yes, 0=No): "
                    ).strip()

                    if relevance in ("0", "1"):
                        break

                    print("Please enter only 1 or 0.")

                rows.append({
                    "question_id": question_id,
                    "question": question,
                    "method": method,
                    "rank": rank,
                    "source": result["source"],
                    "score": result["score"],
                    "relevant": int(relevance)
                })

    csv_file = output_path / "retrieval_results.csv"

    with csv_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question_id",
                "question",
                "method",
                "rank",
                "source",
                "score",
                "relevant"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    calculate_precision(rows, output_path)

    print("\nEvaluation completed!")
    print("Results:", csv_file)


def calculate_precision(rows, output_path):

    metrics = []

    for method in ("BM25", "Dense"):

        method_rows = [
            row for row in rows
            if row["method"] == method
        ]

        question_scores = []

        for question_id in range(1, len(QUESTIONS) + 1):

            question_rows = sorted(
                [
                    row for row in method_rows
                    if row["question_id"] == question_id
                ],
                key=lambda row: row["rank"]
            )

            top_results = question_rows[:3]

            if not top_results:
                continue

            precision = (
                sum(row["relevant"] for row in top_results)
                / len(top_results)
            )

            question_scores.append(precision)

            metrics.append({
                "method": method,
                "question_id": question_id,
                "precision_at_3": round(precision, 4)
            })

        average = (
            sum(question_scores) / len(question_scores)
            if question_scores else 0
        )

        print(
            f"\n{method} Average Precision@3: "
            f"{average:.2%}"
        )

    metrics_file = output_path / "precision_comparison.csv"

    with metrics_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "question_id",
                "precision_at_3"
            ]
        )

        writer.writeheader()
        writer.writerows(metrics)

    print("\nComparison saved:", metrics_file)
