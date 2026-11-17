
import csv
from pathlib import Path

from src.retrieval import (
    build_bm25,
    build_dense,
    retrieve_bm25,
    retrieve_dense,
    display_results
)

QUESTIONS = [
    "What is artificial intelligence?",
    "Explain machine learning.",
    "What is supervised learning?",
    "What is unsupervised learning?",
    "Explain tokenization in NLP.",
    "What is sentiment analysis?",
    "How does BM25 work?",
    "What is dense retrieval?",
    "What is retrieval augmented generation?",
    "Why is retrieval quality important in RAG?"
]


def run_evaluation(store):

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    bm25 = build_bm25(store)
    dense = build_dense(store)

    rows = []

    for qid, question in enumerate(QUESTIONS, start=1):

        print("\n" + "=" * 60)
        print(f"QUESTION {qid}: {question}")

        for method, pipeline, function in [
            ("BM25", bm25, retrieve_bm25),
            ("Dense", dense, retrieve_dense)
        ]:

            documents = function(pipeline, question)

            display_results(method, documents)

            for rank, document in enumerate(documents, start=1):

                while True:
                    label = input(
                        "\nRelevant? Enter 1 for Yes, 0 for No: "
                    ).strip()

                    if label in ("0", "1"):
                        break

                    print("Invalid input.")

                rows.append({
                    "question_id": qid,
                    "question": question,
                    "method": method,
                    "rank": rank,
                    "source": document.meta.get(
                        "file_path", "Unknown"
                    ),
                    "score": document.score,
                    "relevant": int(label)
                })

    save_results(rows, output_dir)
    calculate_precision(rows, output_dir)


def save_results(rows, output_dir):

    path = output_dir / "retrieval_results.csv"

    with path.open("w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nResults saved:", path)


def calculate_precision(rows, output_dir):

    comparison = []

    for method in ["BM25", "Dense"]:

        scores = []

        for qid in range(1, 11):

            results = sorted(
                [
                    row for row in rows
                    if row["method"] == method
                    and row["question_id"] == qid
                ],
                key=lambda row: row["rank"]
            )[:3]

            if results:
                precision = (
                    sum(row["relevant"] for row in results) / 3
                )

                scores.append(precision)

                comparison.append({
                    "method": method,
                    "question_id": qid,
                    "precision_at_3": round(precision, 4)
                })

        average = sum(scores) / len(scores) if scores else 0

        print(f"\n{method} Average Precision@3: {average:.2%}")

    path = output_dir / "precision_comparison.csv"

    with path.open("w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "question_id",
                "precision_at_3"
            ]
        )

        writer.writeheader()
        writer.writerows(comparison)

    print("\nPrecision comparison saved:", path)
    