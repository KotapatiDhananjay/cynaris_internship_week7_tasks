import time
import requests
import pandas as pd
from pathlib import Path


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OLLAMA_URL = "http://localhost:11434"

MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b"
]

SYSTEM_PROMPT = """
You are a helpful AI assistant acting as a technical tutor
for an undergraduate Information Technology student.

Instructions:
1. Explain concepts in simple English.
2. Keep answers clear and technically correct.
3. Use examples wherever useful.
4. Avoid unnecessary complicated terminology.
5. Structure answers using short points when appropriate.
6. If you do not know something, say so honestly.
"""

FIVE_PROMPTS = [
    "What is artificial intelligence? Explain with two examples.",
    "Explain supervised and unsupervised machine learning.",
    "What is Retrieval Augmented Generation (RAG)?",
    "Explain vector databases and embeddings.",
    "What is the difference between NLP and traditional programming?"
]

COMPARISON_QUESTIONS = [
    "What is the difference between artificial intelligence and machine learning?",
    "Explain how Retrieval Augmented Generation works.",
    "What are the advantages and limitations of running an LLM locally?"
]


# ==========================================
# CHECK OLLAMA CONNECTION
# ==========================================

def check_ollama():

    print("\nChecking Ollama connection...")

    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=10
        )

        response.raise_for_status()

        available_models = [
            model["name"]
            for model in response.json().get("models", [])
        ]

        print("Ollama is running.")

        print("\nAvailable models:")

        for model in available_models:
            print("-", model)

        for required_model in MODELS:

            if required_model not in available_models:
                raise RuntimeError(
                    f"Required model {required_model} is not installed. "
                    f"Run: ollama pull {required_model}"
                )

        return True

    except requests.exceptions.RequestException as error:

        raise RuntimeError(
            "Cannot connect to Ollama. Start Ollama and try again. "
            f"Details: {error}"
        )


# ==========================================
# CALL OLLAMA CHAT API
# ==========================================

def generate_response(model, prompt):

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "keep_alive": 0,
        "options": {
            "temperature": 0.2,
            "num_ctx": 1024,
            "num_predict": 150
        }
    }

    start_time = time.perf_counter()

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=300
    )

    latency = time.perf_counter() - start_time

    response.raise_for_status()

    result = response.json()

    answer = result["message"]["content"].strip()

    return answer, latency


# ==========================================
# FIRST LOCAL INFERENCE
# ==========================================

def first_inference():

    print("\n" + "=" * 65)
    print("FIRST LOCAL INFERENCE")
    print("=" * 65)

    prompt = "Explain artificial intelligence in simple English."

    answer, latency = generate_response(
        "llama3.2:3b",
        prompt
    )

    print("\nModel: llama3.2:3b")
    print("Prompt:", prompt)
    print("\nResponse:")
    print(answer)
    print(f"\nResponse time: {latency:.2f} seconds")


# ==========================================
# TEST FIVE PROMPTS
# ==========================================

def test_five_prompts():

    print("\n" + "=" * 65)
    print("TESTING FIVE PROMPTS WITH LLAMA3.2:3B")
    print("=" * 65)

    results = []

    for number, prompt in enumerate(FIVE_PROMPTS, start=1):

        print("\n" + "-" * 65)
        print(f"PROMPT {number}: {prompt}")

        answer, latency = generate_response(
            "llama3.2:3b",
            prompt
        )

        print("\nResponse:")
        print(answer)
        print(f"\nLatency: {latency:.2f} seconds")

        results.append({
            "prompt_number": number,
            "model": "llama3.2:3b",
            "prompt": prompt,
            "response": answer,
            "latency_seconds": round(latency, 3)
        })

    df = pd.DataFrame(results)

    df.to_csv(
        OUTPUT_DIR / "five_prompt_results.csv",
        index=False
    )

    print("\nFive-prompt results saved.")


# ==========================================
# COMPARE TWO MODELS
# ==========================================

def compare_models():

    print("\n" + "=" * 65)
    print("LLAMA3.2:3B VS QWEN2.5:3B")
    print("=" * 65)

    results = []

    for question_number, question in enumerate(
        COMPARISON_QUESTIONS,
        start=1
    ):

        print("\n" + "=" * 65)
        print(f"QUESTION {question_number}: {question}")
        print("=" * 65)

        for model in MODELS:

            print(f"\nRunning model: {model}")

            answer, latency = generate_response(
                model,
                question
            )

            print("\nResponse:")
            print(answer)

            print(f"\nLatency: {latency:.2f} seconds")

            results.append({
                "question_number": question_number,
                "question": question,
                "model": model,
                "response": answer,
                "latency_seconds": round(latency, 3)
            })

    df = pd.DataFrame(results)

    df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    print("\nModel comparison saved.")

    return df


# ==========================================
# GENERATE COMPARISON SUMMARY
# ==========================================

def create_summary(df):

    print("\n" + "=" * 65)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 65)

    summary = (
        df.groupby("model")["latency_seconds"]
        .mean()
        .reset_index()
    )

    summary.rename(
        columns={
            "latency_seconds": "average_latency_seconds"
        },
        inplace=True
    )

    summary.to_csv(
        OUTPUT_DIR / "model_summary.csv",
        index=False
    )

    print(summary.to_string(index=False))

    print("\nManual quality evaluation is required.")

    print(
        "Compare correctness, relevance, clarity, "
        "completeness and practical examples."
    )


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("=" * 65)
    print("W7D4 - OLLAMA LOCAL LLM INFERENCE")
    print("=" * 65)

    try:

        check_ollama()

        first_inference()

        test_five_prompts()

        comparison_df = compare_models()

        create_summary(comparison_df)

        print("\n" + "=" * 65)
        print("W7D4 EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 65)

        print("\nGenerated files:")
        print("outputs/five_prompt_results.csv")
        print("outputs/model_comparison.csv")
        print("outputs/model_summary.csv")

    except requests.exceptions.Timeout:

        print("\nERROR: Ollama took too long to respond.")
        print("Close other applications and try again.")

        raise

    except requests.exceptions.RequestException as error:

        print("\nOLLAMA API ERROR:")
        print(error)

        raise

    except Exception as error:

        print("\nEXECUTION ERROR:")
        print(type(error).__name__, ":", error)

        raise


if __name__ == "__main__":
    main()