@'
# W7D4 - Ollama Local LLM Inference

## Objective
Run local Large Language Models using Ollama and compare their responses.

## Technologies
- Python
- Ollama
- Llama 3.2 3B
- Qwen 2.5 3B
- REST API
- Requests
- Pandas

## Implementation
1. Verified Ollama installation.
2. Checked locally installed models.
3. Created a Python script to call the Ollama Chat API.
4. Configured a custom system prompt.
5. Tested five prompts using Llama 3.2 3B.
6. Compared Llama 3.2 3B and Qwen 2.5 3B on three identical questions.
7. Recorded response latency.
8. Saved model responses and comparison results.

## Output Files
- outputs/five_prompt_results.csv
- outputs/model_comparison.csv
- outputs/model_summary.csv

## Execution

Install dependencies:

pip install -r requirements.txt

Run:

python main.py

## Model Evaluation

Manually evaluate both models based on:
- Accuracy
- Relevance
- Clarity
- Completeness
- Examples
- Response latency

## Note
Latency depends on hardware, available RAM, model loading
and system resources. Models are executed sequentially
to reduce memory usage.
'@ | Set-Content -Encoding UTF8 README.md