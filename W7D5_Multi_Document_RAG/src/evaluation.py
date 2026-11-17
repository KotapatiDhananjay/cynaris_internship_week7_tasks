
def evaluate_rag(samples):
    """Evaluate RAG responses using Ragas."""

    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    from langchain_ollama import ChatOllama, OllamaEmbeddings

    dataset = EvaluationDataset.from_list(samples)

    evaluator_llm = LangchainLLMWrapper(
        ChatOllama(
            model="qwen2.5:3b",
            temperature=0,
            num_ctx=1024
        )
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="nomic-embed-text")
    )

    results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall()
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    return results
