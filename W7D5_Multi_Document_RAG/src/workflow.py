
from typing import TypedDict, List

from langgraph.graph import StateGraph, END

from src.ingestion import create_vector_store
from src.agents import generate_answer


class RAGState(TypedDict):
    question: str
    context: str
    sources: List[str]
    answer: str


def build_rag_graph():
    """Build the multi-document RAG workflow."""

    vector_store = create_vector_store()

    def retrieve_documents(state: RAGState):
        results = vector_store.similarity_search(
            state["question"],
            k=3
        )

        context_parts = []
        sources = []

        for document in results:
            context_parts.append(document.page_content)
            sources.append(document.metadata.get("source", "Unknown"))

        return {
            "context": "\n\n".join(context_parts),
            "sources": sorted(set(sources))
        }

    def generate_response(state: RAGState):
        answer = generate_answer(
            state["question"],
            state["context"]
        )

        return {"answer": answer}

    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_documents)
    graph.add_node("generate", generate_response)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def ask_question(question):
    """Execute the complete RAG workflow."""

    app = build_rag_graph()

    result = app.invoke({
        "question": question,
        "context": "",
        "sources": [],
        "answer": ""
    })

    return result
