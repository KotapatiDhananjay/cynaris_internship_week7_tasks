
from crewai import Agent, Task, Crew, LLM


def generate_answer(question, context):
    """Use a CrewAI agent to answer from retrieved document context."""

    local_llm = LLM(
        model="ollama/qwen2.5:3b",
        base_url="http://localhost:11434",
        temperature=0.1,
        max_tokens=250
    )

    researcher = Agent(
        role="Document Research Assistant",
        goal="Answer questions accurately using only the supplied documents.",
        backstory=(
            "You are a careful AI research assistant. "
            "You read document extracts and provide clear answers. "
            "You never invent facts that are not in the context."
        ),
        llm=local_llm,
        allow_delegation=False,
        verbose=False
    )

    answer_task = Task(
        description=f"""
        Answer the following question using only the context below.

        Question:
        {question}

        Retrieved context:
        {context}

        Instructions:
        1. Use only the retrieved context.
        2. Give a clear answer in simple English.
        3. If the context does not contain the answer, say so.
        4. Do not invent information.
        """,
        expected_output="A concise, accurate answer based on the documents.",
        agent=researcher
    )

    crew = Crew(
        agents=[researcher],
        tasks=[answer_task],
        verbose=False
    )

    result = crew.kickoff()
    return str(result)
