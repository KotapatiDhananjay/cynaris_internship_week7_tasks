
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

DATA = Path("data")

DOCUMENTS = {
    "ai.pdf": (
        "Artificial Intelligence",
        """
        Artificial Intelligence is a branch of computer science
        that develops intelligent systems capable of learning,
        reasoning, problem solving and decision making.

        Machine learning is a subset of artificial intelligence.
        Deep learning uses neural networks with multiple layers.
        AI applications include robotics, chatbots and automation.
        """
    ),
    "ml.pdf": (
        "Machine Learning",
        """
        Machine Learning enables computers to learn patterns
        from data without explicitly programming every rule.

        Supervised learning uses labelled datasets.
        Unsupervised learning identifies hidden patterns.
        Reinforcement learning uses rewards and penalties.

        Classification predicts categories while regression
        predicts continuous numerical values.
        Accuracy, precision, recall and F1 score are evaluation
        metrics used in machine learning.
        """
    ),
    "nlp.pdf": (
        "Natural Language Processing",
        """
        Natural Language Processing enables computers to
        understand and process human language.

        Tokenization divides text into smaller units.
        Stemming reduces words to their root forms.
        Lemmatization converts words into their dictionary forms.

        Sentiment analysis identifies opinions and emotions.
        Named entity recognition identifies people, locations
        and organizations in text.
        """
    ),
    "retrieval.pdf": (
        "Information Retrieval",
        """
        Information retrieval finds relevant documents based
        on a search query.

        BM25 is a keyword-based retrieval algorithm.
        It considers term frequency, inverse document frequency
        and document length when ranking search results.

        Dense retrieval uses vector embeddings to represent
        documents and queries. Similarity calculations identify
        semantically related information.
        """
    ),
    "rag.pdf": (
        "Retrieval Augmented Generation",
        """
        Retrieval Augmented Generation combines information
        retrieval with generative language models.

        A RAG system retrieves relevant documents and provides
        their content as context to a language model.

        The RAG workflow includes document loading, cleaning,
        chunking, embedding, indexing, retrieval and generation.

        Good retrieval quality improves answer accuracy and
        reduces unsupported responses.
        """
    )
}


def create_pdfs():

    DATA.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()

    for filename, (title, content) in DOCUMENTS.items():

        story = [
            Paragraph(title, styles["Title"]),
            Spacer(1, 20)
        ]

        for paragraph in content.strip().split("\n\n"):
            story.append(
                Paragraph(paragraph.strip(), styles["BodyText"])
            )
            story.append(Spacer(1, 12))

        SimpleDocTemplate(
            str(DATA / filename),
            pagesize=A4
        ).build(story)

        print(f"Created {filename}")


if __name__ == "__main__":
    create_pdfs()
