
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

DOCUMENTS = {
    "document1.pdf": (
        "Introduction to Artificial Intelligence",
        """
        Artificial Intelligence is a branch of computer science that
        develops systems capable of performing tasks that normally
        require human intelligence. These tasks include learning,
        reasoning, problem solving, language understanding and
        decision making.

        Machine learning is a major area of artificial intelligence.
        It allows computers to learn patterns from data without
        explicitly programming every rule. Applications include
        recommendation systems, fraud detection and image recognition.
        """
    ),
    "document2.pdf": (
        "Machine Learning Fundamentals",
        """
        Machine learning is the process of training computer models
        using data. The three common categories are supervised learning,
        unsupervised learning and reinforcement learning.

        Supervised learning uses labelled data for classification
        and regression. Unsupervised learning discovers hidden patterns
        in unlabelled data. Reinforcement learning trains agents using
        rewards and penalties.

        Model evaluation measures how well a trained model performs
        on unseen data. Accuracy, precision, recall and F1 score are
        common evaluation metrics.
        """
    ),
    "document3.pdf": (
        "Natural Language Processing",
        """
        Natural Language Processing, also called NLP, enables computers
        to understand, process and generate human language.

        Common NLP tasks include tokenization, stemming, lemmatization,
        sentiment analysis, text classification and named entity
        recognition.

        Tokenization divides text into smaller units called tokens.
        Sentiment analysis identifies opinions and emotions in text.
        NLP is used in chatbots, search engines and translation systems.
        """
    ),
    "document4.pdf": (
        "Information Retrieval Systems",
        """
        Information retrieval systems find relevant documents based
        on a user's search query. Search engines are common examples.

        BM25 is a keyword-based ranking algorithm. It considers
        term frequency, inverse document frequency and document length.
        It is effective when important query keywords appear in documents.

        Dense retrieval converts documents and queries into numerical
        embeddings. Similarity between embeddings helps identify
        semantically related content even when exact keywords differ.
        """
    ),
    "document5.pdf": (
        "Retrieval Augmented Generation",
        """
        Retrieval Augmented Generation, known as RAG, combines
        information retrieval with generative language models.

        A RAG system first retrieves relevant information from a
        document collection. The retrieved context is then supplied
        to a language model to generate a grounded response.

        A typical RAG workflow includes document loading, cleaning,
        chunking, embedding generation, document storage, retrieval
        and answer generation.

        RAG can reduce unsupported answers by providing relevant
        external context to the language model. Retrieval quality
        is important for producing accurate responses.
        """
    )
}


def create_pdfs():

    styles = getSampleStyleSheet()

    for filename, (title, content) in DOCUMENTS.items():

        path = DATA_DIR / filename

        document = SimpleDocTemplate(str(path))

        story = [
            Paragraph(title, styles["Title"]),
            Spacer(1, 20)
        ]

        for paragraph in content.strip().split("\n\n"):
            story.append(
                Paragraph(paragraph.strip(), styles["BodyText"])
            )
            story.append(Spacer(1, 12))

        document.build(story)

        print(f"Created: {path}")


if __name__ == "__main__":
    create_pdfs()
