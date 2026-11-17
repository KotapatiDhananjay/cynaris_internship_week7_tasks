
from pathlib import Path
import shutil

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENT_DIR = BASE_DIR / "documents"
DB_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "nomic-embed-text"


def load_documents():
    """Load all TXT and PDF documents."""

    documents = []

    for file_path in DOCUMENT_DIR.iterdir():

        if file_path.suffix.lower() == ".txt":
            text = file_path.read_text(encoding="utf-8")

        elif file_path.suffix.lower() == ".pdf":
            reader = PdfReader(str(file_path))
            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

        else:
            continue

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path.name}
                )
            )

    if not documents:
        raise ValueError("No readable PDF or TXT documents found.")

    return documents


def create_vector_store(reset=False):
    """Create or load a persistent ChromaDB vector store."""

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    if reset and DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    if DB_DIR.exists() and any(DB_DIR.iterdir()):
        return Chroma(
            collection_name="multi_document_rag",
            persist_directory=str(DB_DIR),
            embedding_function=embeddings
        )

    documents = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )

    chunks = splitter.split_documents(documents)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="multi_document_rag",
        persist_directory=str(DB_DIR)
    )

    print(f"Loaded {len(documents)} documents.")
    print(f"Created {len(chunks)} chunks.")

    return vector_store
