# rag_pipeline.py
# Builds a FAISS vector store from patient PDF reports
# and provides semantic search over them.

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import PATIENT_REPORTS_DIR, FAISS_INDEX_DIR, EMBEDDING_MODEL

_vectorstore = None
_embeddings = None


def _get_embeddings():
    """Lazy-load embedding model so we don't reload it every call."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def build_vector_store(force_rebuild: bool = False):
    """
    Load patient report PDFs, chunk them, and build a FAISS index.
    If the index already exists on disk and force_rebuild is False,
    just load it from there instead.
    """
    global _vectorstore

    # if we already have an index on disk, just load it
    if not force_rebuild and os.path.exists(FAISS_INDEX_DIR):
        _vectorstore = FAISS.load_local(
            FAISS_INDEX_DIR,
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        return _vectorstore

    documents = []
    for filename in os.listdir(PATIENT_REPORTS_DIR):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(PATIENT_REPORTS_DIR, filename)
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = filename
            documents.extend(docs)

    if not documents:
        raise FileNotFoundError(
            f"No PDF files found in {PATIENT_REPORTS_DIR}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = _get_embeddings()
    _vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    _vectorstore.save_local(FAISS_INDEX_DIR)

    return _vectorstore


def get_vector_store():
    """Return the vector store (builds it first time if needed)."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_vector_store()
    return _vectorstore


def retrieve_patient_info(query: str, k: int = 4) -> list[dict]:
    """Run a similarity search and return the top-k chunks with scores."""
    vs = get_vector_store()
    results = vs.similarity_search_with_score(query, k=k)
    output = []
    for doc, score in results:
        output.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source_file", "unknown"),
            "page": doc.metadata.get("page", 0),
            "relevance_score": round(float(1 - score), 4),
        })
    return output
