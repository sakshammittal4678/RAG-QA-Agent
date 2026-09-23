"""
vector_store.py
Splits documents into chunks, embeds them, and builds/saves/loads a FAISS index.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "models/gemini-embedding-001"


def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks, embedding_model=EMBEDDING_MODEL):
    """Builds an in-memory FAISS vector store from document chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def save_vector_store(vector_store: FAISS, path: str):
    vector_store.save_local(path)


def load_vector_store(path: str, embedding_model=EMBEDDING_MODEL) -> FAISS:
    embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def get_retriever(vector_store: FAISS, k: int = 4):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
