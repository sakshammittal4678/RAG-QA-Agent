"""
loader.py
Loads uploaded documents (PDF, TXT, DOCX) into LangChain Document objects.
"""

import os
import tempfile
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def _get_loader_for_file(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx":
        return Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_documents_from_paths(file_paths: list[str]):
    """Load documents from a list of file paths on disk."""
    documents = []
    for path in file_paths:
        loader = _get_loader_for_file(path)
        documents.extend(loader.load())
    return documents


def load_documents_from_uploaded_files(uploaded_files):
    """
    Load documents from Streamlit UploadedFile objects.
    Writes each to a temp file first since LangChain loaders need a file path.
    """
    documents = []
    for uploaded_file in uploaded_files:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {uploaded_file.name}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        loader = _get_loader_for_file(tmp_path)
        docs = loader.load()

        # tag each chunk with the original filename for source display
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name

        documents.extend(docs)
        os.remove(tmp_path)

    return documents
