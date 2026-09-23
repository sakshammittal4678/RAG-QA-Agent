"""
app.py
Streamlit frontend for the multi-document RAG QA system.
Run with: streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from loader import load_documents_from_uploaded_files
from vector_store import split_documents, build_vector_store, get_retriever
from chain import build_chain, stream_answer

load_dotenv()
if not os.getenv("GOOGLE_API_KEY") and "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="Multi-Doc RAG QA", page_icon="📄")
st.title("📄 Multi-Document RAG QA")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY not found. Add it to a .env file in this folder.")
    st.stop()

# session state holds the chain across reruns
if "chain" not in st.session_state:
    st.session_state.chain = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"question": str, "answer": str}

with st.sidebar:
    st.header("Upload documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if st.button("Process documents", disabled=not uploaded_files):
        with st.spinner("Loading and indexing documents..."):
            documents = load_documents_from_uploaded_files(uploaded_files)
            chunks = split_documents(documents)
            vector_store = build_vector_store(chunks)
            retriever = get_retriever(vector_store, k=4)
            st.session_state.chain = build_chain(retriever)
            st.session_state.processed_files = [f.name for f in uploaded_files]
            st.session_state.chat_history = []
        st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.processed_files:
        st.caption("Indexed files:")
        for name in st.session_state.processed_files:
            st.caption(f"- {name}")

st.divider()

if st.session_state.chain is None:
    st.info("Upload and process documents in the sidebar to start asking questions.")
else:
    # replay prior turns
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])

    question = st.chat_input("Ask a question about your documents")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            answer = st.write_stream(
                stream_answer(
                    st.session_state.chain,
                    question,
                    chat_history=st.session_state.chat_history,
                )
            )

        st.session_state.chat_history.append({"question": question, "answer": answer})
