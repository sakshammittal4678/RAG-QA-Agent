"""
chain.py
Builds the RAG prompt and the runnable chain (retrieve -> format -> prompt -> generate -> parse).
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

API_KEY=st.secrets["GOOGLE_API_KEY"]
LLM_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.2

PROMPT_TEMPLATE = """
You are a helpful assistant.
Answer ONLY from the provided document context. Use the chat history only to
resolve conversational references (e.g. "it", "that", follow-ups) — not as a
source of facts.
If the context is insufficient, just say you don't know.

Chat History:
{chat_history}

Context:
{context}

Question: {question}
"""


def format_docs(retrieved_docs):
    """Joins retrieved chunks into a single context string, with source tags."""
    parts = []
    for doc in retrieved_docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


def format_chat_history(history):
    """history: list of {'question': str, 'answer': str} dicts, oldest first."""
    if not history:
        return "None"
    lines = []
    for turn in history:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


def build_chain(retriever, model=LLM_MODEL, temperature=TEMPERATURE):
    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature,google_api_key=API_KEY)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question", "chat_history"],
    )
    parser = StrOutputParser()

    # input is now a dict: {"question": str, "chat_history": list[dict]}
    parallel_chain = RunnableParallel({
        "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
        "question": RunnableLambda(lambda x: x["question"]),
        "chat_history": RunnableLambda(lambda x: format_chat_history(x.get("chat_history", []))),
    })

    return parallel_chain | prompt | llm | parser


def get_answer(chain, question: str, chat_history: list = None) -> str:
    return chain.invoke({"question": question, "chat_history": chat_history or []})


def stream_answer(chain, question: str, chat_history: list = None):
    """Yields answer chunks as they're generated."""
    for chunk in chain.stream({"question": question, "chat_history": chat_history or []}):
        yield chunk
