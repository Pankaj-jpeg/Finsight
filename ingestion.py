import os
import tempfile

import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.tools import Tool

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 400
RETRIEVER_K = 2


def make_retriever_tool(retriever, company_name: str, safe_name: str) -> Tool:
    """Wrap a retriever as an agent Tool scoped to one company's report."""

    def retrieve(input):
        # Handle both string and dict input from the model
        if isinstance(input, dict):
            query = input.get("query") or input.get("input") or str(input)
        else:
            query = input
        docs = retriever.invoke(query)
        return "\n\n".join(d.page_content for d in docs)

    return Tool(
        name=f"search_{safe_name}_report",
        func=retrieve,
        description=(
            f"MANDATORY: Call this tool for ALL data about {company_name}. "
            f"This is the ONLY source for {company_name} financials. "
            f"Input: a plain string search query like 'total revenue 2024'."
        ),
    )


def extract_company_name(llm, docs, filename: str) -> str:
    """Ask the LLM to name the company from the first page; fall back to the filename."""
    try:
        first_page = docs[0].page_content[:500]
        response = llm.invoke(
            f"Extract only the company name from this text. "
            f"Reply with just the company name, nothing else, no punctuation:\n\n{first_page}"
        )
        name = response.content.strip()
        if len(name) > 30 or len(name) < 2:
            raise ValueError("Bad extraction")
        return name
    except Exception:
        name = filename.rsplit(".", 1)[0]
        for suffix in ["_10K", "_10k", "_annual", "_report", "_2024", "_2023"]:
            name = name.replace(suffix, "")
        return name.strip("_")


@st.cache_resource(show_spinner=False)
def create_retrievers(_llm, uploaded_files) -> list[Tool]:
    # _llm has a leading underscore so Streamlit doesn't try to hash it for the cache key
    tools = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded.getvalue())
            fp = tmp_file.name

        try:
            loader = UnstructuredPDFLoader(fp)
            docs = loader.load()
        finally:
            os.remove(fp)

        documents = text_splitter.split_documents(docs)
        for chunk in documents:
            chunk.page_content = f"【Source Report: {uploaded.name}】\n\n" + chunk.page_content

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma.from_documents(documents, embedding=embeddings)
        chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = RETRIEVER_K

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5],
        )

        safe_name = "".join(c if c.isalnum() else "_" for c in uploaded.name)
        company_name = extract_company_name(_llm, docs, uploaded.name)
        tools.append(make_retriever_tool(ensemble_retriever, company_name, safe_name))

    return tools
