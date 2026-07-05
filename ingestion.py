import os
import tempfile

import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import Tool

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 400
RETRIEVER_K = 2


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


def make_retriever_tool(vectorstore, bm25_retriever, company_name: str, safe_name: str) -> Tool:
    """
    Wrap one company's slice of the shared vectorstore (via metadata filter) plus its
    own BM25 index as a single agent tool. Isolation comes from the "company" filter,
    not from a separate Chroma collection per file.
    """

    def retrieve(input):
        if isinstance(input, dict):
            query = input.get("query") or input.get("input") or str(input)
        else:
            query = input

        chroma_hits = vectorstore.similarity_search(query, k=RETRIEVER_K, filter={"company": company_name})
        bm25_hits = bm25_retriever.invoke(query)

        seen, chunks = set(), []
        for doc in chroma_hits + bm25_hits:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                chunks.append(doc.page_content)
        return "\n\n".join(chunks)

    return Tool(
        name=f"search_{safe_name}_report",
        func=retrieve,
        description=(
            f"MANDATORY: Call this tool for ALL data about {company_name}. "
            f"This is the ONLY source for {company_name} financials. "
            f"Input: a plain string search query like 'total revenue 2024'."
        ),
    )


@st.cache_resource(show_spinner=False)
def create_retrievers(_llm, uploaded_files):
    # _llm has a leading underscore so Streamlit doesn't try to hash it for the cache key
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)  # one model instance, shared by every file

    all_documents = []
    per_company_documents = {}
    company_meta = []

    for uploaded in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded.getvalue())
            fp = tmp_file.name

        try:
            # "fast" = direct text extraction, no vision/layout-detection model.
            # Much lighter on memory and faster; fine for text-based financial PDFs.
            loader = UnstructuredPDFLoader(fp, strategy="fast")
            docs = loader.load()
        finally:
            os.remove(fp)

        company_name = extract_company_name(_llm, docs, uploaded.name)
        safe_name = "".join(c if c.isalnum() else "_" for c in uploaded.name)

        documents = text_splitter.split_documents(docs)
        for chunk in documents:
            chunk.page_content = f"【Source Report: {uploaded.name}】\n\n" + chunk.page_content
            chunk.metadata["company"] = company_name

        all_documents.extend(documents)
        per_company_documents.setdefault(company_name, []).extend(documents)
        company_meta.append((company_name, safe_name, uploaded.name))

    if not all_documents:
        return [], None, []

    # One shared vectorstore for every uploaded report. Per-company isolation happens
    # via the "company" metadata filter at query time, not via separate Chroma collections
    # or separate embedding-model instances.
    vectorstore = Chroma.from_documents(all_documents, embedding=embeddings)

    tools = []
    for company_name, safe_name, _ in company_meta:
        bm25_retriever = BM25Retriever.from_documents(per_company_documents[company_name])
        bm25_retriever.k = RETRIEVER_K
        tools.append(make_retriever_tool(vectorstore, bm25_retriever, company_name, safe_name))

    return tools, vectorstore, company_meta
