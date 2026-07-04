import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

BACKGROUND_IMAGE_URL = (
    "https://images.unsplash.com/photo-1611325058416-db7794e8e32c"
    "?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0"
    "&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
)

THEME_COLORS = {
    "Moon": ("#0e1117", "#ffffff"),
    "Azure": ("#80b6d8", "#408CAB"),
    "Tree": ("#8cd8a6", "#26A42F"),
    "Sunset": ("#f1a05e", "#D45A09"),
    "Dark": ("#ffffff", "#000000"),
}

SYSTEM_PROMPT = """You are an elite Quantitative Financial Analyst and Risk Assessment AI.
Your primary goal is to synthesize historical corporate data with live market news.

IMPORTANT RULES FOR ANALYSIS:
1.    If user uploads two or more reports, you will have to use a separate retriever tool for each report and use them accordingly when the user asks about a specific company. Do not mix up the data between the reports. Always use the specific retriever tool for the specific report to get the data. You can also use the Search tool to find more info about the company if needed but do not use it to find the historical financial data. Always use the uploaded reports to get the historical data.
2.    Identify the stock market then if the user mentions an indian company you will need to get the ticker for that company using the ticker_search tool and use the suffix ".NS" for NSE and ".BO" for BSE,
3.    Use .BO only if the company is exclusively listed on the BSE.
4.    IF the mentions a US company you will need to get the ticker for that company using ticker_search.
5.    When searching Indian reports, prioritize 'Consolidated' figures to ensure you are seeing the entire group's health, not just one subsidiary.
6.    If the market is India, values are in INR (Crores/Lakhs). If the market is US, values are in USD (Billions/Millions). Always convert to a single base currency before performing a comparative analysis
7.    If a company is listed on multiple global exchanges, prioritize the exchange that matches the currency of the uploaded financial report
8.    Use the get_stock_metrics tool to get the metrics to get an analysis on the stock of that company
9.    TERMINOLOGY MATCHING: When searching Indian financial reports, do not search for the US term "Net Income". You must search for "Profit After Tax" or "PAT". For revenues, search for "Total Income" or "Turnover".
10.   DO NOT GUESS ABOUT THE COMPANIES ALWAYS CHECK THE UPLOADED PDFS FOR THE DETAILS OF THE COMPANY. You can use the first page of the report to identify the details about the company.
11.   IF THE USER ASKS TO COMPARE TWO COMPANIES, FIRST CHECK IF THE USER HAS UPLOADED REPORTS FOR BOTH COMPANIES. IF YES THEN USE THE REPORTS TO GET THE DATA FOR BOTH COMPANIES AND USE RECENT NEWS ABOUT THEIR RISK FACTORS AND THEN COMPARE TO GIVE A VERDICT. IF NOT THEN USE THE TOOLS TO GET THE DATA FOR THE MISSING COMPANY AND THEN DO THE SAME.

CRITICAL RULES:
1. ZERO HALLUCINATION: Only use factual data. If you cannot find the answer, state: "Data not found."
2. TONE: Be concise, highly analytical, and completely objective. Use bullet points.
3. CITATIONS: Always mention where you got the information. Use simple names to address the sources and DO NOT STATE THE TOOLS AS THE SOURCE
4. STRICT TOOL CALLING: When you decide to use a tool, you must output the tool call immediately. Do not provide any introductory text, conversational preamble, or explanations before calling the tool"""


def get_llm():
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY is not set. Add it to your .env file (local) or Secrets (Streamlit Cloud).")
        st.stop()
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        groq_api_key=GROQ_API_KEY,
        reasoning_effort="low",
    )


def build_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


def render_sidebar():
    theme = st.sidebar.selectbox("🎨 Select Theme", list(THEME_COLORS.keys()))

    with st.sidebar:
        st.markdown("""
            **Finsight** is an edge-native, Agentic RAG quantitative terminal designed to autonomously analyze complex SEC filings and live market conditions.

            **System Architecture:**
            * 🧠 **AI Engine:** Powered by GPT-OSS 120B via Groq's LPU inference for near-instant responses.
            * 🔍 **Hybrid Vector Search:** Combines **ChromaDB** (semantic meaning) and **BM25** (exact keyword matching) to flawlessly extract dense financial tables without hallucinations.
            * 🔀 **Isolated Routing:** Filters a shared vector store by company metadata for every uploaded PDF, keeping each report's data cleanly separated without duplicating infrastructure.
            * 📈 **Live Synthesis:** Cross-references historical 10-K data with real-time stock metrics (Financial Modeling Prep) and breaking market news (DuckDuckGo).
            """)

    bg_color, text_color = THEME_COLORS[theme]
    st.markdown(
        f"""
        <style>
        section[data-testid="stSidebar"], .stApp {{
            background-image: url("{BACKGROUND_IMAGE_URL}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        body, .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
