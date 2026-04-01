from langchain_ollama import ChatOllama 
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_classic.agents import AgentExecutor,create_tool_calling_agent
import tempfile
from langchain_classic.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from langchain_core.tools import Tool
import requests
import json
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background-image: url("https://images.unsplash.com/photo-1611325058416-db7794e8e32c?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");

        background-size: cover;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1611325058416-db7794e8e32c?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)


Theme = st.sidebar.selectbox("🎨 Select Theme", ["Dark", "Azure", "Tree", "Sunset","Moon"])

with st.sidebar:
    st.markdown("""
            **FInsight** is an edge-native, Agentic RAG quantitative terminal designed to autonomously analyze complex SEC filings and live market conditions.
            
            **System Architecture:**
            * 🧠 **Local AI Engine:** Powered by Qwen 3.5 running entirely locally via Ollama, ensuring absolute data privacy.
            * 🔍 **Hybrid Vector Search:** Combines **ChromaDB** (semantic meaning) and **BM25** (exact keyword matching) to flawlessly extract dense financial tables without hallucinations.
            * 🔀 **Isolated Routing:** Dynamically spins up dedicated, siloed vector databases for every uploaded PDF to completely eliminate data cross-pollination.
            * 📈 **Live Synthesis:** Cross-references historical 10-K data with real-time stock metrics (Financial Modeling Prep) and breaking market news (DuckDuckGo).
            """)

if Theme == "Moon":
    bg_color = "#0e1117"
    text_color = "#ffffff"
elif Theme == "Azure":
    bg_color = "#80b6d8"
    text_color = "#408CAB"
elif Theme == "Tree":
    bg_color = "#8cd8a6"
    text_color = "#26A42F"
elif Theme=="Sunset":
   bg_color = "#f1a05e"
   text_color = "#D45A09"
elif Theme=="Dark":
    bg_color ="#ffffff"
    text_color= "#000000"


st.markdown(
    f"""
    <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
    </style>
    """,
    unsafe_allow_html=True
)


def make_retriever_tool(retriever, company_name, safe_name):
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
        )
    )


def extract_company_name(docs, filename):
    try:
        # Take first 500 characters of the first page
        first_page = docs[0].page_content[:500]
        
        response = llm.invoke(
            f"Extract only the company name from this text. "
            f"Reply with just the company name, nothing else, no punctuation:\n\n{first_page}"
        )
        name = response.content.strip()
        # Fallback: if LLM returns something weird, use the filename
        if len(name) > 30 or len(name) < 2:
            raise ValueError("Bad extraction")
        return name
    except:
        # Strip extension and common suffixes from filename as fallback
        name = filename.rsplit(".", 1)[0]
        for suffix in ["_10K", "_10k", "_annual", "_report", "_2024", "_2023"]:
            name = name.replace(suffix, "")
        return name.strip("_")
@tool
def ticker_search(company_name  : str) -> str:
    """
     Search for ticker for the the given company name using live web search
        
        Args : 
            company_name: name of the company we need the ticker for

    """
    search_query = f"{company_name} stock ticker"
    search_results = DuckDuckGoSearchRun().run(search_query)
    return search_results
     

@tool
def get_stock_metrics(ticker : str) -> str:
    """
    Get LIVE market metrics for the company with this tool.
    CRITICAL RULE: DO NOT use this tool to find Revenue, Net Income, or historical data.
    You MUST use the 'search_financial_report' tool to read the PDFs for those numbers. 
    Args:
        ticker: stock ticker symbol 
    """
    url = f"https://financialmodelingprep.com/stable/quote?symbol={ticker}&apikey={FMP_API_KEY}"
    try:
        
        response = requests.get(url)
        data_list = response.json()
        if not data_list or len(data_list) == 0:
            return  f"Could not find data for ticker '{ticker}'. Please use the ticker_search tool to verify the symbol and try again."
        data = data_list[0]
        metrics  = {
            "symbol": data.get("symbol"),
            "company_name": data.get("name"),
            "live_price": data.get("price"),
            "pe_ratio": data.get("pe"),
            "market_cap": data.get("marketCap"),
            "earnings_per_share": data.get("eps")
        }
        return json.dumps(metrics)
    except Exception as e:
        return f"API Error fetching data for {ticker}: {str(e)}"

load_dotenv()   


llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0
)

FMP_API_KEY = os.getenv("FMP_API_KEY")

st.title("Finsight")




prompt = ChatPromptTemplate.from_messages([   
    ("system", """You are an elite Quantitative Financial Analyst and Risk Assessment AI. 
    Your primary goal is to synthesize historical corporate data with live market news.
    
    IMPORTANT RULES FOR ANALYSIS:
    1.    If user uploads two or more reports , you will have to use a separate retriever tool for each report and use them accordingly when the user asks about a specific company. Do not mix up the data between the reports. Always use the specific retriever tool for the specific report to get the data. You can also use the Search tool to find more info about the company if needed but do not use it to find the historical financial data. Always use the uploaded reports to get the historical data.
    2.    Identify the stock market then if the user mentions an indian company you will need to get the ticker for that company using the ticker_search tool and use the suffix ".NS" for NSE and ".BO" for BSE,
    3.    Use .BO only if the company is exclusively listed on the BSE. 
    4.    IF the mentions a US company you will need to get the ticker for that company using ticker_search.
    5.    When searching Indian reports, prioritize 'Consolidated' figures to ensure you are seeing the entire group's health, not just one subsidiary.
    6.    If the market is India, values are in INR (Crores/Lakhs). If the market is US, values are in USD (Billions/Millions). Always convert to a single base currency before performing a comparative analysis 
    7.    If a company is listed on multiple global exchanges, prioritize the exchange that matches the currency of the uploaded financial report
    8.    Use the get_stock_metrics tool to get the metrics to get an analysis on the stock of that company 
    9.    TERMINOLOGY MATCHING: When searching Indian financial reports, do not search for the US term "Net Income". You must search for "Profit After Tax" or "PAT". For revenues, search for "Total Income" or "Turnover".
    10.    DO NOT GUESS ABOUT THE COMAPNIES ALWAYS CHECK THE UPLOADED PDFS FOR THE DETAILS OF THE COMPANY.You can use the first page of the report to identify the details about the company.
    11.   IF THE USER ASKS TO COMPARE TWO COMPANIES, FIRST CHECK IF THE USER HAS UPLOADED REPORTS FOR BOTH COMPANIES. IF YES THEN USE THE REPORTS TO GET THE DATA FOR BOTH COMPANIES AND USE RECENT NEWS ABOUT THEIR RISK FACTORS AND THEN COMPARE TO GIVE A VERDICT. IF NOT THEN USE THE TOOLS TO GET THE DATA FOR THE MISSING COMPANY AND THEN DO THE SAME.
    CRITICAL RULES:
    1. ZERO HALLUCINATION: Only use factual data. If you cannot find the answer, state: "Data not found."
    2. TONE: Be concise, highly analytical, and completely objective. Use bullet points. 
    3. CITATIONS: Always mention where you got the information. Use simple names to address the sources and DO NOT STATE THE TOOLS AS THE SOURCE
    4. STRICT TOOL CALLING: When you decide to use a tool, you must output the tool call immediately. Do not provide any introductory text, conversational preamble, or explanations before calling the tool"""),
    

    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 

user_input = st.text_input("Ask a question about the financial report:")

uploaded_file = st.file_uploader("Upload a financial report", type=["pdf"],accept_multiple_files=True)


@st.cache_resource
def create_retrievers(uploaded_file):
    tools = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    for uploaded in uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded.getvalue())
                fp = tmp_file.name
        loader = UnstructuredPDFLoader(fp)
        docs = loader.load()
        
        documents = text_splitter.split_documents(docs)
        for chunk in documents:
            chunk.page_content = f"【Source Report: {uploaded.name}】\n\n" + chunk.page_content

        os.remove(fp)
    
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents, embedding= embeddings)
        chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = 2

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever], 
            weights=[0.5, 0.5] # Give them equal power
        )
        safe_name = "".join(c if c.isalnum() else "_" for c in uploaded.name)
        company_name = extract_company_name(docs, uploaded.name)
        tools.append(
            make_retriever_tool(ensemble_retriever, company_name, safe_name)
        )

    return tools





if uploaded_file:
    
    
        
    report_reader = create_retrievers(uploaded_file)
        




    searcher = DuckDuckGoSearchRun(name="market_searcher", description="""USE THIS TOOL to search the live internet for current events, breaking news, 
        and recent market sentiment. 
        Crucial: When checking 'Risk Factors' found in a 10-K report, use this tool to search if 
        those specific risks (e.g., 'supply chain disruption', 'lawsuits') are currently happening in the news today."""
    )


    tools = report_reader + [ searcher , ticker_search, get_stock_metrics]

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt = prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)




    if "vector" not in st.session_state and user_input:
        with st.spinner("Analyzing..."):
            response = agent_executor.invoke({"input":user_input,"chat_history": st.session_state.chat_history,})
            #clean_text = re.sub(r'【.*?】', '', response['output']) 
            #st.markdown(clean_text)
            st.markdown(response['output'])

