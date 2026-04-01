# ⚡ FInsight: Agentic RAG Quantitative Terminal

FInsight is an edge-native, locally hosted financial AI agent. It seamlessly merges historical corporate data (extracted from uploaded SEC filings) with live market metrics and breaking news, operating entirely on local LLMs to guarantee absolute data privacy.

## 🌟 Key Engineering Features

Unlike standard "ChatGPT wrappers," FInsight utilizes an advanced, multi-agent architecture designed for strict financial accuracy:

* **100% Local & Private AI:** Powered by Qwen 2.5 (via Ollama), ensuring sensitive financial documents never leave your machine.
* **Dynamic Document Routing:** Solves the classic RAG "cross-pollination" bug by dynamically spinning up mathematically isolated vector databases for every uploaded PDF. The AI Agent routes its queries to the exact document needed, eliminating data bleeding between companies.
* **Hybrid Vector Search:** Combines **ChromaDB** (semantic similarity) and **BM25** (exact keyword matching via EnsembleRetriever) to flawlessly extract dense, highly-specific financial tables (e.g., "Consolidated Statements of Operations") without hallucination.
* **Live Market Synthesis:** Cross-references historical 10-K data with real-time stock metrics (Financial Modeling Prep API) and breaking market news (DuckDuckGo Search) to provide holistic investment verdicts.
* **Interactive UI:** Built with Streamlit, featuring a dual-tab layout for an AI Command Center and an interactive Plotly Candlestick Market Dashboard.

## 🏗️ System Architecture

1.  **Ingestion:** PDFs are parsed using `UnstructuredPDFLoader`.
2.  **Processing:** Text is chunked and embedded using HuggingFace `all-MiniLM-L6-v2`.
3.  **Storage:** Dedicated Chroma collections are generated per document.
4.  **Retrieval:** Hybrid Search (50% Semantic / 50% Keyword) surfaces relevant chunks.
5.  **Execution:** LangChain's `AgentExecutor` manages tool-calling (Retrievers, Web Search, FMP API) and synthesizes the final output.

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) installed on your machine.
* A free API key from [Financial Modeling Prep (FMP)](https://financialmodelingprep.com/).

### Installation

1. **Clone the repository:**
   `git clone https://github.com/yourusername/FInsight.git`
   `cd FInsight`

2. **Create and activate a virtual environment:**
   Windows: `python -m venv venv` and `.\venv\Scripts\activate`
   Mac/Linux: `python3 -m venv venv` and `source venv/bin/activate`

3. **Install dependencies:**
   `pip install -r requirements.txt`

4. **Environment Variables:**
   Create a `.env` file in the root directory and add your FMP API key:
   `FMP_API_KEY=your_api_key_here`

5. **Pull the Local LLM:**
   Ensure Ollama is running, then pull the required model:
   `ollama pull qwen2.5:7b`

## 💻 Usage

Run the Streamlit application:
`streamlit run app.py`

1. Open your browser to `http://localhost:8501`.
2. Upload one or more financial PDFs (e.g., 10-K, 10-Q reports) in the sidebar.
3. Use the **Agent Chat** tab to ask comparative quantitative questions.
4. Use the **Market Dashboard** tab to view live, interactive price action.

## ⚠️ Disclaimer
This project is for educational and portfolio purposes only. AI-generated financial analysis can contain errors. Always conduct your own due diligence before making investment decisions.
