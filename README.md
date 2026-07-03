# Finsight

Finsight is an agentic RAG terminal for financial reports. Upload one or more 10-K/10-Q PDFs and ask questions — the agent reads the filings, cross-checks live stock data and news, and answers with citations back to the source report.

It's built to avoid the two classic RAG failure modes: mixing up numbers between different companies' filings, and hallucinating figures that aren't actually in the document.

## How it works

- **Ingestion** — each uploaded PDF is parsed with `UnstructuredPDFLoader`, chunked, and embedded with `all-MiniLM-L6-v2`.
- **Isolated retrieval** — every document gets its own Chroma collection and its own retriever tool, so the agent can't accidentally blend Company A's numbers into an answer about Company B.
- **Hybrid search** — each retriever combines BM25 (keyword) and Chroma (semantic) search via `EnsembleRetriever`, which matters a lot for financial tables where exact terms like "Profit After Tax" need to match precisely.
- **Agent loop** — a LangChain tool-calling agent decides when to pull from a report, search the web for breaking news, look up a ticker, or fetch live quote data, then synthesizes an answer.
- **LLM** — runs on Groq (`llama-3.3-70b-versatile`) for fast inference.

## Project layout

```
app.py         # Streamlit UI, wires everything together
config.py      # env vars, LLM setup, theme/CSS, system prompt
tools.py       # ticker lookup, live stock metrics, market news search
ingestion.py   # PDF parsing + hybrid retriever construction
```

## Setup

**Requirements:** Python 3.10+, a free [Groq API key](https://console.groq.com/keys), and a free [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs) key.

```bash
git clone https://github.com/Pankaj-jpeg/Finsight.git
cd Finsight
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your keys:

```
GROQ_API_KEY=your_key_here
FMP_API_KEY=your_key_here
```

Run it:

```bash
streamlit run app.py
```

Open `http://localhost:8501`, upload a PDF report, and start asking questions.

## Deploying

Runs cleanly on [Streamlit Community Cloud](https://share.streamlit.io) — no GPU or local model needed now that inference is via Groq.

1. Push this repo to GitHub (already done).
2. On Streamlit Cloud: New app → point it at this repo → main file `app.py`.
3. Under Advanced settings → Secrets, add `GROQ_API_KEY` and `FMP_API_KEY`.
4. Deploy. `packages.txt` handles the system-level PDF dependencies (`poppler-utils`, `tesseract-ocr`, `libmagic1`) automatically.

## Notes / limitations

- Financial data accuracy depends entirely on what's in the uploaded PDF and what the retriever surfaces — always verify anything used for an actual investment decision.
- `sentence-transformers` + `chromadb` are the heaviest dependencies; on memory-constrained hosting tiers, PDF ingestion is the most likely thing to fail first.
- This is a portfolio/educational project, not financial advice.
