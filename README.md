# Finsight

Finsight is an agentic RAG terminal for financial reports. Upload one or more 10-K/10-Q PDFs and ask questions — the agent reads the filings, cross-checks live stock data and news, and answers with citations back to the source report. It can also export headline financial figures straight to an Excel file.

It's built to avoid the two classic RAG failure modes: mixing up numbers between different companies' filings, and hallucinating figures that aren't actually in the document.

## How it works

- **Ingestion** — each uploaded PDF is parsed with `UnstructuredPDFLoader` (fast, text-based strategy — no heavy vision/layout models), chunked, and embedded with `all-MiniLM-L6-v2`.
- **Shared vectorstore, metadata-filtered isolation** — every chunk from every uploaded report lives in one Chroma collection, tagged with a `company` field. Isolation between companies' data comes from filtering on that metadata at query time, not from spinning up a separate database per file — a single embedding model and a single vectorstore serve every upload, which keeps memory usage in check.
- **Hybrid search** — each company's retriever tool combines BM25 (keyword) and the filtered Chroma search (semantic), which matters a lot for financial tables where exact terms like "Profit After Tax" need to match precisely.
- **Agent loop** — a LangChain tool-calling agent decides when to pull from a report, search the web for breaking news, look up a ticker, or fetch live quote data, then synthesizes an answer. It's scoped to financial topics only and declines unrelated requests.
- **LLM** — runs on Groq (`openai/gpt-oss-120b`, low reasoning effort) for fast, low-latency inference.
- **Excel export** — a dedicated extraction step (independent of chat) pulls headline figures (revenue, net income, gross margin, operating income, EPS, assets, liabilities) per company into a structured `.xlsx` file, with values kept exactly as reported — no unit or currency conversion, no guessing. Missing fields are marked `N/A` rather than filled in.

## Project layout

```
app.py         # Streamlit UI, wires everything together
config.py      # env vars, LLM setup, theme/CSS, system prompt
tools.py       # ticker lookup, live stock metrics, market news search
ingestion.py   # PDF parsing + shared vectorstore + retriever tool construction
export.py      # structured financial data extraction + Excel file generation
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

## Using the Excel export

Once at least one report is uploaded, an **"📊 Export Financials to Excel"** button appears. Clicking it runs a dedicated retrieval + extraction pass per company (separate from whatever's been asked in chat) and produces a downloadable spreadsheet with one row per company:

| Company | Fiscal Year | Total Revenue | Net Income / PAT | Gross Margin | Operating Income | EPS | Total Assets | Total Liabilities | Source Report |
|---|---|---|---|---|---|---|---|---|---|

Values are reported exactly as they appear in the filing (including currency symbol and unit), and any figure the model can't find is marked `N/A` rather than estimated.

## Deploying

Runs cleanly on [Streamlit Community Cloud](https://share.streamlit.io) — no GPU or local model needed, inference is via Groq.

1. Push this repo to GitHub (already done).
2. On Streamlit Cloud: New app → point it at this repo → main file `app.py`.
3. Under Advanced settings → Secrets, add `GROQ_API_KEY` and `FMP_API_KEY`.
4. Deploy. `packages.txt` handles the system-level PDF dependencies (`poppler-utils`, `tesseract-ocr`, `libgl1`) automatically, and `requirements.txt` pre-installs the spaCy model needed by `unstructured` at build time (the container's site-packages is read-only at runtime, so it can't be downloaded lazily).

## Notes / limitations

- Financial data accuracy depends entirely on what's in the uploaded PDF and what the retriever surfaces — always verify anything used for an actual investment decision. In testing, headline figures (revenue, net income, EPS) have checked out accurately against real filings, but retrieval can occasionally grab the wrong line item from a densely tabular statement (e.g. confusing adjacent fiscal years, or cost of revenue vs. gross profit) — cross-check anything that matters.
- `sentence-transformers` + `chromadb` are the heaviest dependencies; on memory-constrained hosting tiers, PDF ingestion is the most likely thing to fail first.
- The agent is scoped to financial topics and will decline unrelated requests (e.g. "write me code unrelated to this project").
- This is a portfolio/educational project, not financial advice.
