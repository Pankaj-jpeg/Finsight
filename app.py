import streamlit as st
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage

from config import get_llm, build_prompt, render_sidebar
from ingestion import create_retrievers
from tools import ticker_search, get_stock_metrics, get_market_searcher
from export import extract_financial_summary, build_excel_bytes

st.set_page_config(page_title="Finsight", page_icon="📊")
render_sidebar()

st.title("Finsight")

llm = get_llm()
prompt = build_prompt()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader("Upload a financial report", type=["pdf"], accept_multiple_files=True)

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

user_input = st.chat_input("Ask a question about the financial report:")

if uploaded_files:
    with st.spinner("Reading and indexing uploaded reports..."):
        report_tools, vectorstore, company_meta = create_retrievers(llm, uploaded_files)
    tools = report_tools + [get_market_searcher(), ticker_search, get_stock_metrics]

    if vectorstore is not None:
        if st.button("📊 Export Financials to Excel"):
            with st.spinner("Extracting figures from each report..."):
                summaries = []
                for company_name, _, filename in company_meta:
                    summary = extract_financial_summary(llm, vectorstore, company_name)
                    summary["company"] = company_name
                    summary["source_report"] = filename
                    summaries.append(summary)
                excel_bytes = build_excel_bytes(summaries)

            st.download_button(
                "Download financial_summary.xlsx",
                data=excel_bytes,
                file_name="financial_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    @st.cache_resource(show_spinner=False)
    def get_agent_executor(_llm, _tools, _prompt):
        agent = create_tool_calling_agent(llm=_llm, tools=_tools, prompt=_prompt)
        return AgentExecutor(
            agent=agent, tools=_tools, verbose=True, handle_parsing_errors=True, max_iterations=6
        )

    agent_executor = get_agent_executor(llm, tools, prompt)

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    response = agent_executor.invoke(
                        {"input": user_input, "chat_history": st.session_state.chat_history}
                    )
                    output = response["output"]
                    if not isinstance(output, str):
                        # Defensive: some providers can return content as a list of blocks
                        # instead of a plain string; flatten it rather than crash or show raw repr.
                        output = "\n\n".join(
                            block.get("text") or block.get("thinking") or str(block)
                            for block in output
                        ) if isinstance(output, list) else str(output)
                    st.markdown(output)
                except Exception as e:
                    msg = str(e)
                    if "rate_limit" in msg.lower() or "429" in msg:
                        output = None
                        st.error(
                            "Hit Groq's rate limit for this model. Wait a bit and try again, "
                            "or switch to a lighter model / upgrade to Groq's Developer tier for higher limits."
                        )
                    else:
                        output = None
                        st.error(f"Something went wrong: {msg}")

        if output is not None:
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            st.session_state.chat_history.append(AIMessage(content=output))
elif user_input:
    st.info("Upload at least one financial report (PDF) before asking a question.")
