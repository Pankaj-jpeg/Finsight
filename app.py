import streamlit as st
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage

from config import get_llm, build_prompt, render_sidebar
from ingestion import create_retrievers
from tools import ticker_search, get_stock_metrics, get_market_searcher

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
    report_tools = create_retrievers(llm, uploaded_files)
    tools = report_tools + [get_market_searcher(), ticker_search, get_stock_metrics]

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = agent_executor.invoke(
                    {"input": user_input, "chat_history": st.session_state.chat_history}
                )
                st.markdown(response["output"])

        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=response["output"]))
elif user_input:
    st.info("Upload at least one financial report (PDF) before asking a question.")
