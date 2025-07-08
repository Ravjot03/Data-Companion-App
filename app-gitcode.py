import os, io, re
import pandas as pd
import streamlit as st
from openai import OpenAI
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# === Configuration ===
api_key = "NVIDIA-API"
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)

# === Utility Functions ===
def extract_first_code_block(text: str) -> str:
    start = text.find("```python")
    if start == -1:
        return ""
    start += len("```python")
    end = text.find("```", start)
    if end == -1:
        return ""
    return text[start:end].strip()

# === LLM Tools ===
def QueryUnderstandingTool(query: str) -> bool:
    messages = [
        {"role": "system", "content": "Respond only with 'true' if this query asks for a chart, graph, or plot. Otherwise, respond 'false'."},
        {"role": "user", "content": query}
    ]
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=messages,
        temperature=0.1,
        max_tokens=5
    )
    return response.choices[0].message.content.strip().lower() == "true"

def CodeWritingTool(cols: List[str], query: str, plot: bool = False) -> str:
    base = f"Given DataFrame `df` with columns: {', '.join(cols)}\n"
    task = f"Write Python code to answer: \"{query}\"\n"
    if plot:
        rules = ("Use pandas for manipulation and matplotlib for plotting. "
                 "Assign final result to `result`. Only one plot. Wrap in ```python fence.")
    else:
        rules = ("Use pandas only. Assign result to `result`. Wrap in ```python fence.")
    return base + task + rules

def CodeGenerationAgent(query: str, df: pd.DataFrame):
    should_plot = QueryUnderstandingTool(query)
    prompt = CodeWritingTool(df.columns.tolist(), query, should_plot)
    messages = [
        {"role": "system", "content": "Write clean pandas (and matplotlib if needed) code. Return only code inside ```python block."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=messages,
        temperature=0.2,
        max_tokens=1024
    )
    code = extract_first_code_block(response.choices[0].message.content)
    return code, should_plot

def ExecutionAgent(code: str, df: pd.DataFrame, should_plot: bool):
    env = {"pd": pd, "df": df, "plt": plt, "io": io} if should_plot else {"pd": pd, "df": df}
    try:
        exec(code, {}, env)
        return env.get("result", None)
    except Exception as e:
        return f"Error: {e}"

def ReasoningAgent(query: str, result: Any) -> str:
    desc = str(result)[:300] if not isinstance(result, (plt.Figure, plt.Axes)) else "[Plot Output]"
    prompt = f"The user asked: {query}\nResult: {desc}\nExplain in 2-3 sentences."
    messages = [
        {"role": "system", "content": "Provide concise analytical explanations."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=messages,
        temperature=0.2,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

def DataInsightAgent(df: pd.DataFrame) -> str:
    prompt = f"Columns: {', '.join(df.columns)}\nTypes: {df.dtypes.to_dict()}\nNulls: {df.isnull().sum().to_dict()}"
    messages = [
        {"role": "system", "content": "Describe this dataset and suggest 3-4 analysis questions."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        messages=messages,
        temperature=0.2,
        max_tokens=512
    )
    return response.choices[0].message.content.strip()

# === Streamlit App ===
def main():
    st.set_page_config(layout="wide")

    st.markdown("""
    <h1 style='text-align:center;'>📊 Data Companion</h1>
    <p style='text-align:center;'>An AI-powered assistant for smart, explainable data analysis</p>
    <hr>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📂 Upload & Explore", "💬 Chat with Your Data"])

    with tab1:
        file = st.file_uploader("Upload a CSV file", type=["csv"])
        if file:
            df = pd.read_csv(file)
            st.session_state.df = df
            st.success(f"Loaded `{file.name}` ({df.shape[0]} rows, {df.shape[1]} cols)")
            st.dataframe(df.head())
            with st.expander("📌 Dataset Insights", expanded=True):
                summary = DataInsightAgent(df)
                st.markdown(summary)

    with tab2:
        if "df" not in st.session_state:
            st.info("Upload a dataset to start chatting.")
            return

        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.plots = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"], unsafe_allow_html=True)
                if msg.get("plot_index") is not None:
                    st.pyplot(st.session_state.plots[msg["plot_index"]], use_container_width=True)

        user_q = st.chat_input("Ask your data a question…")
        if user_q:
            st.session_state.messages.append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.spinner("Thinking…"):
                code, should_plot = CodeGenerationAgent(user_q, st.session_state.df)
                result = ExecutionAgent(code, st.session_state.df, should_plot)
                explanation = ReasoningAgent(user_q, result)

            plot_idx = None
            if isinstance(result, (plt.Figure, plt.Axes)):
                fig = result.figure if isinstance(result, plt.Axes) else result
                st.session_state.plots.append(fig)
                plot_idx = len(st.session_state.plots) - 1

            assistant_msg = f"{explanation}\n\n<details><summary>🔍 View Code</summary><pre><code>{code}</code></pre></details>"
            st.session_state.messages.append({"role": "assistant", "content": assistant_msg, "plot_index": plot_idx})
            st.rerun()

if __name__ == "__main__":
    main()
