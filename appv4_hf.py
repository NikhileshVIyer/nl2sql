import streamlit as st
import pandas as pd
import psycopg2
import sqlparse
import requests

# ----------------- Hugging Face Setup -----------------
HF_API_KEY = st.secrets["hf_api_key"]
HF_MODEL_ENDPOINT = "https://huggingface.co/defog/llama-3-sqlcoder-8b"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}",
    "Content-Type": "application/json"
}

# ----------------- Prompt Template -----------------
def build_prompt(schema: str, question: str) -> str:
    return f"""You are an AI assistant for legal risk and compliance analysts.
Based on the following database schema:

{schema}

Translate the following natural language request into an accurate SQL query:

"{question}"

Only return the SQL. Do not explain anything.
"""

# ----------------- Generate SQL Query -----------------
@st.cache_data(show_spinner=False, ttl=600)
def generate_sql_query(prompt: str) -> dict:
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.0
        }
    }
    try:
        response = requests.post(HF_MODEL_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and "generated_text" in result[0]:
            return {"content": result[0]["generated_text"]}
        else:
            return None
    except Exception as e:
        st.error(f"Error generating SQL query: {e}")
        return None

# ----------------- Execute SQL -----------------
def execute_sql_query(query: str) -> pd.DataFrame:
    try:
        connection = psycopg2.connect(
            host=st.secrets["pg_host"],
            port=st.secrets["pg_port"],
            database=st.secrets["pg_db"],
            user=st.secrets["pg_user"],
            password=st.secrets["pg_password"]
        )
        cursor = connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        cursor.close()
        connection.close()
        return pd.DataFrame(rows, columns=colnames)
    except Exception as e:
        st.error(f"Error executing SQL query: {e}")
        return pd.DataFrame()

# ----------------- UI -----------------
st.title("📊 LRC: Natural Language to SQL (Hugging Face Edition)")

st.sidebar.subheader("🧠 Schema Input")
schema_input = st.sidebar.text_area("Paste your DB schema here (text or SQL format)", height=250)

st.subheader("💬 Ask your question")
user_question = st.text_input("Ask in plain English (e.g., 'Show all high-risk contracts for India')")

if st.button("🔍 Generate SQL"):
    if not schema_input or not user_question:
        st.warning("Please provide both schema and question.")
    else:
        with st.spinner("Generating SQL from Hugging Face..."):
            full_prompt = build_prompt(schema_input, user_question)
            completion = generate_sql_query(full_prompt)

            if completion and "content" in completion:
                raw_sql = completion["content"]
                formatted_sql = sqlparse.format(raw_sql, reindent=True)
                st.code(formatted_sql, language="sql")

                if st.button("▶️ Execute SQL"):
                    with st.spinner("Running query..."):
                        df = execute_sql_query(raw_sql)
                        if not df.empty:
                            st.dataframe(df)
                        else:
                            st.warning("No results or empty result set.")
            else:
                st.error("Failed to generate query.")
