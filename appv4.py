import streamlit as st
import psycopg2
import numpy as np
import sqlparse
import pandas as pd
import plotly.express as px
import re
import uuid
from openai import OpenAI
from llm_confidence.logprobs_handler import LogprobsHandler


# --------------------------------

OPENAI_API_KEY = st.secrets["openai_api_key"]
BASE_URL = st.secrets["base_url"]
MODEL = st.secrets["model"]

PG_HOST = st.secrets["pg_host"]
PG_PORT = st.secrets["pg_port"]
PG_DB = st.secrets["pg_db"]
PG_USER = st.secrets["pg_user"]
PG_PASSWORD = st.secrets["pg_password"]


client = OpenAI(
    base_url=BASE_URL,
    api_key=OPENAI_API_KEY,
)

#---------------
# Streamlit App UI Configuration and Session State
# -----------------------------------------------
st.set_page_config(page_title="NL to SQL Generator", layout="wide")
st.title("📝 Natural Language to SQL Generator")
st.markdown("Convert your natural language questions into SQL queries effortlessly!")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df_results" not in st.session_state:
    st.session_state.df_results = None
if "generated_sql" not in st.session_state:
    st.session_state.generated_sql = ""
if "perplexity" not in st.session_state:
    st.session_state.perplexity = None
if "current_visualization" not in st.session_state:
    st.session_state.current_visualization = None
if "db_connection_status" not in st.session_state:
    st.session_state.db_connection_status = None
# New session state to track query execution status
if "query_execution_status" not in st.session_state:
    st.session_state.query_execution_status = None  # Options: None, "success", "no_data", "error"
if "query_error_message" not in st.session_state:
    st.session_state.query_error_message = ""

# -----------------------------------------------
# Sidebar: Database Schema Editor and Controls
# -----------------------------------------------
st.sidebar.header("Database Schema Editor")
schema_input = st.sidebar.text_area(
    "Enter your database schema:", 
    placeholder="Enter your database schema here...", 
    height=400
)

# Database connection test
if st.sidebar.button("Test Database Connection"):
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD,
            connect_timeout=5
        )
        conn.close()
        st.session_state.db_connection_status = "success"
        st.sidebar.success("✅ Database connection successful!")
    except Exception as e:
        st.session_state.db_connection_status = "error"
        st.sidebar.error(f"❌ Database connection failed: {e}")

# Clear results button
if st.sidebar.button("Clear Results"):
    st.session_state.df_results = None
    st.session_state.generated_sql = ""
    st.session_state.perplexity = None
    st.session_state.current_visualization = None
    st.session_state.query_execution_status = None
    st.session_state.query_error_message = ""
    st.sidebar.success("Results cleared!")

# Reset app button
if st.sidebar.button("Reset App"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.chat_history = []
    st.session_state.df_results = None
    st.session_state.generated_sql = ""
    st.session_state.perplexity = None
    st.session_state.current_visualization = None
    st.session_state.query_execution_status = None
    st.session_state.query_error_message = ""
    st.sidebar.success("App has been reset!")

# -----------------------------------------------
# Helper Functions
# -----------------------------------------------
def extract_sql_query(response_text):
    """
    Extract SQL query from model response, handling different formats.
    """
    # Try to extract SQL between ```sql and ``` tags
    sql_pattern = r"```sql\n(.*?)```"
    matches = re.search(sql_pattern, response_text, re.DOTALL)
    
    if matches:
        return matches.group(1).strip()
    
    # If no match with SQL tag, try without specifying language
    general_pattern = r"```\n?(.*?)```"
    matches = re.search(general_pattern, response_text, re.DOTALL)
    
    if matches:
        return matches.group(1).strip()
    
    # If still no match, return the whole text as it might be just SQL without code blocks
    return response_text.strip()

@st.cache_data(show_spinner=False, ttl=600)
def generate_sql_query(prompt: str) -> dict:
    """
    Generate a SQL query from the LLM API.
    Returns a dict containing the SQL query and the response logprobs.
    """
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            temperature=0.0,  # Lower temperature for more consistent SQL
            logprobs=True,
        )
        return completion
    except Exception as e:
        st.error(f"Error generating SQL query: {str(e)}")
        return None

@st.cache_data(show_spinner=False, ttl=600)
def execute_query_on_db(query: str) -> tuple:
    """
    Execute a SQL query on the PostgreSQL database.
    Returns a tuple (columns, results, status, error_message).
    """
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DB,
            user=PG_USER,
            password=PG_PASSWORD,
            connect_timeout=10
        )
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(query)
        
        if cursor.description is not None:
            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            
            if results:
                status = "success"  # Data returned
            else:
                status = "no_data"  # Query executed but no data returned
        else:
            # For non-SELECT queries (INSERT, UPDATE, DELETE)
            columns, results = None, None
            status = "no_data"  # Query executed successfully but no data to return
            
        cursor.close()
        conn.close()
        return columns, results, status, ""
    except Exception as e:
        error_message = str(e)
        return None, None, "error", error_message

def is_data_visualizable(df):
    """
    Determine if the data is suitable for visualization.
    """
    if df is None or df.empty:
        return False
    
    # Too few rows
    if len(df) <= 1:
        return False
    
    # Check if we have at least one numeric column for most visualizations
    has_numeric = any(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns)
    
    # For categorical data, ensure we have reasonable number of categories
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    has_good_categorical = any(df[col].nunique() <= 15 and df[col].nunique() > 1 for col in categorical_cols) if len(categorical_cols) > 0 else False
    
    # Too many columns might not visualize well
    too_many_cols = len(df.columns) > 30
    
    # If the result is primarily a lookup (just 1 or 2 rows)
    is_lookup_result = len(df) <= 2 and len(df.columns) >= 5
    
    return (has_numeric or has_good_categorical) and not too_many_cols and not is_lookup_result

def get_visualization(df: pd.DataFrame):
    """
    Create an interactive Plotly chart based on the DataFrame using actual data columns.
    Only generate a graph if the chosen configuration makes sense.
    """
    if df is None or df.empty:
        return None
        
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    temporal_cols = [col for col in all_cols if 'date' in col.lower() or 'time' in col.lower() 
                     or df[col].dtype.name in ['datetime64[ns]', 'timedelta[ns]']]
    
    if not all_cols:
        return None

    # Determine suitable chart types based on data structure
    chart_options = []
    
    # Add appropriate chart types based on data
    if numeric_cols and (categorical_cols or temporal_cols):
        chart_options.extend(["Bar Chart", "Line Chart"])
    
    if numeric_cols and len(numeric_cols) >= 2:
        chart_options.append("Scatter Plot")
    
    if numeric_cols:
        chart_options.append("Histogram")
        chart_options.append("Box Plot")
    
    if categorical_cols and (numeric_cols or len(df) >= 5):
        chart_options.append("Pie Chart")
    
    if not chart_options:
        return None
            
    chart_type = st.selectbox("Choose chart type:", 
                              chart_options,
                              key=f"chart_type_{id(df)}")
    
    # Logic for different chart types
    if chart_type == "Scatter Plot":
        x_col = st.selectbox("Select X-axis (numeric):", 
                            numeric_cols, index=0, key=f"x_col_scatter_{id(df)}")
        y_col = st.selectbox("Select Y-axis (numeric):", 
                            numeric_cols, index=min(1, len(numeric_cols)-1), 
                            key=f"y_col_scatter_{id(df)}")
        
        color_col = None
        if len(categorical_cols) > 0:
            use_color = st.checkbox("Group by category?", key=f"use_color_{id(df)}")
            if use_color:
                color_col = st.selectbox("Select grouping variable:", 
                                        categorical_cols, key=f"color_col_{id(df)}")
        
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                        title=f"Scatter Plot: {x_col} vs {y_col}")
        
    elif chart_type == "Bar Chart":
        has_candidates = bool(categorical_cols) and bool(numeric_cols)
        
        if not has_candidates:
            if categorical_cols:
                x_col = st.selectbox("Select X-axis:", 
                                    categorical_cols, index=0, key=f"x_col_bar_{id(df)}")
                # Count frequency
                count_df = df[x_col].value_counts().reset_index()
                count_df.columns = [x_col, 'count']
                fig = px.bar(count_df, x=x_col, y='count', 
                            title=f"Bar Chart: Frequency of {x_col}")
            else:
                return None
        else:
            x_col = st.selectbox("Select X-axis (category):", 
                                categorical_cols, index=0, key=f"x_col_bar_{id(df)}")
            y_col = st.selectbox("Select Y-axis (numeric):", 
                                numeric_cols, index=0, key=f"y_col_bar_{id(df)}")
            fig = px.bar(df, x=x_col, y=y_col, 
                        title=f"Bar Chart: {x_col} vs {y_col}")
        
    elif chart_type == "Line Chart":
        if temporal_cols:
            x_col = st.selectbox("Select X-axis (temporal):", 
                                temporal_cols, index=0, key=f"x_col_line_{id(df)}")
        else:
            x_col = st.selectbox("Select X-axis:", 
                                all_cols, index=0, key=f"x_col_line_{id(df)}")
            
        if not numeric_cols:
            return None
            
        y_col = st.selectbox("Select Y-axis (numeric):", 
                            numeric_cols, index=0, key=f"y_col_line_{id(df)}")
        
        fig = px.line(df, x=x_col, y=y_col, 
                    title=f"Line Chart: {x_col} vs {y_col}")
        
    elif chart_type == "Pie Chart":
        if not categorical_cols:
            return None
            
        names_col = st.selectbox("Select category column:", 
                                categorical_cols, index=0, key=f"names_col_pie_{id(df)}")
        
        if numeric_cols:
            values_col = st.selectbox("Select values column (numeric):", 
                                    numeric_cols, index=0, key=f"values_col_pie_{id(df)}")
            fig = px.pie(df, names=names_col, values=values_col, 
                        title=f"Pie Chart: {names_col} by {values_col}")
        else:
            # Count frequency
            count_df = df[names_col].value_counts().reset_index()
            count_df.columns = [names_col, 'count']
            fig = px.pie(count_df, names=names_col, values='count', 
                        title=f"Pie Chart: Distribution of {names_col}")
    
    elif chart_type == "Box Plot":
        if not numeric_cols:
            return None
            
        y_col = st.selectbox("Select Y-axis (numeric):", 
                            numeric_cols, index=0, key=f"y_col_box_{id(df)}")
        
        x_col = None
        if categorical_cols:
            use_category = st.checkbox("Group by category?", key=f"use_category_box_{id(df)}")
            if use_category:
                x_col = st.selectbox("Select category column:", 
                                    categorical_cols, index=0, key=f"x_col_box_{id(df)}")
        
        fig = px.box(df, x=x_col, y=y_col, 
                    title=f"Box Plot: Distribution of {y_col}")
        
    elif chart_type == "Histogram":
        if not numeric_cols:
            return None
            
        x_col = st.selectbox("Select column for histogram:", 
                            numeric_cols, index=0, key=f"x_col_hist_{id(df)}")
        
        bins = st.slider("Number of bins:", 
                        min_value=5, max_value=50, value=20, key=f"bins_hist_{id(df)}")
        
        fig = px.histogram(df, x=x_col, nbins=bins,
                        title=f"Histogram: Distribution of {x_col}")
    
    else:
        return None

    # Common figure updates
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20), 
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# -----------------------------------------------
# Layout: Two Columns (Main Content & Chat History)
# -----------------------------------------------
left_col, right_col = st.columns([3, 1])

# Left Column: Main Content
with left_col:
    # Get previous question if it exists
    previous_question = ""
    if st.session_state.chat_history:
        previous_question = st.session_state.chat_history[-1]["question"]
    
    # Query input area
    user_question = st.text_area(
        "Enter your question:", 
        value=previous_question,
        placeholder="Enter your natural language question here...",
        key="user_question",
        height=100
    )

    # Generate SQL button with progress bar
    if st.button("Generate SQL Query", key="generate_query", use_container_width=True):
        if not schema_input:
            st.warning("⚠️ Please enter the database schema before generating the SQL query.")
        elif not user_question:
            st.warning("⚠️ Please enter a question before generating the SQL query.")
        elif st.session_state.db_connection_status == "error":
            st.warning("⚠️ Database connection is not working. Please check your connection settings.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.info("Starting SQL query generation...")
    
            instructions = (
                "- Carefully analyze the database schema and generate valid SQL query\n"
                "- If the question cannot be answered given the schema, return 'I do not know'.\n"
                "- Return ONLY the SQL query, not explanations\n"
                "- Ensure your SQL is compatible with PostgreSQL\n"
                "- Recall that the current date in YYYY-MM-DD format is 2025-01-31."
            )
            
            prompt = (
                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
                f"Generate a SQL query to answer this question: `{user_question}`\n"
                f"{instructions}\n"
                f"DDL statements:\n"
                f"{schema_input}\n"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
                f"The following SQL query best answers the question `{user_question}`:\n"
                f"```sql\n"
            )
            progress_bar.progress(20)
            status_text.info("Generating SQL query...")
    
            try:
                completion = generate_sql_query(prompt)
                if completion is None:
                    st.error("Failed to generate SQL query. Please try again.")
                    progress_bar.empty()
                    status_text.empty()
                    st.stop()
                    
                progress_bar.progress(50)
                status_text.info("Processing query results...")
        
                raw_response = completion.choices[0].message.content
                sql_query = extract_sql_query(raw_response)
                
                response_logprobs = (
                    completion.choices[0].logprobs.content
                    if hasattr(completion.choices[0], 'logprobs') and hasattr(completion.choices[0].logprobs, 'content')
                    else []
                )
        
                logprobs_handler = LogprobsHandler()
                logprobs_formatted = logprobs_handler.format_logprobs(response_logprobs)
                logprobs = [entry['logprob'] for entry in logprobs_formatted if entry['logprob'] is not None]
                perplexity_score = np.exp(-np.mean(logprobs)) if logprobs else None
                
                progress_bar.progress(70)
                status_text.info("Formatting and executing query...")
        
                formatted_sql = sqlparse.format(sql_query, reindent=True, keyword_case='upper')
        
                # Execute query with improved error handling
                columns, results, status, error_message = execute_query_on_db(sql_query)
                progress_bar.progress(90)
        
                # Store results and status in session state
                st.session_state.generated_sql = formatted_sql
                st.session_state.perplexity = perplexity_score
                st.session_state.query_execution_status = status
                st.session_state.query_error_message = error_message
                
                if columns and results:
                    df_results = pd.DataFrame(results, columns=columns)
                    st.session_state.df_results = df_results
                else:
                    st.session_state.df_results = None
        
                # Append to chat history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "sql": formatted_sql,
                    "results": st.session_state.df_results,
                    "perplexity": perplexity_score,
                    "status": status,
                    "error_message": error_message
                })
        
                progress_bar.progress(100)
                status_text.success("Operation complete!")
                
                # Force a rerun to refresh the UI with new results
                st.rerun()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"An error occurred: {str(e)}")

    # Display results if available
    if st.session_state.generated_sql:
        st.subheader("Generated SQL Query")
        
        # User can copy the SQL with a button
        col1, col2 = st.columns([6, 1])
        with col1:
            st.code(st.session_state.generated_sql, language="sql")
        with col2:
            if st.button("📋 Copy", key="copy_sql"):
                st.toast("SQL copied to clipboard!", icon="✅")
                
        # Display perplexity score if available
        if st.session_state.perplexity is not None:
            st.metric(
                label="Confidence Score", 
                value=f"{st.session_state.perplexity:.2f}"
            )
        
        # Display query execution status in main content area
        if st.session_state.query_execution_status == "error":
            st.error(f"SQL Execution Error: {st.session_state.query_error_message}")
        elif st.session_state.query_execution_status == "no_data":
            st.info("Query executed successfully, but returned no data.")
        
        # Display query results
        if st.session_state.df_results is not None and not st.session_state.df_results.empty:
            st.subheader("Query Results")
            
            # Download button for results
            csv = st.session_state.df_results.to_csv(index=False)
            download_col1, download_col2 = st.columns([6, 1])
            with download_col2:
                st.download_button(
                    label="📥 CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv",
                    key="download_results"
                )
            
            # Display dataframe
            st.dataframe(
                st.session_state.df_results, 
                use_container_width=True, 
                height=min(400, 35 + 35 * len(st.session_state.df_results))
            )
            
            # Data visualization - only show if the data is appropriate for visualization
            if is_data_visualizable(st.session_state.df_results):
                st.subheader("Data Visualization")
                st.session_state.current_visualization = get_visualization(st.session_state.df_results)
                
                if st.session_state.current_visualization is not None:
                    st.plotly_chart(st.session_state.current_visualization, use_container_width=True)
            else:
                st.session_state.current_visualization = None
                
    else:
        # Show helpful prompt when no query has been generated yet
        st.info("Enter your question and database schema, then click 'Generate SQL Query' to begin.")

# -----------------------------------------------
# Right Column: Chat History
# -----------------------------------------------
with right_col:
    st.subheader("Chat History")
    
    if not st.session_state.chat_history:
        st.info("No chat history yet. Ask your first question!")
    else:
        for idx, chat in enumerate(st.session_state.chat_history[::-1]):
            with st.expander(f"Q: {chat['question'][:50]}...", expanded=(idx == 0)):
                st.markdown(f"**Question:**\n{chat['question']}")
                st.markdown("**SQL Query:**")
                st.code(chat['sql'], language="sql")
                
                # Show perplexity if available
                if chat.get('perplexity') is not None:
                    st.caption(f"Perplexity Score: {chat['perplexity']:.2f}")
                
                # Show execution status in chat history
                status = chat.get('status')
                if status == "error":
                    st.error(f"SQL Error: {chat.get('error_message', 'Unknown error')}")
                elif status == "no_data":
                    st.info("Query executed successfully, but returned no data.")
                elif chat.get('results') is not None and not chat['results'].empty:
                    st.markdown(f"**Results:** {len(chat['results'])} rows returned")
                else:
                    st.markdown("**Results:** No data returned")
                    
                # Button to reuse this query
                if st.button("Reuse this query", key=f"reuse_query_{idx}", use_container_width=True):
                    st.session_state.generated_sql = chat['sql']
                    st.session_state.df_results = chat.get('results')
                    st.session_state.perplexity = chat.get('perplexity')
                    st.session_state.query_execution_status = chat.get('status')
                    st.session_state.query_error_message = chat.get('error_message', '')
                    st.rerun()
                
                st.markdown("---")

# Display app info in the footer
st.markdown("---")
st.caption("NL to SQL Generator v2.0 | Powered by Llama-3-SQLCoder-8B")
