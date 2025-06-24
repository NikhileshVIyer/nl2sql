import streamlit as st
import os
import tempfile
from rag_functions import (
    extract_text_from_pdf,
    extract_tables_from_pdf,
    convert_tables_to_text,
    clean_and_split_text,
    generate_embeddings,
    index_embeddings,
    rag_pipeline
)

st.set_page_config(page_title="PDF Question Answering", layout="wide")

st.title("PDF Question Answering")
st.write("Upload a PDF document and ask questions about its content.")

# Sidebar for file upload and settings
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    st.header("Settings")
    chunk_size = st.slider("Chunk Size", min_value=500, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=100, step=50)
    extract_tables = st.checkbox("Extract tables from PDF", value=True)

    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing document..."):
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Extract and process text
                text = extract_text_from_pdf(tmp_path)

                if text and not text.startswith("Error"):
                    # Extract tables if enabled
                    tables = []
                    if extract_tables:
                        tables = extract_tables_from_pdf(tmp_path)
                        st.session_state.tables = tables
                        if isinstance(tables, str) and tables.startswith("Error"):
                            st.warning(f"Table extraction issue: {tables}")
                        elif tables:
                            st.success(f"Extracted {len(tables)} tables from the document")

                    # Clean and split text into chunks
                 # Convert tables to text
                table_texts = convert_tables_to_text(tables) if extract_tables and tables and not isinstance(tables, str) else []

                # Combine text and tables before chunking
                full_text = text + '\n\n' + '\n\n'.join(table_texts)

                # Clean and split combined content into chunks
                chunks = clean_and_split_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                if chunks:
                    st.success(f"Document processed with {len(table_texts)} table(s) and {len(chunks)} total chunks.")


                    if chunks:
                        # Generate embeddings
                        embeddings = generate_embeddings(chunks)

                        # Create index
                        index = index_embeddings(embeddings)

                        # Save to session state
                        st.session_state.chunks = chunks
                        st.session_state.index = index
                        st.session_state.document_name = uploaded_file.name

                        st.success(f"Document processed: {len(chunks)} chunks created")
                    else:
                        st.error("No chunks were created from the document.")
                else:
                    st.error("Failed to extract text from PDF.")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

# Main area for querying
if 'chunks' in st.session_state and 'index' in st.session_state:
    st.header(f"Ask questions about: {st.session_state.document_name}")
    query = st.text_input("Enter your question:")

    if query and st.button("Submit Question"):
        with st.spinner("Generating answer..."):
            try:
                # Append "no need reasoning" to the query
                modified_query = query.strip() + " no need reasoning"

                response, retrieved_chunks = rag_pipeline(
                    modified_query,
                    st.session_state.index,
                    st.session_state.chunks
                )

                st.subheader("Answer:")
                st.write(response)

                # Show source chunks
                with st.expander("View source chunks"):
                    for i, chunk in enumerate(retrieved_chunks):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.text(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                        st.divider()

            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")

    
    # Show extracted tables if available
    if 'tables' in st.session_state and st.session_state.tables and not isinstance(st.session_state.tables, str):
        with st.expander("View extracted tables"):
            for i, table in enumerate(st.session_state.tables):
                st.markdown(f"**Table {i+1}:**")
                st.dataframe(table)
                st.divider()
else:
    st.info("Please upload and process a document using the sidebar first.")

# Add information about the system
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info(
    "This app uses a Retrieval-Augmented Generation (RAG) pipeline to answer "
    "questions about PDF documents. It extracts text, splits it into chunks, "
    "embeds them using Sentence Transformers, and uses Ollama with Mistral "
    "for generating answers."
)
