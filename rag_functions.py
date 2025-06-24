import os
import re
import faiss
import pdfplumber
import camelot
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------
# Data Ingestion Functions
# ---------------------------
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF using pdfplumber."""
    text = ''
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"


def extract_tables_from_pdf(pdf_path):
    """Extracts tables from a PDF using pdfplumber."""
    try:
        extracted_tables = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        df = pd.DataFrame(table)# Make sure it's not empty
                        df = df.ffill(axis=0).ffill(axis=1)
                        extracted_tables.append(df)

        return extracted_tables

    except Exception as e:
        return f"Error extracting tables: {str(e)}"




# ---------------------------
# Table Processing Helper
# ---------------------------
def convert_tables_to_text(tables):
    """
    Converts a list of table dataframes into text representations.
    Each table is prefixed with a header (e.g., TABLE 1:) and converted to a string.
    """
    table_texts = []
    if tables and not isinstance(tables, str):
        for i, table_df in enumerate(tables):
            # Convert DataFrame to a string representation without index
            table_text = f"TABLE {i+1}:\n{table_df.to_string(index=False)}"
            table_texts.append(table_text)
    return table_texts

# ---------------------------
# Preprocessing Functions
# ---------------------------
def clean_and_split_text(text, chunk_size=1000, chunk_overlap=100):
    """Cleans text and splits it into chunks using LangChain's splitter."""
    # Basic cleaning: replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks

# ---------------------------
# Embedding and Indexing
# ---------------------------
def generate_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    """Generates vector embeddings for a list of text chunks using SentenceTransformer."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

def index_embeddings(embeddings):
    """Indexes embeddings using FAISS for efficient similarity search."""
    # Convert tensor embeddings to a NumPy array
    embeddings = embeddings.cpu().detach().numpy()
    # Handle the case of a single embedding vector
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, axis=0)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

# ---------------------------
# RAG Pipeline Functions
# ---------------------------
def retrieve_documents(query, index, chunks, model_name='all-MiniLM-L6-v2', top_k=5):
    """Retrieves the top-k document chunks relevant to the query using similarity search."""
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query], convert_to_tensor=True)
    query_embedding = query_embedding.cpu().detach().numpy()
    # Ensure query_embedding is 2D (even for a single query)
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks

def generate_response(prompt):
    """Generates a response using the Mistral model via Ollama."""
    llm = Ollama(model="tinyllama:1.1b")
    response = llm.invoke(prompt)
    return response

def rag_pipeline(query, index, chunks):
    """Combines retrieval and generation to form a complete RAG pipeline."""
    retrieved_chunks = retrieve_documents(query, index, chunks)
    context = ' '.join(retrieved_chunks)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = generate_response(prompt)
    return response, retrieved_chunks  # Return both response and chunks for reference
