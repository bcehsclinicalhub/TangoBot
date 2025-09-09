import streamlit as st
import os
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load embedding model once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load and chunk PDF text
def extract_chunks_from_pdfs(folder, chunk_size=100):
    chunks = []
    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            path = os.path.join(folder, filename)
            doc = fitz.open(path)
            text = " ".join([page.get_text() for page in doc])
            words = text.split()
            chunks += [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Embed and index chunks
@st.cache_resource
def create_index(chunks):
    embeddings = model.encode(chunks, batch_size=16, show_progress_bar=True)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index, chunks

# Load and index documents
chunks = extract_chunks_from_pdfs("documents")
index, chunk_texts = create_index(chunks)

# Load generator model
@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="distilgpt2")

generator = load_generator()

# Streamlit UI
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Document Chatbot")
st.write("Ask a question and get answers from your uploaded PDFs.")

query = st.text_input("🔍 Your question:")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)
    context = "\n\n".join([chunk_texts[i] for i in I[0]])

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_length=100, do_sample=True)[0]['generated_text']

    st.markdown("### 💡 Answer:")
    st.write(response)
