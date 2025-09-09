import streamlit as st
import os
import fitz  # PyMuPDF
import numpy as np
import faiss
import base64
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from streamlit_pdf_viewer import pdf_viewer

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load and chunk PDF text
def extract_chunks_from_pdfs(folder, chunk_size=100):
    folder_path = os.path.join(os.path.dirname(__file__), folder)
    if not os.path.exists(folder_path):
        st.error(f"📁 Folder not found: {folder_path}")
        st.stop()

    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
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

# Load generator model
@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="distilgpt2")

generator = load_generator()

# Streamlit UI
st.set_page_config(page_title="Tango Bot", page_icon="📄")
st.title("📄 Tango Bot")
st.write("Ask a question and get answers from your uploaded PDFs.")

# Display PDF viewer
st.subheader("📑 View a PDF")
pdf_files = [f for f in os.listdir("documents") if f.endswith(".pdf")]
selected_pdf = st.selectbox("Choose a PDF to view:", pdf_files)

if selected_pdf:
    with open(os.path.join("documents", selected_pdf), "rb") as f:
        binary_data = f.read()
    pdf_viewer(input=binary_data, width=700)

# Load and index documents
chunks = extract_chunks_from_pdfs("documents")
index, chunk_texts = create_index(chunks)

# Chatbot interface
st.subheader("💬 Ask a Question")
query = st.text_input("🔍 Your question:")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)
    context = "\n\n".join([chunk_texts[i] for i in I[0]])

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_length=100, do_sample=True)[0]['generated_text']

    st.markdown("### 💡 Answer:")
    st.write(response)
