import streamlit as st
import os
import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from streamlit_pdf_viewer import pdf_viewer

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Chunking function with overlap
def extract_chunks_from_all_pdfs(base_folder, chunk_size=100, overlap=20):
    chunks = []
    for subject in os.listdir(base_folder):
        subject_path = os.path.join(base_folder, subject)
        if os.path.isdir(subject_path):
            for filename in os.listdir(subject_path):
                if filename.endswith(".pdf"):
                    path = os.path.join(subject_path, filename)
                    doc = fitz.open(path)
                    full_text = " ".join([page.get_text() for page in doc])
                    words = full_text.split()

                    # Chunking with overlap
                    i = 0
                    while i < len(words):
                        chunk = words[i:i+chunk_size]
                        chunks.append(" ".join(chunk))
                        i += chunk_size - overlap
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
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("📄 PDF Document Chatbot")
st.write("Organized by subject folders. Ask questions or view documents.")

# Folder-based filtering
base_folder = "documents"
subject_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
selected_subject = st.selectbox("📁 Choose a subject:", subject_folders)

# PDF selection
pdf_path = os.path.join(base_folder, selected_subject)
pdf_files = [f for f in os.listdir(pdf_path) if f.endswith(".pdf")]
selected_pdf = st.selectbox("📄 Choose a PDF:", pdf_files)

# Display selected PDF
if selected_pdf:
    full_path = os.path.join(pdf_path, selected_pdf)
    with open(full_path, "rb") as f:
        binary_data = f.read()
    pdf_viewer(input=binary_data, width=700)

# Load and index all documents
chunks = extract_chunks_from_all_pdfs(base_folder)
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
