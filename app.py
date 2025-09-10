import streamlit as st
import os
import fitz  # PyMuPDF for PDFs
import docx  # python-docx for Word files
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

# Chunking function for PDFs and Word docs
def extract_chunks_from_folder(folder_path, chunk_size=100, overlap=20):
    chunks = []
    for filename in os.listdir(folder_path):
        path = os.path.join(folder_path, filename)
        text = ""

        if filename.endswith(".pdf"):
            try:
                doc = fitz.open(path)
                text = " ".join([page.get_text() for page in doc])
            except Exception as e:
                st.warning(f"⚠️ Could not read PDF {filename}: {e}")
                continue

        elif filename.endswith((".docx", ".doc")):
            try:
                doc = docx.Document(path)
                text = "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                st.warning(f"⚠️ Could not read Word file {filename}: {e}")
                continue

        words = text.split()
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
st.set_page_config(page_title="🚑 Tango Bot ", page_icon="📄")
st.title("🚑 Tango Bot")
st.write("Search and view documents by subject. Ask questions and get answers.")

# Folder-based filtering
base_folder = "documents"
subject_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
selected_subject = st.selectbox("📁 Choose a subject:", subject_folders)

# File selection
folder_path = os.path.join(base_folder, selected_subject)
files = [f for f in os.listdir(folder_path) if f.endswith((".pdf", ".docx", ".doc"))]
selected_file = st.selectbox("📄 Choose a document:", files)

# Display message if no file selected
if not selected_file:
    st.info("👆 Please select a document to view or search.")
else:
    # Display PDF viewer or info message
    if selected_file.endswith(".pdf"):
        full_path = os.path.join(folder_path, selected_file)
        with open(full_path, "rb") as f:
            binary_data = f.read()
        pdf_viewer(input=binary_data, width=700)
    else:
        st.info("📄 Word document selected — content will be used for search but not displayed.")

    # Load and index selected folder
    chunks = extract_chunks_from_folder(folder_path)
    index, chunk_texts = create_index(chunks)

    # Search interface
    st.subheader("🔍 Ask a Question")
    query = st.text_input("Type your question here:")

    if query:
        query_embedding = model.encode([query])
        D, I = index.search(np.array(query_embedding), k=3)
        context = "\n\n".join([chunk_texts[i] for i in I[0]])

        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = generator(prompt, max_length=100, do_sample=True)[0]['generated_text']

        st.markdown("### 💡 Answer:")
        st.write(response)
