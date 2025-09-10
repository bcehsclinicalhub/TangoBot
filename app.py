import streamlit as st
import os
import fitz  # PyMuPDF
import docx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from streamlit_pdf_viewer import pdf_viewer

# Load models
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_generator():
    return pipeline("text-generation", model="distilgpt2")

model = load_model()
generator = load_generator()

# Chunking function
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

# Semantic filename search
def search_filenames_semantically(base_folder, query, scope="Selected Folder", selected_folder=None):
    file_paths = []
    file_labels = []

    folders = [selected_folder] if scope == "Selected Folder" else os.listdir(base_folder)

    for folder in folders:
        folder_path = os.path.join(base_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".pdf", ".docx", ".doc")):
                file_paths.append(os.path.join(folder_path, filename))
                file_labels.append(f"{folder}/{filename}")

    if not file_labels:
        return []

    query_embedding = model.encode([query])
    file_embeddings = model.encode(file_labels)
    index = faiss.IndexFlatL2(len(file_embeddings[0]))
    index.add(np.array(file_embeddings))
    D, I = index.search(np.array(query_embedding), k=min(10, len(file_labels)))

    results = [(file_labels[i], file_paths[i]) for i in I[0]]
    return results

# UI
st.set_page_config(page_title="🚑 Tango Bot", page_icon="📄")
st.title("🚑 Tango Bot")
st.write("Search and view documents by subject. Ask questions and get answers.")

base_folder = "documents"
subject_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
selected_subject = st.selectbox("📁 Choose a subject:", subject_folders)

folder_path = os.path.join(base_folder, selected_subject)
files = [f for f in os.listdir(folder_path) if f.endswith((".pdf", ".docx", ".doc"))]
selected_file = st.selectbox("📄 Choose a document:", files)

# File name search
st.subheader("🔎 Search for a file name")
search_scope = st.radio("Search scope:", ["Selected Folder", "All Folders"])
filename_query = st.text_input("Type a keyword or phrase:")

clicked_file = None
if filename_query:
    results = search_filenames_semantically(base_folder, filename_query, search_scope, selected_subject)
    if results:
        for label, path in results:
            if st.button(f"📄 {label}"):
                clicked_file = path
    else:
        st.warning("No matching files found.")

# Determine which file to load
file_to_load = clicked_file if clicked_file else os.path.join(folder_path, selected_file) if selected_file else None

if file_to_load:
    if file_to_load.endswith(".pdf"):
        with open(file_to_load, "rb") as f:
            binary_data = f.read()
        pdf_viewer(input=binary_data, width=700)
    else:
        st.info("📄 Word document selected — content will be used for search but not displayed.")

    # Load and index content
    chunks = extract_chunks_from_folder(os.path.dirname(file_to_load))
    index, chunk_texts = create_index(chunks)

    st.subheader("🧠 Ask a Question")
    query = st.text_input("Type your question here:")

    if query:
        query_embedding = model.encode([query])
        D, I = index.search(np.array(query_embedding), k=3)
        context = "\n\n".join([chunk_texts[i] for i in I[0]])

        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = generator(prompt, max_length=100, do_sample=True)[0]['generated_text']

        st.markdown("### 💡 Answer:")
        st.write(response)
else:
    st.info("👆 Please select or search for a document to view or query.")
