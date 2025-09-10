import streamlit as st
import os
import fitz  # PyMuPDF
import docx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
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

# Semantic filename search with fixed threshold
def search_filenames_semantically(base_folder, query, scope="Selected Folder", selected_folder=None, threshold=0.15):
    file_paths = []
    file_labels = []

    folders = [selected_folder] if scope == "Selected Folder" and selected_folder else os.listdir(base_folder)

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

    similarities = cosine_similarity(query_embedding, file_embeddings)[0]
    filtered = [(file_labels[i], file_paths[i]) for i in range(len(similarities)) if similarities[i] >= threshold]
    return filtered

# UI
st.set_page_config(page_title="🚑 Tango Bot", page_icon="📄")
st.title("🚑 Tango Bot")
st.write("Search and view documents by subject. Ask questions and get answers.")

base_folder = "documents"
subject_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
selected_subject = st.selectbox("📁 Select folder if file location known:", [""] + subject_folders)

# File name search
st.subheader("🔎 Search for a file name")
search_scope = st.radio("Search scope:", ["Selected Folder", "All Folders"])
filename_query = st.text_input("Type a keyword or phrase:")

clicked_file = None
if filename_query:
    results = search_filenames_semantically(base_folder, filename_query, search_scope, selected_subject if selected_subject else None, threshold=0.15)
    if results:
        st.markdown("### 📄 Matching Files:")
        for label, path in results:
            if st.button(f"{label}"):
                clicked_file = path
    else:
        st.warning("No matching files found above the threshold.")

# Load and index content from clicked file
if clicked_file:
    # Inline PDF viewer
    if clicked_file.endswith(".pdf"):
        with open(clicked_file, "rb") as f:
            binary_data = f.read()
        pdf_viewer(input=binary_data, width=700)
    else:
        st.info("📄 Word document selected — content will be used for search but not displayed.")

    # Index content for Q&A
    chunks = extract_chunks_from_folder(os.path.dirname(clicked_file))
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
    st.info("👆 Search for a file and click to select it before asking a question.")
