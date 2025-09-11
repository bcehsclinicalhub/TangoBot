import streamlit as st
import os
import fitz  # PyMuPDF
import docx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_pdf_viewer import pdf_viewer
import re

# Load models
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Extract links from text
def extract_links(text):
    return re.findall(r'https?://\S+', text)

# Chunking function
def extract_chunks_and_links(folder_path, chunk_size=100, overlap=20):
    chunks = []
    all_links = []
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

        all_links.extend(extract_links(text))

    return chunks, list(set(all_links))

# Semantic filename search with fixed threshold
def search_filenames_semantically(base_folder, query, scope="Selected Folder", selected_folder=None, threshold=0.7):
    file_paths = []
    file_labels = []

    folders = [selected_folder] if scope == "Selected Folder" and selected_folder else os.listdir(base_folder)

    for folder in folders:
        folder_path = os.path.join(base_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".pdf", ".docx", ".doc")):
                full_path = os.path.join(folder_path, filename)
                label = f"{folder}/{filename}"
                file_paths.append(full_path)
                file_labels.append(label)

    if not file_labels:
        return []

    # Step 1: Exact keyword match boost
    keyword_matches = [(label, path) for label, path in zip(file_labels, file_paths) if query.lower() in label.lower()]

    # Step 2: Semantic similarity
    query_embedding = model.encode([query])
    file_embeddings = model.encode(file_labels)
    similarities = cosine_similarity(query_embedding, file_embeddings)[0]

    semantic_matches = [
        (file_labels[i], file_paths[i], similarities[i])
        for i in range(len(similarities)) if similarities[i] >= threshold
    ]

    # Step 3: Combine and prioritize
    seen = set(label for label, _ in keyword_matches)
    combined = keyword_matches + [(label, path) for label, path, _ in sorted(semantic_matches, key=lambda x: x[2], reverse=True) if label not in seen]

    return combined

# UI setup
st.set_page_config(page_title="🚑 Tango Bot", page_icon="📄")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("BCEHS_Logo_2.jpg", width=300)


# Hide Streamlit footer
st.markdown("""
    <style>
    footer {visibility: hidden;}
    .stApp {margin-bottom: 0px;}
    </style>
""", unsafe_allow_html=True)

st.title("🚑 Tango Bot")
st.write("Search and view documents by subject.")

# Folder selection
base_folder = "documents"
subject_folders = sorted([f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))])
selected_subject = st.selectbox("📁 Select folder if file location known:", [""] + subject_folders)

# Initialize session state for clicked file
if "clicked_file" not in st.session_state:
    st.session_state.clicked_file = None

# File dropdown under folder with sort toggle
if selected_subject:
    folder_path = os.path.join(base_folder, selected_subject)
    sort_by_date = st.toggle("📅 Sort by last modified date", value=False)

    file_candidates = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".pdf", ".docx", ".doc"))
    ]

    if not file_candidates:
        st.info("📂 No supported files found in this folder.")
    else:
        file_options = sorted(
            file_candidates,
            key=lambda f: os.path.getmtime(os.path.join(folder_path, f)),
            reverse=sort_by_date
        )

        selected_file = st.selectbox("📄 Select a file:", [""] + file_options)

        if selected_file:
            st.session_state.clicked_file = os.path.join(folder_path, selected_file)

# Optional semantic search
st.subheader("🔎 Search for a file name")
search_scope = st.radio("Search scope:", ["Selected Folder", "All Folders"], index=1)
filename_query = st.text_input("Type a keyword or phrase:")

if filename_query:
    results = search_filenames_semantically(base_folder, filename_query, search_scope, selected_subject if selected_subject else None, threshold=0.15)
    if results:
        st.markdown("### 📄 Matching Files:")
        for label, path in results:
            if st.button(f"{label}"):
                st.session_state.clicked_file = path
    else:
        st.warning("No matching files found above the threshold.")

# Display selected file and extract links
if st.session_state.clicked_file:
    if st.session_state.clicked_file.endswith(".pdf"):
        with open(st.session_state.clicked_file, "rb") as f:
            binary_data = f.read()
        pdf_viewer(input=binary_data, width=700)

        # Extract links from the selected PDF
        folder_path = os.path.dirname(st.session_state.clicked_file)
        chunks, links = extract_chunks_and_links(folder_path)

        if links:
            st.markdown("### 🔗 Links found in this document:")
            for link in links:
                st.markdown(f"- [{link}]({link})")
        else:
            st.info("No links found in this document.")
    else:
        st.info("📄 Word document selected — content will be used for search but not displayed.")
else:
    st.info("👆 Select a file or search to view it.")
