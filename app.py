import streamlit as st
import os
import fitz  # PyMuPDF for PDFs
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and parse PDF documents
def load_documents(folder):
    texts = []
    if not os.path.exists(folder):
        return ["No documents found."]
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if filename.endswith(".pdf"):
            doc = fitz.open(path)
            texts.append("\n".join([page.get_text() for page in doc]))
    return texts

# Create FAISS index
def create_index(texts):
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, texts

# Load documents and build index
texts = load_documents("documents")
index, text_chunks = create_index(texts)

# Load text generation model
generator = pipeline("text-generation", model="gpt2")

# Streamlit UI
st.set_page_config(page_title="Document Chatbot", page_icon="📚")
st.title("📚 Document Chatbot")
st.write("Ask a question and get answers from your uploaded PDFs.")

query = st.text_input("🔍 Your question:")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)
    context = "\n".join([text_chunks[i] for i in I[0]])

    prompt = f"Use the following context to answer:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_length=100, do_sample=True)[0]['generated_text']

    st.markdown("### 💡 Answer:")
    st.write(response)
