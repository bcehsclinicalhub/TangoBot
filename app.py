import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import docx
import xlrd
import fitz  # PyMuPDF

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and parse documents
def load_documents(folder):
    texts = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if filename.endswith(".docx"):
            doc = docx.Document(path)
            texts.append("\n".join([para.text for para in doc.paragraphs]))
        elif filename.endswith(".xls") or filename.endswith(".xlsx"):
            book = xlrd.open_workbook(path)
            for sheet in book.sheets():
                for row in range(sheet.nrows):
                    texts.append(" ".join(str(cell) for cell in sheet.row_values(row)))
        elif filename.endswith(".pdf"):
            doc = fitz.open(path)
            texts.append("\n".join([page.get_text() for page in doc]))
    return texts

# Embed and index
def create_index(texts):
    embeddings = model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, texts

# Load documents and create index
texts = load_documents("documents")
index, text_chunks = create_index(texts)

# Streamlit UI
st.title("📚 Document Chatbot")
query = st.text_input("Ask a question:")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)
    context = "\n".join([text_chunks[i] for i in I[0]])

    from transformers import pipeline
    generator = pipeline("text-generation", model="gpt2")
    prompt = f"Use the following context to answer:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_length=100, do_sample=True)[0]['generated_text']

    st.write("💡 Answer:")
    st.write(response)
