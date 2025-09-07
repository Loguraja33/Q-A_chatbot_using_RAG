import streamlit as st
from rag_pipeline import load_pipeline, answer_query

st.set_page_config(page_title="📚 Local LLaMA 3 RAG", layout="wide")

st.title("📖 Local RAG with LLaMA 3 + FAISS")
st.write("Upload a PDF and ask questions (runs fully locally with Ollama 🚀)")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Save uploaded file
    pdf_path = "temp.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load pipeline
    qa_chain = load_pipeline(pdf_path)

    # Ask questions
    query = st.text_input("Ask a question about your PDF:")

    if query:
        with st.spinner("Thinking..."):
            answer = answer_query(qa_chain, query)
        st.success("Answer:")
        st.write(answer)
