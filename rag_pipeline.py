from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.llms import Ollama

def load_pipeline(pdf_path: str):
    """Builds a retrieval QA pipeline from a PDF file."""

    # 1. Load and split documents
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # 2. Create embeddings & vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)

    # 3. Local LLaMA 3 with Ollama
    llm = Ollama(model="llama3")

    # 4. Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    return qa_chain

def answer_query(qa_chain, query: str) -> str:
    """Takes a query string, returns the answer text."""
    result = qa_chain.invoke({"query": query})
    return result["result"]
