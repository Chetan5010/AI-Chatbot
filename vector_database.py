from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Setup
ollama_model_name = "deepseek-r1:1.5b"
FAISS_DB_PATH = "vectorstore/db_faiss"
file_path = "universal_declaration_of_human_rights.pdf"

# Load PDF
loader = PDFPlumberLoader(file_path)
documents = loader.load()

# Chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
text_chunks = text_splitter.split_documents(documents)

# Embed
embeddings = OllamaEmbeddings(model=ollama_model_name)
faiss_db = FAISS.from_documents(text_chunks, embedding=embeddings)

# Save index
faiss_db.save_local(FAISS_DB_PATH)