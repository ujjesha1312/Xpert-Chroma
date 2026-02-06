import os
import glob
from datetime import datetime
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
# -----------------------------
# Config
# -----------------------------
DB_NAME = "vector_db"
KNOWLEDGE_BASE = "knowledge-base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

load_dotenv()

# -----------------------------
# Load documents
# -----------------------------

documents = []
folders = glob.glob(f"{KNOWLEDGE_BASE}/*")

for folder in folders:
    doc_type = os.path.basename(folder)

    loader = DirectoryLoader(
        folder,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    folder_docs = loader.load()

    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        doc.metadata["source_file"] = os.path.basename(
            doc.metadata.get("source", "unknown")
        )
        documents.append(doc)

print(f"Loaded {len(documents)} documents")

# -----------------------------
# Chunking
# -----------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = text_splitter.split_documents(documents)

ingested_at = datetime.utcnow().isoformat()

for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = i
    chunk.metadata["ingested_at"] = ingested_at

print(f"Created {len(chunks)} chunks")

# -----------------------------
# Embeddings + Chroma
# -----------------------------

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

if os.path.exists(DB_NAME):
    Chroma(
        persist_directory=DB_NAME,
        embedding_function=embeddings
    ).delete_collection()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_NAME
)

print(f"Vector store created with {vectorstore._collection.count()} vectors")
