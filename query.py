from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
DB_NAME = "vector_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
try:
    vectorstore = Chroma(
        persist_directory=DB_NAME,
        embedding_function=embeddings
    )
except Exception:
    raise RuntimeError("Vector DB not found. Run ingest.py first.")
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
query = input("Enter your query: ")
docs = retriever.invoke(query)
print("\n--- Retrieved Chunks ---\n")
for i, doc in enumerate(docs, start=1):
    print(f"[{i}] Source: {doc.metadata.get('source_file')}")
    print(doc.page_content[:500])
    print("-" * 60)
