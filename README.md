# Xpert-Chroma

Xpert-Chroma is a **low-cost, enterprise-style Retrieval Augmented Generation (RAG) system** designed to serve as an internal knowledge assistant.

It enables employees to query internal documents using **semantic search over vector embeddings**, ensuring answers are **grounded in real data** rather than hallucinated by a language model.

The project is inspired by real-world InsurTech use cases, where **accuracy, traceability, and cost efficiency** are critical.

## What Xpert-Chroma Does

- Ingests internal markdown documents
- Splits documents into semantic, context-preserving chunks
- Generates vector embeddings using a cost-efficient HuggingFace model
- Stores embeddings in a persistent Chroma vector database
- Retrieves the most relevant chunks for a user query
- Uses retrieved context to generate grounded answers via an LLM
- Visualizes the embedding space for explainability

## Why Retrieval Augmented Generation (RAG)?

Large Language Models on their own can:
- hallucinate facts
- give outdated answers
- respond confidently without evidence

Xpert-Chroma uses **RAG** to:

- Ground answers in verified internal documents  
- Improve trust and reliability  
- Reduce reliance on expensive model calls  
- Update knowledge without retraining models  

## System Architecture
```bash
User Query
↓
Vector Similarity Search (ChromaDB)
↓
Top-K Relevant Document Chunks
↓
LLM with Context Injection
↓
Grounded Answer
```

## Project Structure
```
Xpert-Chroma/
│
├── ingest.py # Load, chunk, embed, and store documents
├── query.py # Inspect semantic retrieval results
├── visualize.py # 2D visualization of embedding space (t-SNE)
├── app.py # RAG-based chat interface
│
├── knowledge-base/ # Internal markdown documents (local)
│ ├── employees/
│ ├── products/
│ ├── contracts/
│ └── company/
│
├── requirements.txt
├── .gitignore
└── .env # API keys (not committed)
```

## Metadata Strategy

Each document chunk is enriched with metadata to ensure traceability:

- `doc_type` — document category  
- `source_file` — original file name  
- `chunk_id` — unique identifier  
- `ingested_at` — ingestion timestamp  

This enables:
- debugging and inspection
- source attribution
- future citation-based answers
- enterprise auditability

## Tech Stack

- **Language**: Python  
- **RAG Framework**: LangChain  
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`  
- **Vector Database**: ChromaDB  
- **LLM**: OpenAI `gpt-4.1-nano` (low-cost)  
- **Visualization**: scikit-learn (t-SNE) + Plotly  
- **UI**: Gradio  
- **Environment Management**: python-dotenv  

## How to Run

### 1️. Install dependencies
```bash
pip install -r requirements.txt
```
2️. Build the vector database
```bash
python ingest.py
```
3️. Inspect semantic retrieval
```bash
python query.py
```
4️. Visualize embeddings (optional)
```bash
python visualize.py
```
5️. Launch the RAG assistant
```bash
python app.py
```
### Design Principles
Low cost by default — efficient embeddings and lightweight LLMs

Separation of concerns — ingest, retrieval, visualization, generation

Explainability — retrieval can be inspected independently

Enterprise-ready — metadata, traceability, predictable behavior


### Use Cases
Internal employee knowledge base

Policy and contract lookup

Product documentation search

Enterprise AI assistant backend

RAG experimentation and prototyping
