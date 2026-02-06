from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
import gradio as gr

# -----------------------------
# Config
# -----------------------------

MODEL = "gpt-4.1-nano"
DB_NAME = "vector_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

load_dotenv()

# -----------------------------
# Setup
# -----------------------------

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=DB_NAME,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model=MODEL, temperature=0)

SYSTEM_PROMPT = """
You are a knowledgeable, precise assistant for the company Insurellm.
Use the provided context to answer the user's question.
If the answer is not present in the context, say you do not know.
Context:
{context}
"""

# -----------------------------
# RAG Function
# -----------------------------

def answer_question(question, history):
    docs = retriever.invoke(question)
    context = "\n\n".join(
        f"Source: {d.metadata.get('source_file')}\n{d.page_content}"
        for d in docs
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages)
    return response.content

# -----------------------------
# UI
# -----------------------------

gr.ChatInterface(
    answer_question,
    title="Xpert-Chroma | Enterprise Knowledge Assistant",
).launch()
