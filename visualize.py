import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.manifold import TSNE
import plotly.graph_objects as go

DB_NAME = "vector_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=DB_NAME,
    embedding_function=embeddings
)

collection = vectorstore._collection
result = collection.get(include=["embeddings", "documents", "metadatas"])

vectors = np.array(result["embeddings"])
documents = result["documents"]
metadatas = result["metadatas"]

doc_types = [m["doc_type"] for m in metadatas]
color_map = {
    "employees": "blue",
    "products": "green",
    "contracts": "red",
    "company": "orange"
}
colors = [color_map.get(t, "gray") for t in doc_types]

tsne = TSNE(n_components=2, random_state=42)
reduced = tsne.fit_transform(vectors)

fig = go.Figure(
    data=[
        go.Scatter(
            x=reduced[:, 0],
            y=reduced[:, 1],
            mode="markers",
            marker=dict(size=6, color=colors, opacity=0.8),
            text=[d[:120] for d in documents],
            hoverinfo="text"
        )
    ]
)

fig.update_layout(
    title="Xpert-Chroma Vector Space (t-SNE)",
    width=900,
    height=600
)

fig.show()
