from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def get_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

def get_embeddings(passages, model):
    embeddings = model.encode(passages, show_progress_bar=True)
    return embeddings

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search(query, model, index, passages, sources, top_k=3):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, top_k)
    results = [(passages[i], sources[i], D[0][j]) for j, i in enumerate(I[0])]
    return results
