from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
import re

def get_model(model_name='all-mpnet-base-v2'):
    return SentenceTransformer(model_name)

def get_embeddings(passages, model):
    embeddings = model.encode(passages, show_progress_bar=True)
    return embeddings

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_index(embeddings, index, passages, sources, medicines, out_dir='embeddings'):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'embeddings.npy'), embeddings)
    faiss.write_index(index, os.path.join(out_dir, 'faiss.index'))
    with open(os.path.join(out_dir, 'passages.pkl'), 'wb') as f:
        pickle.dump(passages, f)
    with open(os.path.join(out_dir, 'sources.pkl'), 'wb') as f:
        pickle.dump(sources, f)
    with open(os.path.join(out_dir, 'medicines.pkl'), 'wb') as f:
        pickle.dump(medicines, f)

def load_index(out_dir='embeddings'):
    required_files = ['embeddings.npy', 'faiss.index', 'passages.pkl', 'sources.pkl', 'medicines.pkl']
    for f in required_files:
        if not os.path.exists(os.path.join(out_dir, f)):
            raise FileNotFoundError(f"{f} não encontrado na pasta {out_dir}. Execute o processamento inicial!")
    embeddings = np.load(os.path.join(out_dir, 'embeddings.npy'))
    index = faiss.read_index(os.path.join(out_dir, 'faiss.index'))
    with open(os.path.join(out_dir, 'passages.pkl'), 'rb') as f:
        passages = pickle.load(f)
    with open(os.path.join(out_dir, 'sources.pkl'), 'rb') as f:
        sources = pickle.load(f)
    with open(os.path.join(out_dir, 'medicines.pkl'), 'rb') as f:
        medicines = pickle.load(f)
    return embeddings, index, passages, sources, medicines

def keyword_match_search(query, passages, sources, top_k=5):
    """Busca por palavras-chave da query nos trechos."""
    keywords = [w for w in re.findall(r'\w+', query.lower()) if len(w) > 3]
    hits = []
    for idx, passage in enumerate(passages):
        count = sum(1 for word in keywords if word in passage.lower())
        if count > 0:
            hits.append(("KEYWORD", passage, sources[idx], count))
    hits.sort(key=lambda x: -x[3])  # mais palavras-chave primeiro
    return hits[:top_k]

def regex_search(patterns, passages, sources, top_k=5):
    """Busca por padrões regex nos trechos (patterns = lista de regexs ou palavras)."""
    hits = []
    for idx, passage in enumerate(passages):
        for pat in patterns:
            if re.search(pat, passage, re.IGNORECASE):
                hits.append(("REGEX", passage, sources[idx], pat))
                break
    return hits[:top_k]

def expand_query(query):
    expansions = {
        "pressão alta": ["hipertensão", "cardíaco", "problemas cardíacos", "doença cardiovascular", "circulação", "vasculopatia"],
        "hipertensão": ["pressão alta", "cardíaco", "problemas cardíacos", "doença cardiovascular", "circulação", "vasculopatia"],
        "dor de cabeça": ["analgésico", "alívio de dores", "dores leves", "cefaleia"],
        "criança": ["uso pediátrico", "pediátrico", "idade", "adolescente", "infantil"],
        "gravidez": ["gestante", "gestação", "risco na gravidez", "mulheres grávidas"],
    }
    for key in expansions:
        if key in query.lower():
            extras = [w for w in expansions[key] if w not in query.lower()]
            if extras:
                return query + " " + " ".join(extras)
    return query

def log_results(question, faiss_results, keyword_results, regex_results):
    print("\n====== LOG DE BUSCA ======")
    print(f"Pergunta: {question}")
    print(f"Trechos FAISS: {len(faiss_results)}")
    print(f"Trechos por keyword: {len(keyword_results)}")
    print(f"Trechos por regex: {len(regex_results)}")
    print("--- FAISS Top 3 ---")
    for tag, passage, src, score in faiss_results[:3]:
        print(f" [FAISS] Fonte: {src} | Score (L2): {score:.4f}")
        print(f"   Início: {passage[:80]}")
    print("--- Keyword Match ---")
    for tag, passage, src, count in keyword_results:
        print(f" [KW] Fonte: {src} | Palavra-chave count: {count}")
        print(f"   Início: {passage[:80]}")
    print("--- Regex Match ---")
    for tag, passage, src, pat in regex_results:
        print(f" [REGEX] Fonte: {src} | Padrão: {pat}")
        print(f"   Início: {passage[:80]}")
    print("==========================\n")

def search(query, model, index, passages, sources, top_k=8, threshold=None, debug=False, hybrid=True):
    expanded_query = expand_query(query)
    query_embedding = model.encode([expanded_query])
    D, I = index.search(query_embedding, top_k)
    faiss_results = []
    for j, i in enumerate(I[0]):
        if 0 <= i < len(passages):
            if (threshold is None) or (D[0][j] < threshold):
                faiss_results.append(("FAISS", passages[i], sources[i], D[0][j]))
    # Busca por keyword match e regex match
    keyword_results = []
    regex_results = []
    if hybrid:
        keyword_results = keyword_match_search(query, passages, sources, top_k=top_k)
        # Exemplos de padrões médicos para regex: "hipertens" pega hipertensão, hipertensivo, etc.
        patterns = []
        if any(kw in query.lower() for kw in ["pressão alta", "hipertensão"]):
            patterns += [r'hipertens', r'press[ãa]o alta', r'card[íi]aco', r'card[ií]aca', r'doen[cç]a cardiovascular']
        if any(kw in query.lower() for kw in ["criança", "pediátrico"]):
            patterns += [r'crian[çc]a', r'pedi[áa]trico', r'infantil']
        if "gravidez" in query.lower():
            patterns += [r'gravid', r'gestante', r'gesta[çc][aã]o']
        regex_results = regex_search(patterns, passages, sources, top_k=top_k)
    # Juntar todos, evitando duplicatas
    all_results = []
    seen = set()
    for tag, passage, src, score in faiss_results + keyword_results + regex_results:
        key = (passage[:50], src)
        if key not in seen:
            all_results.append((tag, passage, src, score))
            seen.add(key)
    # Debug/log detalhado
    if debug:
        log_results(query, faiss_results, keyword_results, regex_results)
    return all_results
