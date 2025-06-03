import time
from datetime import datetime
import pandas as pd
from rapidfuzz import fuzz
from embedder import get_model, load_index, search
from answer_engine import generate_response_gemini

# Função opcional para calcular similaridade entre duas respostas (consistência)
def calcula_similaridade(resposta1, resposta2):
    if not resposta1 or not resposta2:
        return 0
    return fuzz.token_set_ratio(resposta1, resposta2)

# Pipeline
pdf_folder = 'pdfs'
out_dir = 'embeddings'
model = get_model()
embeddings, index, passages, sources, medicines = load_index(out_dir)

gemini_api_key = input("Cole sua chave da API Gemini: ").strip()

# Perguntas do experimento (inclua aqui suas perguntas e as versões reformuladas)
perguntas = [
    "Liste os medicamentos analisados que podem ser usados por gestantes segundo as bulas.",
    "Liste os medicamentos que apresentam sonolência como efeito colateral nas bulas.",
    "Quais medicamentos têm indicação pediátrica? Informe faixas etárias e doses mencionadas.",
    "Liste os medicamentos indicados para dor de cabeça ou dores leves conforme as bulas.",
    "Quais medicamentos apresentam risco de reações alérgicas? Cite exemplos e trechos das bulas.",
    "Algum medicamento analisado é contraindicado para pacientes hipertensos segundo as bulas?",
    "Liste os medicamentos que devem ser tomados com alimentos ou em jejum, conforme as bulas.",
    "Cite os medicamentos que possuem advertência ou contraindicação sobre uso de álcool segundo as bulas.",
    "Indique os medicamentos que não devem ser usados por gestantes, segundo as bulas.",
    "Informe as doses recomendadas para adultos segundo as bulas dos medicamentos analisados."
]

perguntas_reformuladas = [
    "Cite os medicamentos que podem ser utilizados por mulheres grávidas conforme as bulas.",
    "Liste os medicamentos cuja bula cita sonolência como efeito adverso.",
    "Quais remédios são indicados para crianças? Mencione faixas etárias e dosagens especificadas.",
    "Quais medicamentos têm indicação para dor de cabeça ou dor leve nas bulas analisadas?",
    "Quais remédios possuem risco de alergia? Apresente exemplos com trechos da bula.",
    "Há medicamentos contraindicados para pessoas com pressão alta conforme as bulas analisadas?",
    "Quais remédios devem ser administrados junto com alimentos ou em jejum conforme a bula?",
    "Liste os medicamentos analisados que têm advertência sobre uso de álcool.",
    "Quais medicamentos têm restrição de uso para gestantes segundo as bulas?",
    "Liste as doses recomendadas para adultos de acordo com as bulas analisadas."
]


resultados = []

for idx, pergunta in enumerate(perguntas):
    inicio = time.time()
    results = search(
        pergunta, model, index, passages, sources,
        top_k=8, threshold=None, debug=False, hybrid=True
    )
    context = "\n---\n".join([
        f"Medicamento: {medicines.get(src, 'Desconhecido')}\nArquivo: {src}\nTipo: {tag}\nTrecho: {passage[:500]}..."
        for tag, passage, src, score in results
    ])
    resposta = generate_response_gemini(context, pergunta, gemini_api_key)
    fim = time.time()
    tempo_resposta = round(fim - inicio, 2)

    time.sleep(5)  # Aguarda 5 segundos para não exceder a quota da API

    # Gera resposta para a pergunta reformulada (para teste de consistência)
    pergunta_ref = perguntas_reformuladas[idx] if idx < len(perguntas_reformuladas) else ""
    resposta_ref = ""
    similaridade = ""
    fim_ref = inicio_ref = 0
    if pergunta_ref:
        inicio_ref = time.time()
        results_ref = search(
            pergunta_ref, model, index, passages, sources,
            top_k=8, threshold=None, debug=False, hybrid=True
        )
        context_ref = "\n---\n".join([
            f"Medicamento: {medicines.get(src, 'Desconhecido')}\nArquivo: {src}\nTipo: {tag}\nTrecho: {passage[:500]}..."
            for tag, passage, src, score in results_ref
        ])
        resposta_ref = generate_response_gemini(context_ref, pergunta_ref, gemini_api_key)
        fim_ref = time.time()
        similaridade = calcula_similaridade(resposta, resposta_ref)
        time.sleep(5)  # Aguarda mais 5 segundos para não exceder a quota da API

    resultados.append({
        "DataHora": datetime.now().isoformat(timespec='seconds'),
        "Pergunta": pergunta,
        "Resposta": resposta,
        "Tempo (s)": tempo_resposta,
        "Precisão A1": "",       # Preencher manualmente por avaliador 1
        "Precisão A2": "",       # Preencher manualmente por avaliador 2
        "Completude A1": "",     # Preencher manualmente por avaliador 1 (escala 1-5)
        "Completude A2": "",     # Preencher manualmente por avaliador 2 (escala 1-5)
        "Consistência": similaridade,
        "Pergunta Reformulada": pergunta_ref,
        "Resposta Reformulada": resposta_ref,
        "Tempo Ref (s)": round(fim_ref - inicio_ref, 2) if pergunta_ref else ""
    })

# Salva CSV para avaliação manual posterior
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("resultados_experimentos.csv", index=False)
print(df_resultados)
print("\nResultados salvos em resultados_experimentos.csv!")
