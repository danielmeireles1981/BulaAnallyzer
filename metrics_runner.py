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
    "Qual medicamento analisado pode ser utilizado durante a gravidez?",
    "Quais medicamentos analisados apresentam sonolência como efeito colateral?",
    "Quais medicamentos analisados são indicados para uso em crianças? Existem restrições de idade ou posologia específica?",
    "Quais medicamentos analisados são recomendados para alívio de dor de cabeça ou dores leves a moderadas?",
    "Quais medicamentos analisados apresentam risco de reações alérgicas? Cite exemplos e os trechos da bula.",
    "Quais medicamentos analisados apresentam contraindicação para pacientes com pressão alta (hipertensão)?",
    "Quais medicamentos analisados necessitam ser administrados junto com alimentos? Existem recomendações específicas sobre jejum ou horários?",
    "Quais medicamentos analisados possuem contraindicação ou advertência sobre o uso concomitante com álcool? Cite os trechos relevantes.",
    "Quais medicamentos analisados possuem restrição de uso durante a gravidez?",
    "Qual é a dosagem recomendada para adultos segundo as bulas dos medicamentos analisados?"
]

perguntas_reformuladas = [
    "Há algum medicamento dos analisados que pode ser usado por gestantes?",
    "Algum dos medicamentos apresenta sonolência como possível efeito adverso?",
    "Existem remédios indicados para crianças? Quais as faixas etárias e dosagens?",
    "Qual remédio desses serve para dor de cabeça ou outras dores leves?",
    "Que medicamentos trazem risco de alergia? Cite trechos da bula.",
    "Existe algum medicamento proibido para quem tem hipertensão?",
    "Há algum remédio que precisa ser tomado junto com comida ou em jejum?",
    "Quais não podem ser usados com álcool ou têm advertência sobre isso?",
    "Existe restrição ao uso dos remédios durante a gravidez?",
    "Quais as doses recomendadas para adultos conforme as bulas?"
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
