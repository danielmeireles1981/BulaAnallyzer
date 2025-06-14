import time
from datetime import datetime
import pandas as pd
from rapidfuzz import fuzz
from embedder import get_model, load_index, search
import requests

def calcula_similaridade(resposta1, resposta2):
    if not resposta1 or not resposta2:
        return 0
    return fuzz.token_set_ratio(resposta1, resposta2)

def generate_response_openrouter(context, pergunta, or_api_key, model_name="openai/gpt-3.5-turbo"):
    prompt = (
        "Você é um assistente que responde perguntas sobre medicamentos usando EXCLUSIVAMENTE o material abaixo, extraído de bulas, segmentado em seções como INDICAÇÕES, POSOLOGIA, CONTRAINDICAÇÕES, REAÇÕES ADVERSAS, EFEITOS COLATERAIS, USO NA GRAVIDEZ, entre outras.\n\n"
        "Regras obrigatórias:\n"
        "- NÃO utilize conhecimento externo.\n"
        "- NÃO invente informações. Só responda se encontrar a informação nas passagens abaixo.\n"
        "- Se a resposta não estiver clara no contexto, diga explicitamente: 'Não foi possível encontrar a resposta no material fornecido.'\n"
        "- Sempre cite o nome do medicamento, o arquivo e, se possível, a seção.\n"
        "- Se a bula mencionar 'alívio de dores leves a moderadas', 'analgésico' ou termos similares em INDICAÇÕES, considere que o medicamento pode ser utilizado para dor de cabeça, pois isso está implícito nas bulas brasileiras.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pergunta: {pergunta}\nResponda de forma clara e objetiva."
    )
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {or_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Você é um assistente sobre bulas de medicamentos."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 512
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Erro na API do OpenRouter: {e}")
        return "Erro na geração de resposta."

# Pipeline
pdf_folder = 'pdfs'
out_dir = 'embeddings'
model = get_model()
embeddings, index, passages, sources, medicines = load_index(out_dir)

or_api_key = input("Cole sua OpenRouter API Key: ").strip()
model_name = "openai/gpt-3.5-turbo"  # Troque se quiser outro modelo

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

print(f"\nIniciando experimento com {len(perguntas)} perguntas...\n")

for idx, pergunta in enumerate(perguntas):
    print(f"⏳ [{idx+1}/{len(perguntas)}] Pergunta: {pergunta}")
    inicio = time.time()
    results = search(
        pergunta, model, index, passages, sources,
        top_k=10, threshold=None, debug=False, hybrid=True
    )
    context = "\n---\n".join([
        f"Medicamento: {medicines.get(src, 'Desconhecido')}\nArquivo: {src}\nTipo: {tag}\nTrecho: {passage[:300]}..."
        for tag, passage, src, score in results
    ])
    resposta = generate_response_openrouter(context, pergunta, or_api_key, model_name)
    fim = time.time()
    tempo_resposta = round(fim - inicio, 2)
    print(f"   ✅ Resposta: {resposta[:100]}... (tempo: {tempo_resposta}s)")

    time.sleep(5)

    # Reformulada
    pergunta_ref = perguntas_reformuladas[idx] if idx < len(perguntas_reformuladas) else ""
    resposta_ref = ""
    similaridade = ""
    fim_ref = inicio_ref = 0
    if pergunta_ref:
        print(f"   ➡️  Reformulando: {pergunta_ref}")
        inicio_ref = time.time()
        results_ref = search(
            pergunta_ref, model, index, passages, sources,
            top_k=8, threshold=None, debug=False, hybrid=True
        )
        context_ref = "\n---\n".join([
            f"Medicamento: {medicines.get(src, 'Desconhecido')}\nArquivo: {src}\nTipo: {tag}\nTrecho: {passage[:300]}..."
            for tag, passage, src, score in results_ref
        ])
        resposta_ref = generate_response_openrouter(context_ref, pergunta_ref, or_api_key, model_name)
        fim_ref = time.time()
        similaridade = calcula_similaridade(resposta, resposta_ref)
        print(f"      🔁 Reformulada: {resposta_ref[:80]}... | Consistência: {similaridade}% | Tempo: {round(fim_ref-inicio_ref,2)}s")
        time.sleep(5)

    resultados.append({
        "DataHora": datetime.now().isoformat(timespec='seconds'),
        "Pergunta": pergunta,
        "Resposta": resposta,
        "Tempo (s)": tempo_resposta,
        "Precisão A1": "",
        "Precisão A2": "",
        "Completude A1": "",
        "Completude A2": "",
        "Consistência": similaridade,
        "Pergunta Reformulada": pergunta_ref,
        "Resposta Reformulada": resposta_ref,
        "Tempo Ref (s)": round(fim_ref - inicio_ref, 2) if pergunta_ref else ""
    })

print("\nSalvando arquivo CSV...")
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("resultados_experimentos_openrouter.csv", index=False)
print("\n✅ Resultados salvos em 'resultados_experimentos_openrouter.csv'!\n")
