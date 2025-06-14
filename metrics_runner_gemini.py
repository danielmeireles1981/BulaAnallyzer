import time
from datetime import datetime
import pandas as pd
from rapidfuzz import fuzz
from embedder import get_model, load_index, search
from answer_engine import generate_response_gemini

def calcula_similaridade(resposta1, resposta2):
    if not resposta1 or not resposta2:
        return 0
    return fuzz.token_set_ratio(resposta1, resposta2)

pdf_folder = 'pdfs'
out_dir = 'embeddings'
model = get_model()
embeddings, index, passages, sources, medicines = load_index(out_dir)

gemini_api_key = input("Cole sua chave da API Gemini: ").strip()

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
    print(f"Pergunta {idx+1}/{len(perguntas)}: {pergunta}")

    inicio = time.time()
    results = search(
        pergunta, model, index, passages, sources,
        top_k=8, threshold=None, debug=False, hybrid=True
    )
    context = "\n---\n".join([
        f"Medicamento: {medicines.get(src, 'Desconhecido')}\nArquivo: {src}\nTipo: {tag}\nTrecho: {passage[:500]}..."
        for tag, passage, src, score in results
    ])

    prompt = f"""
Você é um assistente que responde perguntas sobre medicamentos usando EXCLUSIVAMENTE o material abaixo, extraído de bulas, segmentado em seções como INDICAÇÕES, POSOLOGIA, CONTRAINDICAÇÕES, REAÇÕES ADVERSAS, EFEITOS COLATERAIS, USO NA GRAVIDEZ, entre outras.

REGRAS OBRIGATÓRIAS:
- NÃO utilize conhecimento externo.
- NÃO invente informações. Só responda se encontrar a informação nas passagens abaixo.
- Se a resposta não estiver clara no contexto, diga explicitamente: 'Não foi possível encontrar a resposta no material fornecido.'
- SEMPRE cite: nome do medicamento, o arquivo (nome do PDF) e, se possível, a seção.
- Formate sua resposta da seguinte maneira:
  Medicamento: <nome> | Arquivo: <arquivo> | Seção: <seção ou trecho> | Resposta: <texto direto da bula/contexto>

Contexto extraído das bulas:
{context}

Pergunta: {pergunta}
Responda de forma clara, objetiva e seguindo as instruções acima.
    """

    resposta = generate_response_gemini(prompt, pergunta, gemini_api_key)
    fim = time.time()
    tempo_resposta = round(fim - inicio, 2)
    print(f"Tempo: {tempo_resposta}s | Resposta: {resposta[:90]}...")

    time.sleep(5)

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
        prompt_ref = f"""
Você é um assistente que responde perguntas sobre medicamentos usando EXCLUSIVAMENTE o material abaixo, extraído de bulas, segmentado em seções como INDICAÇÕES, POSOLOGIA, CONTRAINDICAÇÕES, REAÇÕES ADVERSAS, EFEITOS COLATERAIS, USO NA GRAVIDEZ, entre outras.

REGRAS OBRIGATÓRIAS:
- NÃO utilize conhecimento externo.
- NÃO invente informações. Só responda se encontrar a informação nas passagens abaixo.
- Se a resposta não estiver clara no contexto, diga explicitamente: 'Não foi possível encontrar a resposta no material fornecido.'
- SEMPRE cite: nome do medicamento, o arquivo (nome do PDF) e, se possível, a seção.
- Formate sua resposta da seguinte maneira:
  Medicamento: <nome> | Arquivo: <arquivo> | Seção: <seção ou trecho> | Resposta: <texto direto da bula/contexto>

Contexto extraído das bulas:
{context_ref}

Pergunta: {pergunta_ref}
Responda de forma clara, objetiva e seguindo as instruções acima.
        """
        resposta_ref = generate_response_gemini(prompt_ref, pergunta_ref, gemini_api_key)
        fim_ref = time.time()
        similaridade = calcula_similaridade(resposta, resposta_ref)
        print(f"Consistência (similaridade) [%]: {similaridade}")

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

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("resultados_experimentos_gemini.csv", index=False)
print(df_resultados)
print("\nResultados salvos em resultados_experimentos_gemini.csv!")
