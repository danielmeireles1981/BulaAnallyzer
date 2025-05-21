def generate_response_gemini(context, question, api_key):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = (
        "Você é um assistente que responde perguntas sobre medicamentos usando EXCLUSIVAMENTE o material abaixo, "
        "extraído de bulas, segmentado em seções como INDICAÇÕES, POSOLOGIA, CONTRAINDICAÇÕES, REAÇÕES ADVERSAS, EFEITOS COLATERAIS, USO NA GRAVIDEZ, entre outras.\n\n"
        "Regras obrigatórias:\n"
        "- NÃO utilize conhecimento externo.\n"
        "- NÃO invente informações. Só responda se encontrar a informação nas passagens abaixo.\n"
        "- Se a resposta não estiver clara no contexto, diga explicitamente: 'Não foi possível encontrar a resposta no material fornecido.'\n"
        "- Sempre cite o nome do medicamento, o arquivo e, se possível, a seção.\n"
        "- Se a resposta envolver diferentes medicamentos, liste cada um com seu trecho correspondente.\n"
        "Se a bula mencionar 'alívio de dores leves a moderadas', 'analgésico' ou termos similares em INDICAÇÕES, considere que o medicamento pode ser utilizado para dor de cabeça, pois isso está implícito nas bulas brasileiras.\n"
        "\nContexto:\n"
        f"{context}\n"
        f"\nPergunta: {question}\n"
        "Responda de forma clara e objetiva."
    )

    # DEBUG: mostrar prompt
    print("\n==========[PROMPT ENVIADO AO GEMINI]==========")
    print(prompt)
    print("=" * 48)

    response = model.generate_content(prompt)

    # DEBUG: mostrar resposta bruta da API
    print("\n==========[RESPOSTA BRUTA DO GEMINI]==========")
    print(response.text.strip())
    print("=" * 48)

    return response.text.strip()
