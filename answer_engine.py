import google.generativeai as genai

def generate_response_gemini(context, question, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    prompt = (
        f"Você é um assistente que responde perguntas com base em bulas de medicamentos extraídas de PDFs.\n"
        f"Contexto:\n{context}\n"
        f"Pergunta: {question}\n"
        f"Responda de forma clara e objetiva, citando a fonte sempre que possível."
    )
    response = model.generate_content(prompt)
    return response.text.strip()
