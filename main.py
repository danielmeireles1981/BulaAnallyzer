from pdf_extractor import extract_passages_from_pdfs
from embedder import get_model, get_embeddings, build_faiss_index, search
from answer_engine import generate_response_gemini

def main():
    pdf_folder = 'pdfs'
    print("Extraindo passagens dos PDFs...")
    passages, sources = extract_passages_from_pdfs(pdf_folder)

    print("Gerando embeddings...")
    model = get_model()
    embeddings = get_embeddings(passages, model)

    print("Construindo índice FAISS...")
    index = build_faiss_index(embeddings)

    gemini_api_key = input("Cole sua chave da API Gemini: ").strip()

    print("Sistema pronto para perguntas.")
    while True:
        question = input("\nDigite sua pergunta (ou 'sair' para encerrar): ")
        if question.lower() in ['sair', 'exit', 'q']:
            break
        results = search(question, model, index, passages, sources, top_k=3)
        context = "\n---\n".join([f"Fonte: {src}\nTrecho: {txt[:500]}..." for txt, src, _ in results])
        print(f"\nTrechos relevantes encontrados:\n{context}")
        resposta = generate_response_gemini(context, question, gemini_api_key)
        print("\nResposta gerada pela IA:")
        print(resposta)

if __name__ == "__main__":
    main()
