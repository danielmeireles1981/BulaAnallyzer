import os
from pdf_extractor import extract_passages_from_pdfs
from embedder import (
    get_model, get_embeddings, build_faiss_index, save_index, load_index, search
)
from answer_engine import generate_response_gemini

def main():
    pdf_folder = 'pdfs'
    out_dir = 'embeddings'
    use_cache = os.path.exists(os.path.join(out_dir, 'faiss.index'))
    model = get_model()

    if use_cache:
        print("Carregando embeddings e índice FAISS salvos...")
        embeddings, index, passages, sources, medicines = load_index(out_dir)
    else:
        print("Extraindo passagens dos PDFs e gerando embeddings...")
        passages, sources, medicines = extract_passages_from_pdfs(pdf_folder)
        embeddings = get_embeddings(passages, model)
        print("Construindo índice FAISS...")
        index = build_faiss_index(embeddings)
        print("Salvando embeddings e índice FAISS...")
        save_index(embeddings, index, passages, sources, medicines, out_dir)

    gemini_api_key = input("Cole sua chave da API Gemini: ").strip()

    print("Sistema pronto para perguntas.")
    while True:
        question = input("\nDigite sua pergunta (ou 'sair' para encerrar): ")
        if question.lower() in ['sair', 'exit', 'q']:
            break
        # O search agora retorna uma tupla: (tag, passage, src, score)
        results = search(
            question,
            model,
            index,
            passages,
            sources,
            top_k=8,
            threshold=None,
            debug=True,      # Ativa log detalhado do embedder!
            hybrid=True      # Ativa busca híbrida
        )
        print("\n[DEBUG] Trechos retornados pelo sistema:")
        for tag, passage, src, score in results:
            print(f"Tipo: {tag} | Arquivo: {src} | Score: {score} | Medicamento: {medicines.get(src, 'Desconhecido')}")
            print(f"Trecho: {passage[:120].replace('\n', ' ')}...")
            print("-" * 80)
        context = "\n---\n".join([
            f"Medicamento: {medicines.get(src, 'Desconhecido')}\nArquivo: {src}\nTipo: {tag}\nTrecho: {passage[:500]}..."
            for tag, passage, src, score in results
        ])
        print(f"\nTrechos relevantes encontrados:\n{context}")
        resposta = generate_response_gemini(context, question, gemini_api_key)
        print("\nResposta gerada pela IA:")
        print(resposta)

if __name__ == "__main__":
    main()
