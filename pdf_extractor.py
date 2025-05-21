import os
import pdfplumber
import re
import unicodedata

def extract_medicine_name(text):
    lines = text.strip().split('\n')
    ignore_words = {
        "comprimido", "mg", "ml", "solução", "genérico", "farmacêutica", "sa", "ltda",
        "identificação do medicamento", "apresentação", "bula", "medicamento genérico",
        "uso oral", "apresentações", "comprimidos", "medicamento", "modelo", "brainfarma", "hipolabor",
        "composição", "indicações", "posologia", "contrainidicação", "contraindicações",
        "legrand", "ems", "medley", "teuto", "zambon", "prati-donaduzzi", "sandoz", "sanofi", "ache", "novaquimica", "neo quimica",
        "receberam"
    }
    # 1. Busca padrão “XXX 200 mg”, “XXX 500mg”, etc
    for line in lines[:40]:
        match = re.search(r'\b([A-ZÇÃÕÉÊÁÍÓÚÂÔÜa-zçãõâêôúüéíóáà]{5,})\b[\s\-]*[\d\.,]+\s*mg', line)
        if match:
            nome = match.group(1)
            if nome.lower() not in ignore_words and not nome.isdigit():
                return nome.title()
    # 2. Busca “composição:” ou “princípio ativo:” na linha
    for line in lines[:40]:
        if "composição:" in line.lower() or "princípio ativo:" in line.lower():
            nome = line.split(":")[-1].strip().split()[0]
            if nome.lower() not in ignore_words:
                return nome.title()
    # 3. Busca “paracetamol”, “ibuprofeno”, “dipirona”, etc. em qualquer parte das 40 primeiras linhas
    principios_ativos = [
        "paracetamol", "dipirona", "ibuprofeno", "amoxicilina", "azitromicina",
        "cefalexina", "omeprazol", "pantoprazol", "loratadina", "sinvastatina",
        "atenolol", "losartana", "diazepam", "amoxicilina", "cloridrato", "capotril", "captopril"
    ]
    for line in lines[:40]:
        for principio in principios_ativos:
            if principio in line.lower():
                return principio.title()
    # 4. Busca primeira palavra grande em caixa alta que não seja ignorada
    for line in lines[:40]:
        for word in line.strip().split():
            if len(word) > 4 and word.isupper() and word.lower() not in ignore_words:
                return word.title()
    return "Desconhecido"



def normalize_title(title):
    title = unicodedata.normalize('NFD', title)
    title = ''.join(c for c in title if unicodedata.category(c) != 'Mn')
    title = re.sub(r'[^A-Za-zÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑçãõâêôúüéíóáà ]', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title.upper().strip()

def extract_passages_from_pdfs(pdf_folder):
    passages = []
    sources = []
    medicines = {}

    topic_keywords = [
        "INDICACAO", "INDICACOES", "INDICAÇÃO", "INDICAÇÕES",
        "POSOLOGIA", "POSOLOGÍA",
        "CONTRAINDICACAO", "CONTRAINDICACOES", "CONTRAINDICAÇÃO", "CONTRAINDICAÇÕES",
        "PRECAUCAO", "PRECAUCOES", "PRECAUÇÃO", "PRECAUÇÕES",
        "GRAVIDEZ", "GESTAÇÃO",
        "EFEITOS COLATERAIS", "REAÇÕES ADVERSAS", "REACOES ADVERSAS", "REAÇÃO ADVERSA",
        "ADVERTENCIA", "ADVERTENCIAS", "ADVERTÊNCIA", "ADVERTÊNCIAS",
        "USO EM CRIANCAS", "USO EM CRIANÇAS", "USO PEDIATRICO", "USO PEDIÁTRICO",
        "INTERACOES MEDICAMENTOSAS", "INTERAÇÕES MEDICAMENTOSAS", "INTERAÇÃO MEDICAMENTOSA",
        "INSTRUCOES DE USO", "INSTRUÇÕES DE USO"
    ]
    irrelevant_patterns = [
        "publicação no bulário", "alteração de texto", "embalagem hospitalar",
        "notificação de alteração", "rdc", "vps", "notificação de", "n/a",
        "texto de bula", "composição: ver item", "apresentação: ver item",
        "registro ms", "data da bula"
    ]

    section_pattern = re.compile(
        r'((?:[A-ZÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑa-zçãõâêôúüéíóáà ]{4,})[:]?)(?:\n|\r|\r\n)'
    )

    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith('.pdf'):
            with pdfplumber.open(os.path.join(pdf_folder, filename)) as pdf:
                text = ''
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
                medicine = extract_medicine_name(text)
                medicines[filename] = medicine
                print(f"\n[DEBUG] Medicamento extraído: {medicine} do arquivo {filename}")

                sections = section_pattern.split(text)
                print(f"[DEBUG] Títulos detectados no PDF {filename}:")
                for j in range(1, len(sections), 2):
                    print(f"  - {sections[j].strip()}")

                for i in range(1, len(sections), 2):
                    section_title = sections[i].strip()
                    section_content = sections[i+1].strip() if (i+1) < len(sections) else ''
                    normalized_title = normalize_title(section_title)
                    content_lower = section_content.lower()
                    if any(normalized_title.startswith(t) for t in topic_keywords):
                        if len(section_content) > 50 and not any(pat in content_lower for pat in irrelevant_patterns):
                            print(f"[DEBUG] Salvando passagem: {section_title} | Início: {section_content[:80]}")
                            passages.append(f"{section_title}\n{section_content}")
                            sources.append(filename)
    # Debug final das passagens INDICAÇÕES extraídas:
    for i, (passage, source) in enumerate(zip(passages, sources)):
        if 'indica' in passage.lower():
            print(f"\n[EXTRAÇÃO-DEBUG] Trecho INDICAÇÕES do arquivo {source}:\n{passage[:500]}")
    return passages, sources, medicines
