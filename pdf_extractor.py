import re
import os
import csv
import pdfplumber

principios_ativos = [
    "paracetamol", "dipirona", "ibuprofeno", "amoxicilina", "azitromicina", "cefalexina",
    "omeprazol", "pantoprazol", "ranitidina", "loratadina", "sinvastatina", "atenolol", "losartana",
    "diazepam", "captopril", "metformina", "clonazepam", "hidroclorotiazida", "atorvastatina",
    "lorazepam", "prednisona", "dexametasona", "budesonida", "risperidona", "nimesulida",
    "cetoprofeno", "carbamazepina", "sulfametoxazol"
    # Adicione outros conforme sua base!
]

ignore_words = {
    "comprimido", "mg", "ml", "solução", "genérico", "farmacêutica", "sa", "ltda",
    "identificação", "do", "medicamento", "apresentação", "bula", "medicamento genérico",
    "uso", "oral", "apresentações", "comprimidos", "modelo", "receberam", "legrand",
    "ems", "medley", "teuto", "zambon", "prati-donaduzzi", "sandoz", "sanofi", "ache",
    "novaquimica", "neo", "quimica", "hipolabor", "brainfarma", "farmacêutica", "indústria"
}

# Novos padrões para título de medicamento ou princípio ativo
key_patterns = [
    r"nome do medicamento[:\-]?\s*(.+)",
    r"nome comercial[:\-]?\s*(.+)",
    r"produto[:\-]?\s*(.+)",
    r"princípio ativo[:\-]?\s*(.+)",
    r"composição[:\-]?\s*(.+)",
    r"substância ativa[:\-]?\s*(.+)"
]

def clean_name(raw):
    """Remove números, sinais e ignora nomes genéricos/laboratoriais."""
    nome = raw.strip().split()[0]
    nome = re.sub(r'[^A-Za-zÇçÁÉÍÓÚÃÕÂÊÔÜáéíóúãõâêôü]', '', nome)
    nome = nome.lower()
    if nome in ignore_words or len(nome) < 4:
        return None
    return nome.title()

def extract_medicine_name(text, filename=None):
    lines = text.strip().split('\n')
    lines = [l for l in lines if l.strip()]  # remove linhas vazias
    max_lines = min(80, len(lines))

    # 1. Busca explícita por padrões com palavras-chave
    for idx in range(max_lines):
        line = lines[idx].lower()
        for pat in key_patterns:
            match = re.search(pat, line)
            if match:
                possible = match.group(1)
                cleaned = clean_name(possible)
                if cleaned:
                    return cleaned

    # 2. Busca por princípio ativo conhecido
    for idx in range(max_lines):
        line = lines[idx].lower()
        for principio in principios_ativos:
            if re.search(rf"\b{principio}\b", line):
                return principio.title()

    # 3. Busca padrão “XXX 500 mg”, “XXX comprimido” etc.
    for idx in range(max_lines):
        match = re.search(
            r'\b([A-ZÇÃÕÉÊÁÍÓÚÂÔÜa-zçãõâêôúüéíóà]{4,})\b[\s\-]*(comprimido|capsula|mg|ml|solução)',
            lines[idx], re.IGNORECASE)
        if match:
            possible = match.group(1)
            cleaned = clean_name(possible)
            if cleaned:
                return cleaned

    # 4. Primeira palavra em caixa alta significativa
    for idx in range(max_lines):
        for word in lines[idx].split():
            if len(word) > 4 and word.isupper():
                cleaned = clean_name(word)
                if cleaned:
                    return cleaned

    # 5. Busca pelo nome do arquivo (se fornecido)
    if filename:
        # Remove extensões, pontuação e números
        name_raw = os.path.splitext(os.path.basename(filename))[0]
        name_candidate = re.split(r'[_\-\s]', name_raw)[0]
        name_candidate = re.sub(r'\d+', '', name_candidate)
        cleaned = clean_name(name_candidate)
        if cleaned:
            return cleaned

    # 6. Se nada funcionar, retorna "Desconhecido"
    return "Desconhecido"

def extract_passages_from_pdfs(pdf_folder):
    passages = []
    sources = []
    medicines = {}

    log_path = "medicamentos_log.csv"
    write_header = not os.path.exists(log_path)
    log_file = open(log_path, "a", newline='', encoding="utf-8")
    log_writer = csv.writer(log_file)
    if write_header:
        log_writer.writerow(["Arquivo", "Nome Extraído", "Linha relevante (exemplo)"])

    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith('.pdf'):
            with pdfplumber.open(os.path.join(pdf_folder, filename)) as pdf:
                text = ''
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
                # >>> Passe filename para permitir fallback pelo nome do arquivo!
                medicine = extract_medicine_name(text, filename=filename)
                medicines[filename] = medicine

                if medicine in {"Desconhecido", "Composição", "Medicamento", "Modelo"} or len(medicine) < 4:
                    for l in text.splitlines():
                        if l.strip():
                            linha_relevante = l.strip()
                            break
                    else:
                        linha_relevante = ""
                    log_writer.writerow([filename, medicine, linha_relevante])

                sections = re.split(r'([A-ZÇÃÕÉÊÁÍÓÚÂÔÜa-zçãõâêôúüéíóà\s\-]{4,})\n', text)
                for i in range(1, len(sections) - 1, 2):
                    section_title = sections[i].strip()
                    section_content = sections[i + 1].strip()
                    if len(section_content) > 50:
                        passages.append(section_title + "\n" + section_content)
                        sources.append(filename)
    log_file.close()
    return passages, sources, medicines