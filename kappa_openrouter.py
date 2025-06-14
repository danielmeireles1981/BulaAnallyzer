import pandas as pd
from sklearn.metrics import cohen_kappa_score

dados = {
    'Precisão A1': [1,1,1,1,0,0,1,1,0,1],
    'Precisão A2': [1,1,1,1,0,0,0,1,0,1],
    'Completude A1': [3,5,4,2,2,1,2,5,2,4],
    'Completude A2': [4,5,5,3,3,1,3,5,2,5]
}

# Criando o DataFrame
df = pd.DataFrame(dados)

# Calculando o Kappa entre as colunas de precisão
kappa = cohen_kappa_score(df['Precisão A1'], df['Precisão A2'])

print(f"Kappa de Cohen: {kappa:.2f}")
