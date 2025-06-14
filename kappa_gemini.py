import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Dados copiados do seu trecho
dados = {
    'Precisão A1': [0,1,1,1,1,0,1,1,0,1],
    'Precisão A2': [0,1,1,1,1,0,0,1,1,1],
    'Completude A1': [3,5,3,4,4,1,2,4,3,5],
    'Completude A2': [2,4,4,5,4,1,3,5,3,4]
}
df = pd.DataFrame(dados)

# Calculando o Kappa
kappa = cohen_kappa_score(df['Precisão A1'], df['Precisão A2'])
print(f'Kappa: {kappa:.2f}')
