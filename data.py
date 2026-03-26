import pandas as pd

df = pd.read_csv('saude\\2 - Saude\\dataset_saude_brasil.csv')
df.head()
df0 = df.copy()
target = 'Risco_Doenca'
df = df0.copy()
df['Passos_Diarios'] = pd.to_numeric(df['Passos_Diarios'], errors='coerce')
df['Calorias'] = pd.to_numeric(df['Calorias'], errors='coerce')
df['Colesterol'] = pd.to_numeric(df['Colesterol'], errors='coerce')
transform_sexo = {'Masculino': 0, 'Feminino': 1}
transform_fumante = {'Não': 0, 'Sim': 1}
transform_alcool = {'Baixo': 0, 'Moderado': 1, 'Alto':2} #ordem
transform_historico_familiar = {'Não': 0, 'Sim': 1}
transform_risco_doenca = {'Baixo': 1, 'Moderado': 2, 'Alto': 6, 'Muito Alto': 24} #ordem
df['Sexo'] = df['Sexo'].map(transform_sexo)
df['Fumante'] = df['Fumante'].map(transform_fumante)
df['Alcool'] = df['Alcool'].map(transform_alcool)
df['Historico_Familiar'] = df['Historico_Familiar'].map(transform_historico_familiar)
df['Risco_Doenca'] = df['Risco_Doenca'].map(transform_risco_doenca)

print(df.isnull().sum())
#preenchimento para o isolation forest 
df['Idade'] = df['Idade'].fillna(df['Idade'].mean())
df['IMC'] = df['IMC'].fillna(df['IMC'].mean())
df['Passos_Diarios'] = df['Passos_Diarios'].fillna(df['Passos_Diarios'].mean())
df['Calorias'] = df['Calorias'].fillna(df['Calorias'].mean())
df['Colesterol'] = df['Colesterol'].fillna(df['Colesterol'].mean())
print(df.isnull().sum())

df.to_csv('data_processed.csv', index=False)
