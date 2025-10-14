import pandas as pd
import numpy as np

print("--- PREPARAÇÃO DOS DADOS PARA MODELAGEM PREDITIVA ---")

# --- 1. CARREGAR O DATASET CONSOLIDADO ---
try:
    df = pd.read_csv('dados_consolidados_2015_2024.csv', parse_dates=['DataHora'])
    print("Arquivo 'dados_consolidados_2015_2024.csv' carregado com sucesso!")
except FileNotFoundError:
    print("ERRO: O arquivo 'dados_consolidados_2015_2024.csv' não foi encontrado.")
    exit()

# --- 2. FILTRAR DADOS PARA UM MUNICÍPIO ESPECÍFICO ---
municipio_alvo = 'PORTO VELHO'
df_pvh = df[df['Municipio'] == municipio_alvo].copy()

if df_pvh.empty:
    print(f"ERRO: Nenhum dado encontrado para o município '{municipio_alvo}'. Verifique o nome.")
    exit()

print(f"Dados filtrados para o município de {municipio_alvo}. Total de {len(df_pvh)} registros.")

# --- 3. AGREGAR DADOS POR DIA ---
# Vamos criar uma coluna 'Data' apenas com a data (sem as horas)
df_pvh['Data'] = df_pvh['DataHora'].dt.date

# Agora, agrupamos por essa nova coluna 'Data'
# Usamos .agg para aplicar diferentes funções a cada coluna
print("Agregando dados por dia...")
dados_diarios = df_pvh.groupby('Data').agg(
    qtd_focos=('Data', 'count'),  # Conta o número de ocorrências (focos) no dia
    media_dias_sem_chuva=('DiaSemChuva', 'mean'),
    media_precipitacao=('Precipitacao', 'mean'),
    media_risco_fogo=('RiscoFogo', 'mean')
).reset_index()

# O 'RiscoFogo' pode ter valores NaN (Not a Number) após a média, se todos os valores do dia eram nulos.
# Uma estratégia simples é preencher esses NaNs com 0 ou com a média geral da coluna.
dados_diarios['media_risco_fogo'].fillna(0, inplace=True)


# --- 4. SALVAR O NOVO DATASET PRONTO PARA MODELAGEM ---
nome_arquivo_saida = 'dados_modelagem_pvh.csv'
dados_diarios.to_csv(nome_arquivo_saida, index=False)

print(f"\n✅ SUCESSO! O novo dataset para modelagem foi salvo como: '{nome_arquivo_saida}'")
print("\nAmostra do novo dataset:")
print(dados_diarios.head())
print(f"\nTotal de dias (linhas) no novo dataset: {len(dados_diarios)}")