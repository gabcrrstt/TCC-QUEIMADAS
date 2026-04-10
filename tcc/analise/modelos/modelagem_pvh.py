import pandas as pd

print("\n--- CRIANDO DATASET DE MODELAGEM ---")

df = pd.read_csv('dados_consolidados_2015_2025.csv')

# ==============================
# 1. GARANTIR DATA
# ==============================
df['DataHora'] = pd.to_datetime(df['DataHora'])
df['Data'] = df['DataHora'].dt.date

# ==============================
# 2. FILTRAR LOCAL (IMPORTANTE)
# ==============================
municipio_alvo = "PORTO VELHO"
df = df[df['Municipio'] == municipio_alvo]

print(f"Filtrando dados para: {municipio_alvo}")

# ==============================
# 3. AGREGAÇÃO DIÁRIA 
# ==============================

df_modelo = df.groupby('Data').agg(
    qtd_focos=('Data', 'count'),
    media_dias_sem_chuva=('DiaSemChuva', 'mean'),
    media_precipitacao=('Precipitacao', 'mean'),
    media_risco_fogo=('RiscoFogo', 'mean'),
    media_frp=('FRP', 'mean') 
).reset_index()

print("✔ Dados agregados por dia")

# ==============================
# 4. CRIAR VARIÁVEIS TEMPORAIS
# ==============================

df_modelo['Data'] = pd.to_datetime(df_modelo['Data'])

df_modelo['Mes'] = df_modelo['Data'].dt.month
df_modelo['DiaSemana'] = df_modelo['Data'].dt.dayofweek

# estação seca (você já usava)
df_modelo['Estacao_Seca'] = df_modelo['Mes'].apply(
    lambda x: 1 if x in [7,8,9,10] else 0
)

# ==============================
# 5. LAGS 
# ==============================

df_modelo['lag_1'] = df_modelo['qtd_focos'].shift(1)
df_modelo['lag_2'] = df_modelo['qtd_focos'].shift(2)

# remover NaN dos lags
df_modelo = df_modelo.dropna()

# ==============================
# 6. SALVAR DATASET FINAL
# ==============================

df_modelo.to_csv('backend\data\modelagem\dados_modelagem_pvh.csv', index=False)

print("\n DATASET DE MODELAGEM PRONTO!")
print(df_modelo.head())
