import pandas as pd
import numpy as np

print("\n--- ETAPA DE LIMPEZA DOS DADOS ---")

# ==============================
# 1. REMOVER DUPLICATAS
# ==============================
df = pd.read_csv('dados_consolidados_2015_2025.csv')
print(f"Total de registros antes da limpeza: {len(df)}")
df = df.drop_duplicates()
print(f"✔ Duplicatas removidas. Total atual: {len(df)}")

# ==============================
# 2. TRATAMENTO DE DATAS
# ==============================
df['DataHora'] = pd.to_datetime(df['DataHora'], errors='coerce')

# Remover datas inválidas
df = df.dropna(subset=['DataHora'])
print("✔ Datas tratadas")

# ==============================
# 3. TRATAMENTO DE VALORES AUSENTES
# ==============================

# Substituir valores inválidos
if 'RiscoFogo' in df.columns:
    df['RiscoFogo'] = df['RiscoFogo'].replace(-999.0, np.nan)

# Preencher NaN
df['RiscoFogo'] = df['RiscoFogo'].fillna(0)

if 'Precipitacao' in df.columns:
    df['Precipitacao'] = df['Precipitacao'].fillna(0)

if 'DiaSemChuva' in df.columns:
    df['DiaSemChuva'] = df['DiaSemChuva'].fillna(0)

print("✔ Valores ausentes tratados")

# ==============================
# 4. REMOVER VALORES INVÁLIDOS
# ==============================

# Exemplo: valores negativos que não fazem sentido
df = df[df['Precipitacao'] >= 0]
df = df[df['DiaSemChuva'] >= 0]

print("✔ Valores inválidos removidos")

# ==============================
# 5. PADRONIZAÇÃO DE TEXTO
# ==============================

if 'Municipio' in df.columns:
    df['Municipio'] = df['Municipio'].str.upper().str.strip()

if 'Estado' in df.columns:
    df['Estado'] = df['Estado'].str.upper().str.strip()

print("✔ Texto padronizado")

# ==============================
# 6. CRIAÇÃO DE VARIÁVEIS TEMPORAIS 
# ==============================

df['Ano'] = df['DataHora'].dt.year
df['Mes'] = df['DataHora'].dt.month
df['Dia'] = df['DataHora'].dt.day
df['DiaSemana'] = df['DataHora'].dt.dayofweek

# Estação seca (muito importante pro seu TCC)
df['Estacao_Seca'] = df['Mes'].apply(lambda x: 1 if x in [7,8,9,10] else 0)

print("✔ Variáveis temporais criadas")

# ==============================
# 7. RESUMO FINAL
# ==============================

print("\n📊 RESUMO DOS DADOS LIMPOS:")
print(df.info())
print(df.describe())

print("\n✅ LIMPEZA FINALIZADA COM SUCESSO!")