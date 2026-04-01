import pandas as pd
import numpy as np
import glob
import os

print("\n--- ETAPA 1: CONSOLIDAÇÃO E PRÉ-PROCESSAMENTO ---")

# ==============================
# 1. DEFINIR CAMINHO (ROBUSTO)
# ==============================
caminho_pasta = os.path.join("backend", "data")
arquivos = glob.glob(os.path.join(caminho_pasta, "focos_qmd_inpe*.csv"))

if not arquivos:
    print("❌ Nenhum arquivo CSV encontrado!")
    exit()

print(f"📂 {len(arquivos)} arquivos encontrados. Carregando...")

# ==============================
# 2. CARREGAMENTO COM SEGURANÇA
# ==============================

lista_dfs = []

for arquivo in arquivos:
    try:
        df = pd.read_csv(arquivo)

        # Verificar colunas essenciais
        if "DataHora" not in df.columns:
            print(f"⚠️ Arquivo ignorado (sem DataHora): {arquivo}")
            continue

        lista_dfs.append(df)

    except Exception as e:
        print(f"⚠️ Erro ao ler {arquivo}: {e}")

if not lista_dfs:
    print("❌ Nenhum arquivo válido carregado.")
    exit()

df = pd.concat(lista_dfs, ignore_index=True)

print(f"✅ Total de registros: {len(df)}")

# ==============================
# 3. PRÉ-PROCESSAMENTO
# ==============================

print("\n🔧 Limpando e preparando dados...")

# Converter datas
df['DataHora'] = pd.to_datetime(df['DataHora'], errors='coerce')

# Remover linhas inválidas
df = df.dropna(subset=['DataHora'])

# Criar variáveis temporais
df['Ano'] = df['DataHora'].dt.year
df['Mes'] = df['DataHora'].dt.month
df['Dia'] = df['DataHora'].dt.day
df['DiaSemana'] = df['DataHora'].dt.dayofweek

# Criar variável de estação seca 
#1 → está na estação seca
#0 → não está na estação seca
df['Estacao_Seca'] = df['Mes'].apply(lambda x: 1 if x in [7,8,9,10] else 0)

# Corrigir valores inválidos
if 'RiscoFogo' in df.columns:
    df['RiscoFogo'] = df['RiscoFogo'].replace(-999.0, np.nan)

print("✅ Pré-processamento concluído.")

# ==============================
# 4. ANÁLISE RÁPIDA (BÔNUS 🔥)
# ==============================

print("\n📊 Estatísticas básicas:")
print(df.describe())

# ==============================
# 5. SALVAR DATASET FINAL
# ==============================

nome_saida = "dados_consolidados_2015_2025.csv"
df.to_csv(nome_saida, index=False)

print(f"\n✅ Dataset salvo como: {nome_saida}")
print("🚀 Pronto para modelagem!")