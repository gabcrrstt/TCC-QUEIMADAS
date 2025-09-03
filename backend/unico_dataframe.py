import pandas as pd
import numpy as np
import glob
import os

# --- 1. CARREGAMENTO E CONSOLIDAÇÃO DOS DADOS ---

# Garanta que o script está na mesma pasta dos seus arquivos CSV
# ou ajuste o caminho.
caminho_pasta = './data/'
todos_arquivos = glob.glob(os.path.join(caminho_pasta, "focos_qmd_inpe*.csv"))

if not todos_arquivos:
    print("Nenhum arquivo CSV encontrado! Verifique o caminho da pasta.")
else:
    print(f"Encontrados {len(todos_arquivos)} arquivos. Carregando...")
    lista_dfs = [pd.read_csv(f) for f in todos_arquivos]
    df_completo = pd.concat(lista_dfs, ignore_index=True)
    print("Dados de todos os anos carregados com sucesso!")
    print(f"Total de registros: {len(df_completo)}")
    print("\nAmostra dos dados carregados:")
    print(df_completo.head())

    # --- 2. LIMPEZA E PRÉ-PROCESSAMENTO ---
    print("\nIniciando limpeza e pré-processamento...")
    df_completo['DataHora'] = pd.to_datetime(df_completo['DataHora'])
    df_completo['Ano'] = df_completo['DataHora'].dt.year
    df_completo['Mes'] = df_completo['DataHora'].dt.month
    df_completo['RiscoFogo'] = df_completo['RiscoFogo'].replace(-999.0, np.nan)
    print("Pré-processamento concluído.")

    # --- 3. SALVANDO O DATAFRAME CONSOLIDADO ---
    # Esta é a parte nova e mais importante!
    nome_arquivo_saida = 'dados_consolidados_2015_2024.csv'
    df_completo.to_csv(nome_arquivo_saida, index=False) # index=False evita salvar uma coluna de índice desnecessária

    print(f"\n✅ SUCESSO! O dataframe completo foi salvo no arquivo: '{nome_arquivo_saida}'")
    print("\nAgora, para as próximas análises, você pode carregar diretamente este único arquivo.")