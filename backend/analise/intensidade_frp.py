import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Use o seu arquivo consolidado com todos os anos
ARQUIVO_CONSOLIDADO = 'dados_consolidados_2015_2024.csv'

print(f"Iniciando análise de intensidade das queimadas (FRP)...")

try:
    df = pd.read_csv(ARQUIVO_CONSOLIDADO)
except FileNotFoundError:
    print(f"ERRO: Arquivo '{ARQUIVO_CONSOLIDADO}' não encontrado.")
    exit()

# --- PREPARAÇÃO DOS DADOS DE FRP ---

# A coluna FRP tem muitos valores ausentes (NaNs). Vamos removê-los para a análise.
# É importante mencionar esta remoção na sua monografia como uma limitação da análise.
df_frp = df.dropna(subset=['FRP'])
print(f"Registros totais: {len(df)}. Registros com dados de FRP: {len(df_frp)} ({len(df_frp)/len(df):.2%}).")

# --- ANÁLISE POR MUNICÍPIO ---

# Primeiro, vamos encontrar os 15 municípios com o maior número de focos COM medição de FRP
# para garantir que nossa média seja representativa.
top_municipios_contagem = df_frp['Municipio'].value_counts().nlargest(15).index

# Agora, filtramos o DataFrame para conter apenas esses municípios
df_top_municipios = df_frp[df_frp['Municipio'].isin(top_municipios_contagem)]

# Calculamos a média de FRP para cada um desses municípios
media_frp_municipio = df_top_municipios.groupby('Municipio')['FRP'].mean().sort_values(ascending=False)

print("\nTop 15 Municípios por Média de Intensidade do Fogo (FRP):")
print(media_frp_municipio)

# --- VISUALIZAÇÃO ---
plt.figure(figsize=(12, 9))
sns.set_theme(style="whitegrid")
sns.barplot(x=media_frp_municipio.values, y=media_frp_municipio.index, palette='autumn', hue=media_frp_municipio.index, legend=False)

plt.title('Média de Intensidade das Queimadas (FRP) por Município', fontsize=18, weight='bold')
plt.xlabel('Média de Fire Radiative Power (FRP)', fontsize=12)
plt.ylabel('Município', fontsize=12)
plt.axvline(x=media_frp_municipio.mean(), color='black', linestyle='--', label=f'Média Geral ({media_frp_municipio.mean():.2f})')
plt.legend()

plt.tight_layout()
plt.savefig('grafico_analise_frp_intensidade.png')

print("\nGráfico 'grafico_analise_frp_intensidade.png' salvo com sucesso.")