import pandas as pd
import folium
from folium.plugins import HeatMap

# Use o seu arquivo consolidado com todos os anos
ARQUIVO_CONSOLIDADO = 'dados_consolidados_2015_2024.csv'
MUNICIPIO_ALVO = 'PORTO VELHO'

print(f"Iniciando a geração do mapa de calor para {MUNICIPIO_ALVO}...")

try:
    df = pd.read_csv(ARQUIVO_CONSOLIDADO)
except FileNotFoundError:
    print(f"ERRO: Arquivo '{ARQUIVO_CONSOLIDADO}' não encontrado.")
    exit()

# --- PREPARAÇÃO DOS DADOS GEOGRÁFICOS ---

# Filtra para o município alvo
df_municipio = df[df['Municipio'] == MUNICIPIO_ALVO].copy()

# Remove quaisquer registros sem dados válidos de latitude ou longitude
df_municipio.dropna(subset=['Latitude', 'Longitude'], inplace=True)
print(f"Total de focos de queimada em {MUNICIPIO_ALVO} a serem mapeados: {len(df_municipio)}")

# --- CRIAÇÃO DO MAPA ---

# 1. Encontrar o ponto central do mapa (a média das coordenadas)
lat_media = df_municipio['Latitude'].mean()
lon_media = df_municipio['Longitude'].mean()

# 2. Criar o mapa base, centrado na nossa área de interesse
# O 'zoom_start' controla o nível de zoom inicial
mapa_calor = folium.Map(location=[lat_media, lon_media], zoom_start=8)

# 3. Preparar os dados para o plugin HeatMap
# O plugin precisa de uma lista de listas, onde cada sublista é [latitude, longitude]
dados_heatmap = df_municipio[['Latitude', 'Longitude']].values.tolist()

# 4. Adicionar a camada de calor ao mapa
HeatMap(dados_heatmap, radius=15).add_to(mapa_calor)

# 5. Salvar o mapa como um arquivo HTML
NOME_ARQUIVO_MAPA = 'mapa_calor_porto_velho.html'
mapa_calor.save(NOME_ARQUIVO_MAPA)

print(f"\n✅ Sucesso! O mapa de calor interativo foi salvo como '{NOME_ARQUIVO_MAPA}'.")
print("Abra este arquivo em um navegador de internet para explorar o mapa.")