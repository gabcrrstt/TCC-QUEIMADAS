# Script: passo2_analisar_dados.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("\n--- PASSO 2: Iniciando a análise e geração de gráficos ---")
sns.set_theme(style="whitegrid")

try:
    df = pd.read_csv('dados_consolidados_2015_2024.csv')
    print("Arquivo 'dados_consolidados_2015_2024.csv' carregado com sucesso!")
except FileNotFoundError:
    print("\nERRO CRÍTICO: O arquivo consolidado ainda não foi encontrado.")
    print("Por favor, execute o script 'passo1_unificar_dados.py' primeiro.")
    exit()

# O resto do código é o mesmo...
df['DataHora'] = pd.to_datetime(df['DataHora'])
df['Ano'] = df['DataHora'].dt.year
df['Mes'] = df['DataHora'].dt.month

# --- Gerando Gráfico 1 ---
print("Gerando gráfico 1: Focos por Ano...")
focos_por_ano = df.groupby('Ano').size()
plt.figure(figsize=(12, 7))
focos_por_ano.plot(kind='line', marker='o', linestyle='-', color='firebrick')
plt.title('Evolução Anual dos Focos de Queimada em Rondônia', fontsize=16, weight='bold')
plt.xlabel('Ano', fontsize=12)
plt.ylabel('Número Total de Focos', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(focos_por_ano.index.astype(int))
plt.tight_layout()
plt.savefig('grafico_1_focos_por_ano.png')
print("-> Gráfico 'grafico_1_focos_por_ano.png' salvo.")

# --- Gerando Gráfico 2 ---
print("Gerando gráfico 2: Sazonalidade Mensal...")
focos_por_mes = df.groupby('Mes').size()
nomes_meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
plt.figure(figsize=(12, 7))
sns.barplot(x=focos_por_mes.index, y=focos_por_mes.values, palette='YlOrRd')
plt.title('Sazonalidade das Queimadas em Rondônia (Total Acumulado)', fontsize=16, weight='bold')
plt.xlabel('Mês', fontsize=12)
plt.ylabel('Número Total de Focos', fontsize=12)
plt.xticks(ticks=range(12), labels=nomes_meses)
plt.tight_layout()
plt.savefig('grafico_2_sazonalidade_mensal.png')
print("-> Gráfico 'grafico_2_sazonalidade_mensal.png' salvo.")

# --- Gerando Gráfico 3 ---
print("Gerando gráfico 3: Top 15 Municípios...")
top_15_municipios = df['Municipio'].value_counts().nlargest(15)
plt.figure(figsize=(12, 9))
sns.barplot(x=top_15_municipios.values, y=top_15_municipios.index, orient='h', palette='hot_r')
plt.title('Top 15 Municípios com Mais Focos de Queimada (Total Acumulado)', fontsize=16, weight='bold')
plt.xlabel('Número Total de Focos', fontsize=12)
plt.ylabel('Município', fontsize=12)
plt.tight_layout()
plt.savefig('grafico_3_top_15_municipios.png')
print("-> Gráfico 'grafico_3_top_15_municipios.png' salvo.")

print("\n✅ ANÁLISE CONCLUÍDA! Verifique os arquivos de imagem .png na sua pasta.")