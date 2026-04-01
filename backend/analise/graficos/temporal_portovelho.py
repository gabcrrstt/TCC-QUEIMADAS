import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ATENÇÃO: Usei o nome do arquivo que apareceu no seu terminal.
# Se for diferente, apenas ajuste a linha abaixo.
ARQUIVO_CONSOLIDADO = 'dados_consolidados_2015_2024.csv'
MUNICIPIO_ALVO = 'PORTO VELHO'

print(f"Gerando a série temporal de focos de queimada para {MUNICIPIO_ALVO}...")

try:
    # parse_dates=['DataHora'] já converte a coluna para o formato de data durante a leitura
    df = pd.read_csv(ARQUIVO_CONSOLIDADO, parse_dates=['DataHora'])
except FileNotFoundError:
    print(f"ERRO: Arquivo '{ARQUIVO_CONSOLIDADO}' não encontrado.")
    exit()

# Filtra para o município e define a coluna de data como o índice do DataFrame
df_municipio = df[df['Municipio'] == MUNICIPIO_ALVO].set_index('DataHora')

# Reamostra os dados por dia ('D') e conta o número de ocorrências (focos) em cada dia.
focos_diarios = df_municipio.resample('D').size()

print("Criando o gráfico...")
plt.figure(figsize=(20, 8))
sns.set_theme(style="whitegrid")

# Plota a série temporal
focos_diarios.plot(color='firebrick', linewidth=1.2)

plt.title(f'Série Temporal de Focos Diários de Queimada em {MUNICIPIO_ALVO}', fontsize=20, weight='bold')
plt.xlabel('Ano', fontsize=14)
plt.ylabel('Número de Focos Diários', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Opcional: Anotar o pico máximo em todo o período
if not focos_diarios.empty:
    pico_data = focos_diarios.idxmax()
    pico_valor = focos_diarios.max()
    plt.annotate(f'Pico máximo: {pico_valor} focos\nem {pico_data.strftime("%d/%m/%Y")}',
                 xy=(pico_data, pico_valor),
                 xytext=(pico_data, pico_valor + 50), # Posição do texto
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1, alpha=0.8))

plt.tight_layout()
plt.savefig('grafico_final_serie_temporal.png')
print("\nGráfico 'grafico_final_serie_temporal.png' salvo com sucesso!")
print("\nCom isso, você está oficialmente pronto para começar a construir seu dashboard. Parabéns!")