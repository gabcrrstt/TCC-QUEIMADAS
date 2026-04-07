import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

print("--- DESENVOLVIMENTO DO MODELO PREDITIVO ---")

# --- 1. CARREGAR O DATASET DE MODELAGEM ---
try:
    df_modelo = pd.read_csv('./analise/dados_modelagem_pvh.csv', parse_dates=['Data'])
    df_modelo.set_index('Data', inplace=True) # Definir a data como índice
    print("Arquivo 'dados_modelagem_pvh.csv' carregado com sucesso!")
except FileNotFoundError:
    print("ERRO: O arquivo 'dados_modelagem_pvh.csv' não foi encontrado.")
    print("Execute o script de preparação de dados primeiro.")
    exit()

# --- 2. PREPARAÇÃO PARA O MACHINE LEARNING ---

# Definir quem são as 'features' (X) e quem é o 'alvo' (y)
# X são as variáveis que usamos para prever
# y é o que queremos prever
features = ['media_dias_sem_chuva', 'media_precipitacao', 'media_risco_fogo']
alvo = 'qtd_focos'

X = df_modelo[features]
y = df_modelo[alvo]

# Dividir os dados em conjuntos de treino e teste
# Usaremos 80% dos dados para treinar e 20% para testar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nDados divididos em treino ({len(X_train)} dias) e teste ({len(X_test)} dias).")

# --- 3. TREINAMENTO DO MODELO ---

print("Treinando o modelo RandomForestRegressor...")
# n_estimators é o número de "árvores" na floresta. Mais árvores geralmente leva a um modelo melhor.
modelo = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# O comando .fit() é o momento em que o modelo "aprende" com os dados
modelo.fit(X_train, y_train)
print("Modelo treinado com sucesso!")

# --- 4. AVALIAÇÃO DO MODELO ---

print("\n--- AVALIAÇÃO DE PERFORMANCE ---")
# Fazer previsões com os dados de teste (que o modelo nunca viu)
previsoes = modelo.predict(X_test)

# Calcular métricas de erro
mae = metrics.mean_absolute_error(y_test, previsoes)
r2 = metrics.r2_score(y_test, previsoes)

print(f"Erro Médio Absoluto (MAE): {mae:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.2f}")
print("\nInterpretação:")
print(f"-> Em média, o modelo erra em aproximadamente {mae:.0f} focos para mais ou para menos.")
print(f"-> O valor de R² indica que aproximadamente {r2:.1%} da variação na quantidade de focos pode ser explicada pelas variáveis climáticas no modelo.")


# --- 5. VISUALIZAÇÃO DOS RESULTADOS ---

plt.figure(figsize=(12, 8))
plt.scatter(y_test, previsoes, alpha=0.5, color='royalblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', lw=2, color='red')
plt.title('Valores Reais vs. Previsões do Modelo', fontsize=16, weight='bold')
plt.xlabel('Quantidade Real de Focos', fontsize=12)
plt.ylabel('Quantidade Prevista de Focos', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('grafico_5_avaliacao_modelo.png')
print("\nGráfico de avaliação 'grafico_5_avaliacao_modelo.png' salvo.")
print("-> Neste gráfico, quanto mais próximos os pontos estiverem da linha vermelha, melhores foram as previsões.")


print("\n--- ANÁLISE DE IMPORTÂNCIA DAS CARACTERÍSTICAS ---")

# Obter a importância de cada feature do modelo treinado
importancias = modelo.feature_importances_

# Criar um DataFrame para visualização
df_importancias = pd.DataFrame({'Feature': features, 'Importancia': importancias})
df_importancias = df_importancias.sort_values(by='Importancia', ascending=False)

print("Nível de importância de cada variável para o modelo:")
print(df_importancias)

# Plotar o gráfico de importância
plt.figure(figsize=(10, 6))
sns.barplot(x='Importancia', y='Feature', data=df_importancias, palette='viridis')
plt.title('Importância de Cada Variável para a Previsão de Queimadas', fontsize=16, weight='bold')
plt.xlabel('Nível de Importância', fontsize=12)
plt.ylabel('Variável Climática', fontsize=12)
plt.tight_layout()
plt.savefig('./analise/grafico_6_importancia_features.png')
print("\nGráfico 'grafico_6_importancia_features.png' salvo.")