import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

print("\n--- MODELO FINAL DE PREVISÃO DE QUEIMADAS ---")

# ==============================
# 1. CARREGAR DADOS
# ==============================
df = pd.read_csv('backend/data/modelagem/dados_modelagem_pvh.csv', parse_dates=['Data'])

df = df.sort_values('Data')  # importante para série temporal

print(f"Dataset carregado com {len(df)} registros")

# ==============================
# 2. DEFINIR FEATURES E ALVO
# ==============================

features = [
    'media_dias_sem_chuva',
    'media_precipitacao',
    'media_risco_fogo',
    'media_frp',
    'Estacao_Seca',
    'lag_1',
    'lag_2'
]

X = df[features]
y = df['qtd_focos']

# ==============================
# 3. DIVISÃO TREINO/TESTE (TEMPORAL)
# ==============================

split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

# ==============================
# 4. GRID SEARCH (OTIMIZAÇÃo)
# ==============================

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

print("\n Otimizando modelo...")
grid.fit(X_train, y_train)

modelo = grid.best_estimator_

print("Melhores parâmetros:")
print(grid.best_params_)

# ==============================
# 5. VALIDAÇÃO CRUZADA
# ==============================

scores = cross_val_score(modelo, X, y, cv=5, scoring='r2')
print(f"\nR² médio (validação cruzada): {scores.mean():.3f}")

# ==============================
# 6. AVALIAÇÃO FINAL
# ==============================

previsoes = modelo.predict(X_test)

mae = metrics.mean_absolute_error(y_test, previsoes)
r2 = metrics.r2_score(y_test, previsoes)

print("\n--- RESULTADOS ---")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.3f}")

# ==============================
# 7. GRÁFICO REAL vs PREVISTO
# ==============================

plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label="Real")
plt.plot(previsoes, label="Previsto")
plt.legend()
plt.title("Comparação: Real vs Previsto")
plt.xlabel("Tempo")
plt.ylabel("Focos de Queimada")
plt.grid()
plt.tight_layout()
plt.savefig('backend/imagens/modelo_final.png')

print(" Gráfico salvo")

# ==============================
# 8. IMPORTÂNCIA DAS VARIÁVEIS
# ==============================

importancias = modelo.feature_importances_

df_importancia = pd.DataFrame({
    'Feature': features,
    'Importancia': importancias
}).sort_values(by='Importancia', ascending=False)

print("\nImportância das variáveis:")
print(df_importancia)


# depois disso:
previsoes = modelo.predict(X_test)

# ==============================
# GRÁFICO DE ERRO
# ==============================

erro = y_test - previsoes

plt.figure(figsize=(10,5))
plt.hist(erro, bins=30)
plt.title("Distribuição dos Erros do Modelo")
plt.xlabel("Erro (Real - Previsto)")
plt.ylabel("Frequência")
plt.grid()
plt.tight_layout()

plt.savefig('backend/imagens/erro_modelo.png')
plt.show()

# GRAFICO 1. EVOLUÇÃO AO LONGO DO TEMPO (PORTO VELHO)

plt.figure(figsize=(12,6))

plt.plot(df['Data'], df['qtd_focos'])

plt.title("Evolução dos Focos de Queimadas em Porto Velho")
plt.xlabel("Data")
plt.ylabel("Quantidade de Focos")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()

plt.savefig('backend/imagens/evolucao_pvh.png')
plt.show()

# GRAFICO 2. SAZONIDADE POR MES 
df['Mes'] = df['Data'].dt.month

media_mes = df.groupby('Mes')['qtd_focos'].mean()

plt.figure(figsize=(10,5))

plt.plot(media_mes.index, media_mes.values, marker='o')

plt.title("Média de Focos por Mês (Porto Velho)")
plt.xlabel("Mês")
plt.ylabel("Quantidade Média de Focos")
plt.grid()
plt.tight_layout()

plt.savefig('backend/imagens/sazonalidade_pvh.png')
plt.show()

#  GRAFICO 3: DIA COM MAIS QUEIMADAS
top_dias = df.sort_values(by='qtd_focos', ascending=False).head(10)

plt.figure(figsize=(10,5))

plt.bar(range(len(top_dias)), top_dias['qtd_focos'])

plt.title("Top 10 Dias com Mais Queimadas")
plt.xlabel("Ranking")
plt.ylabel("Quantidade de Focos")
plt.grid()
plt.tight_layout()

plt.savefig('backend/imagens/top_dias_pvh.png')
plt.show()

# REAL VS PREVISTO AO LONGO DO TEMPO (PORTO VELHO)
plt.figure(figsize=(10,5))

plt.plot(y_test.values, label="Real")
plt.plot(previsoes, label="Previsto")

plt.title("Real vs Previsto")
plt.xlabel("Tempo")
plt.ylabel("Focos")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig('backend/imagens/real_vs_previsto.png')
plt.show()

# ==============================
# GRÁFICO 2: PREVISÃO AO LONGO DO TEMPO
# ==============================

plt.figure(figsize=(12,6))

plt.plot(df['Data'][split:], y_test, label="Real")
plt.plot(df['Data'][split:], previsoes, label="Previsto")

plt.title("Previsão de Focos de Queimadas ao Longo do Tempo")
plt.xlabel("Data")
plt.ylabel("Quantidade de Focos")
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig('backend/imagens/previsao_tempo.png')
plt.show()


# ==============================
# GRÁFICO 3: IMPORTÂNCIA DAS VARIÁVEIS
# ==============================

plt.figure(figsize=(10,6))

plt.barh(df_importancia['Feature'], df_importancia['Importancia'])

plt.title("Importância das Variáveis no Modelo")
plt.xlabel("Importância")
plt.ylabel("Variáveis")
plt.gca().invert_yaxis()
plt.grid()

plt.tight_layout()
plt.savefig('backend/imagens/importancia_variaveis.png')
plt.show()