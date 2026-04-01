import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# carregar dados
df = pd.read_csv('backend/data/modelagem/dados_modelagem_pvh.csv', parse_dates=['Data'])
df = df.sort_values('Data')

# features
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

# divisão temporal
split = int(len(df) * 0.8)

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# modelo simples (sem grid, só pra gerar gráfico)
modelo = RandomForestRegressor(random_state=42)
modelo.fit(X_train, y_train)

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

plt.savefig('resultados/graficos/erro_modelo.png')
plt.show()