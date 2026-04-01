import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

print("\n--- PIPELINE COMPLETO DE QUEIMADAS ---")

# ==============================
# 1. CONSOLIDAÇÃO
# ==============================
arquivos = glob.glob(os.path.join("dados/brutos", "*.csv"))

dfs = []
for arq in arquivos:
    try:
        df_temp = pd.read_csv(arq)
        dfs.append(df_temp)
    except:
        print(f"Erro ao ler {arq}")

df = pd.concat(dfs, ignore_index=True)

print(f"Dados carregados: {len(df)}")

# ==============================
# 2. LIMPEZA
# ==============================
df = df.drop_duplicates()

df['DataHora'] = pd.to_datetime(df['DataHora'], errors='coerce')
df = df.dropna(subset=['DataHora'])

df['RiscoFogo'] = df['RiscoFogo'].replace(-999, np.nan)
df['RiscoFogo'] = df['RiscoFogo'].fillna(0)

# ==============================
# 3. FILTRAR PORTO VELHO 🔥
# ==============================
df['Municipio'] = df['Municipio'].str.upper().str.strip()
df = df[df['Municipio'] == "PORTO VELHO"]

# ==============================
# 4. AGREGAÇÃO
# ==============================
df['Data'] = df['DataHora'].dt.date

df_modelo = df.groupby('Data').agg(
    qtd_focos=('Data', 'count'),
    media_dias_sem_chuva=('DiaSemChuva', 'mean'),
    media_precipitacao=('Precipitacao', 'mean'),
    media_risco_fogo=('RiscoFogo', 'mean'),
    media_frp=('FRP', 'mean')
).reset_index()

# ==============================
# 5. FEATURES
# ==============================
df_modelo['Data'] = pd.to_datetime(df_modelo['Data'])

df_modelo['Mes'] = df_modelo['Data'].dt.month
df_modelo['Estacao_Seca'] = df_modelo['Mes'].apply(
    lambda x: 1 if x in [7,8,9,10] else 0
)

df_modelo['lag_1'] = df_modelo['qtd_focos'].shift(1)
df_modelo['lag_2'] = df_modelo['qtd_focos'].shift(2)

df_modelo = df_modelo.dropna()

# ==============================
# 6. MODELO
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

X = df_modelo[features]
y = df_modelo['qtd_focos']

split = int(len(df_modelo) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

param_grid = {
    'n_estimators': [100],
    'max_depth': [20],
    'min_samples_split': [5]
}

grid = GridSearchCV(RandomForestRegressor(random_state=42),
                    param_grid, cv=3)

grid.fit(X_train, y_train)

modelo = grid.best_estimator_

# ==============================
# 7. AVALIAÇÃO
# ==============================
previsoes = modelo.predict(X_test)

print("\n--- RESULTADOS ---")
print("MAE:", metrics.mean_absolute_error(y_test, previsoes))
print("R2:", metrics.r2_score(y_test, previsoes))

# ==============================
# 8. GRÁFICOS
# ==============================
os.makedirs("resultados/graficos", exist_ok=True)

# Real vs Previsto
plt.plot(y_test.values, label="Real")
plt.plot(previsoes, label="Previsto")
plt.legend()
plt.savefig("resultados/graficos/real_vs_previsto.png")
plt.clf()

# Erro
erro = y_test - previsoes
plt.hist(erro)
plt.savefig("resultados/graficos/erro.png")
plt.clf()

print("\n✔ Pipeline final executado com sucesso!")