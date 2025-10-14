import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import metrics

print("--- Competição Final de Modelos Preditivos (Versão Corrigida) ---")

# --- PASSO 1: CARREGAR E PREPARAR OS DADOS ---
# Use o seu arquivo consolidado com todos os anos.
ARQUIVO_CONSOLIDADO = 'dados_consolidados_2015_2024.csv' 
try:
    df_raw = pd.read_csv(ARQUIVO_CONSOLIDADO, parse_dates=['DataHora'])
except FileNotFoundError:
    print(f"ERRO: Arquivo '{ARQUIVO_CONSOLIDADO}' não encontrado.")
    exit()

# Preparação dos dados
df_raw['RiscoFogo'] = df_raw['RiscoFogo'].replace(-999.0, np.nan)
municipio_alvo = 'PORTO VELHO'
df_pvh = df_raw[df_raw['Municipio'] == municipio_alvo].copy()
df_pvh['Data'] = df_pvh['DataHora'].dt.date
dados_diarios = df_pvh.groupby('Data').agg(
    qtd_focos=('Data', 'count'),
    media_dias_sem_chuva=('DiaSemChuva', 'mean'),
    media_precipitacao=('Precipitacao', 'mean'),
    media_risco_fogo=('RiscoFogo', 'mean')
).reset_index()

# --- PASSO 2: PREPARAÇÃO FINAL E LIMPEZA PARA MODELAGEM ---
df_modelo = dados_diarios.set_index(pd.to_datetime(dados_diarios['Data'])).drop('Data', axis=1)
features = ['media_dias_sem_chuva', 'media_precipitacao', 'media_risco_fogo']
alvo = 'qtd_focos'

X = df_modelo[features]
y = df_modelo[alvo]

# ***** A CORREÇÃO ESTÁ AQUI *****
# Preenchemos QUALQUER NaN restante com 0. Isso garante compatibilidade com todos os modelos.
X.fillna(0, inplace=True)
print("Dados preparados e limpos, sem valores NaN.")

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- PASSO 3: TREINAR E AVALIAR TODOS OS MODELOS ---
modelos = {
    "Regressão Linear": LinearRegression(),
    "SVR": SVR(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

resultados = []
for nome, modelo in modelos.items():
    print(f"\nTreinando {nome}...")
    modelo.fit(X_train, y_train)
    previsoes = modelo.predict(X_test)
    
    mae = metrics.mean_absolute_error(y_test, previsoes)
    r2 = metrics.r2_score(y_test, previsoes)
    
    resultados.append({'Modelo': nome, 'Erro Médio Absoluto (MAE)': mae, 'Coeficiente R²': r2})
    print(f"{nome} avaliado.")

# --- PASSO 4: APRESENTAR O RESULTADO FINAL ---
df_resultados = pd.DataFrame(resultados).sort_values(by='Coeficiente R²', ascending=False)
print("\n--- RESULTADO FINAL DA COMPETIÇÃO ---")
print(df_resultados)

# Declarar o campeão
campeao = df_resultados.iloc[0]
print(f"\n🏆 O modelo campeão é o '{campeao['Modelo']}' com R² de {campeao['Coeficiente R²']:.3f} e MAE de {campeao['Erro Médio Absoluto (MAE)']:.2f}.")

# --- PASSO 5: VISUALIZAR A COMPARAÇÃO FINAL ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
sns.set_theme(style="whitegrid")
fig.suptitle('Resultado Final da Competição de Modelos', fontsize=18, weight='bold')

sns.barplot(x='Erro Médio Absoluto (MAE)', y='Modelo', data=df_resultados.sort_values(by='Erro Médio Absoluto (MAE)', ascending=True), ax=axes[0], palette='Reds', hue='Modelo', legend=False)
axes[0].set_title('Comparação do Erro Médio (Menor é Melhor)', fontsize=14)

sns.barplot(x='Coeficiente R²', y='Modelo', data=df_resultados.sort_values(by='Coeficiente R²', ascending=False), ax=axes[1], palette='Greens', hue='Modelo', legend=False)
axes[1].set_title('Comparação do R² (Maior é Melhor)', fontsize=14)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('grafico_final_competicao_modelos.png')
print("\nGráfico 'grafico_final_competicao_modelos.png' salvo com sucesso.")