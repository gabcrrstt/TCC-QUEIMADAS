import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

print("--- Aprimorando o Modelo com Engenharia de Features ---")

# --- PASSO 1: CARREGAR E PREPARAR OS DADOS (com método robusto) ---
file_path = 'dados_consolidados_2015_2024.csv'
try:
    df_raw = pd.read_csv(file_path)
    print(f"Arquivo '{file_path}' lido com sucesso!")
except FileNotFoundError:
    print(f"ERRO CRÍTICO: O arquivo '{file_path}' não foi encontrado.")
    exit()

# Pré-processamento
df_raw['DataHora'] = pd.to_datetime(df_raw['DataHora'])
df_raw['RiscoFogo'] = df_raw['RiscoFogo'].replace(-999.0, np.nan)

# Filtragem e Agregação (usando um método mais seguro para evitar o erro)
municipio_alvo = 'PORTO VELHO'
df_pvh = df_raw[df_raw['Municipio'] == municipio_alvo].copy()
df_pvh['Data'] = df_pvh['DataHora'].dt.date

# Agregando os dados passo a passo
df_grouped = df_pvh.groupby('Data')
dados_diarios = pd.DataFrame()
dados_diarios['qtd_focos'] = df_grouped.size()
dados_diarios['media_dias_sem_chuva'] = df_grouped['DiaSemChuva'].mean()
dados_diarios['media_precipitacao'] = df_grouped['Precipitacao'].mean()
dados_diarios['media_risco_fogo'] = df_grouped['RiscoFogo'].mean()
dados_diarios = dados_diarios.reset_index()
dados_diarios['media_risco_fogo'].fillna(0, inplace=True)
print("Dados preparados para modelagem com sucesso.")

# --- PASSO 2: ENGENHARIA DE FEATURES ---
# Criando a nova feature 'Mes'
dados_diarios['Data'] = pd.to_datetime(dados_diarios['Data'])
dados_diarios['Mes'] = dados_diarios['Data'].dt.month
print("Nova feature 'Mes' criada com sucesso!")

# --- PASSO 3: PREPARAÇÃO PARA O MACHINE LEARNING ---
df_modelo = dados_diarios.set_index('Data')
# Adicionamos 'Mes' à lista de features!
features = ['media_dias_sem_chuva', 'media_precipitacao', 'media_risco_fogo', 'Mes']
alvo = 'qtd_focos'

X = df_modelo[features]
y = df_modelo[alvo]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- PASSO 4: TREINAR E AVALIAR O NOVO MODELO APRIMORADO ---
print("\nTreinando o modelo Random Forest aprimorado (com a feature 'Mes')...")
modelo_aprimorado = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
modelo_aprimorado.fit(X_train, y_train)
previsoes = modelo_aprimorado.predict(X_test)

# Métricas do novo modelo
mae_novo = metrics.mean_absolute_error(y_test, previsoes)
r2_novo = metrics.r2_score(y_test, previsoes)
print("Modelo aprimorado treinado e avaliado.")

# --- PASSO 5: COMPARAR RESULTADOS ---
print("\n--- COMPARAÇÃO DE PERFORMANCE ---")
# Resultados do modelo anterior (campeão da última fase)
mae_antigo = 136.69
r2_antigo = 0.428

resultados = {
    'Modelo': ['Random Forest (Original)', 'Random Forest (com Mês)'],
    'Erro Médio Absoluto (MAE)': [mae_antigo, mae_novo],
    'Coeficiente R²': [r2_antigo, r2_novo]
}
df_resultados = pd.DataFrame(resultados)
df_resultados['Erro Médio Absoluto (MAE)'] = df_resultados['Erro Médio Absoluto (MAE)'].round(2)
df_resultados['Coeficiente R²'] = df_resultados['Coeficiente R²'].round(3)
print(df_resultados)

melhora_mae = (mae_antigo - mae_novo) / mae_antigo
melhora_r2 = (r2_novo - r2_antigo) / r2_antigo

print("\nConclusão:")
print(f"-> O Erro Médio (MAE) melhorou em {melhora_mae:.2%}.")
print(f"-> O poder de explicação (R²) melhorou em {melhora_r2:.2%}.")
if melhora_r2 > 0:
    print("🎉 Sucesso! A adição da feature 'Mes' melhorou significativamente o modelo.")
else:
    print("A adição da feature 'Mes' não resultou em uma melhoria significativa.")
    
# --- PASSO 6: VISUALIZAR A COMPARAÇÃO ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.set_theme(style="whitegrid")
fig.suptitle('Impacto da Engenharia de Features no Desempenho do Modelo', fontsize=18, weight='bold')

# Gráfico para MAE
sns.barplot(x='Modelo', y='Erro Médio Absoluto (MAE)', data=df_resultados, ax=axes[0], palette='coolwarm')
axes[0].set_title('Comparação do Erro Médio Absoluto (MAE)', fontsize=14, weight='bold')
for index, value in enumerate(df_resultados['Erro Médio Absoluto (MAE)']):
    axes[0].text(index, value, f'{value:.2f}', ha='center', va='bottom')

# Gráfico para R²
sns.barplot(x='Modelo', y='Coeficiente R²', data=df_resultados, ax=axes[1], palette='viridis')
axes[1].set_title('Comparação do Coeficiente R²', fontsize=14, weight='bold')
axes[1].set_ylim(0, max(df_resultados['Coeficiente R²']) * 1.15)
for index, value in enumerate(df_resultados['Coeficiente R²']):
    axes[1].text(index, value, f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout(rect=[0, 0.03, 1, 0.93])
plt.savefig('grafico_8_comparacao_feature_engineering.png')
print("\nGráfico 'grafico_8_comparacao_feature_engineering.png' salvo com sucesso.")