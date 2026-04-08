# 🔥 Previsão de Queimadas em Rondônia com Machine Learning

Este projeto apresenta uma análise abrangente dos focos de queimadas no estado de Rondônia, utilizando dados históricos e técnicas de aprendizado de máquina para previsão da quantidade diária de focos no município de Porto Velho.

---

## 📌 Objetivo

Desenvolver um modelo preditivo capaz de estimar a **quantidade diária de focos de queimadas**, a partir de variáveis climáticas e temporais, contribuindo para o monitoramento ambiental e apoio à tomada de decisão.

---

## 🌎 Contexto

As queimadas na Amazônia são um problema ambiental crítico, influenciado por fatores climáticos e antrópicos. Rondônia, localizado no chamado *arco do desmatamento*, apresenta alta incidência de focos de calor, especialmente durante o período de estiagem.

---

## 📊 Base de Dados

Os dados utilizados foram obtidos do:

* **INPE – BDQueimadas**
* Período: **2015 a 2025**
* Formato: CSV

### 🔎 Variáveis principais:

* Precipitação
* Dias sem chuva
* Risco de fogo
* FRP (*Fire Radiative Power*)
* Localização (município, estado)
* Data e hora

---

## ⚙️ Metodologia

### 🔹 1. Pré-processamento

* Limpeza e padronização dos dados
* Conversão de datas
* Criação de variáveis temporais
* Tratamento de valores ausentes

### 🔹 2. Análise Exploratória (EDA)

* Tendência temporal
* Sazonalidade mensal
* Distribuição espacial dos focos
* Análise de intensidade (FRP)

### 🔹 3. Feature Engineering

* Variáveis climáticas agregadas
* Criação de variáveis de defasagem:

  * `lag_1`
  * `lag_2`
* Variável binária: estação seca

### 🔹 4. Modelagem

Modelos testados:

* Regressão Linear
* SVR (Support Vector Regression)
* Random Forest
* Gradient Boosting

---

## 🤖 Modelo Final

* **Algoritmo:** Random Forest
* **Problema:** Regressão
* **Local:** Porto Velho (RO)

### 📈 Resultados:

* R²: **0.788**
* MAE: **112.89**
* R² (validação cruzada): **0.672**

---

## 📊 Principais Insights

* Forte **sazonalidade** (ago–out)
* Alta **dependência temporal** (lag_1)
* Influência significativa de variáveis climáticas
* Porto Velho concentra maior número de focos

---

## 📉 Limitações

* Não inclui variáveis antrópicas diretamente
* Dificuldade na previsão de eventos extremos
* Uso de dados históricos (sem tempo real)

---

## 🚀 Trabalhos Futuros

* Inclusão de variáveis como:

  * NDVI (vegetação)
  * Umidade do solo
  * Vento
* Uso de dados em tempo real
* Aplicação de redes neurais (LSTM)
* Expansão para outros municípios

---

## 🛠️ Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib / Seaborn

---

## 📁 Estrutura do Projeto

```
📂 tcc/
 ├── data/                # Dados brutos e tratados
 ├── notebooks/          # Análises exploratórias
 ├── models/             # Modelos treinados
 ├── imagens/            # Gráficos e visualizações
 ├── src/                # Scripts principais
 └── README.md
```

---

## 📌 Como Executar

```bash
# Clonar repositório
git clone https://github.com/seu-usuario/seu-repositorio.git

# Entrar na pasta
cd seu-repositorio

# Instalar dependências
pip install -r requirements.txt

# Executar análise/modelo
python main.py
```

---

## 📖 Sobre o Projeto

Este projeto foi desenvolvido como Trabalho de Conclusão de Curso (TCC) em Ciência da Computação, com foco na aplicação de técnicas de aprendizado de máquina no monitoramento ambiental.

---

## 👩‍💻 Autora

**Gabrielly Cristine Araujo Rodrigues**

---


