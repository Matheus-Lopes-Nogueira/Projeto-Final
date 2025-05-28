# Previsão de Inadimplência com Machine Learning

## 📌 Problema

A inadimplência representa um risco financeiro crítico para instituições de crédito. Este projeto tem como objetivo prever a probabilidade de um cliente se tornar inadimplente em até 2 anos com base em características demográficas e financeiras.

---

## 📊 Fonte de Dados

- Arquivo: `credit_scoring.ftr`
- Origem: Base disponibilizada pela EBAC
- Quantidade: Dados mensais de crédito com 15 safras (períodos)

---

## 🧪 Etapas do Projeto

### 1. Análise Exploratória (EDA)
- Análise de variáveis numéricas e categóricas
- Tratamento de valores ausentes e outliers
- Transformações logarítmicas para renda e tempo de emprego
- Agrupamento de categorias (tipo de renda, escolaridade, etc.)

### 2. Engenharia de Variáveis
- Criação de variáveis transformadas
- Criação de dummies e agrupamentos
- Detecção e substituição de outliers
- Padronização com `StandardScaler`
- Redução de dimensionalidade com `PCA` e `RFE`

### 3. Modelagem Estatística
- Regressão Logística com `statsmodels`
- Avaliação com métricas:
  - Acurácia
  - KS (Kolmogorov-Smirnov)
  - Gini
  - AUC

### 4. Modelagem com PyCaret
- Pré-processamento automático
- Treinamento com `LightGBM`
- Otimização de hiperparâmetros (`tune_model`)
- Geração de gráficos:
  - Curva ROC
  - Matriz de confusão
  - Importância de variáveis
  - KS

---

## 🎯 Resultados

| Conjunto       | Acurácia | Gini  | KS    |
|----------------|----------|-------|-------|
| Desenvolvimento| 0.87     | 0.72  | 0.48  |
| Out-of-time    | 0.85     | 0.69  | 0.46  |

O modelo demonstrou desempenho robusto na previsão de inadimplência, tanto na base de desenvolvimento quanto em dados fora do tempo.

---

## 💾 Modelos Treinados

- `model_final.pkl`: Regressão logística (manual)
- `modelo_credito_lgbm.pkl`: Modelo LightGBM (PyCaret)

---

## 📁 Estrutura do Repositório

