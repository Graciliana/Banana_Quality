# 🍌 Banana Quality

> Projeto de análise de qualidade de bananas — classificação, regressão e visualização de dados.

---

## 📘 Visão Geral

O **Banana Quality** é um projeto de **ciência de dados** que tem como objetivo analisar e prever a **qualidade de bananas** com base em variáveis quantitativas e qualitativas.  
Através de técnicas de **análise exploratória de dados**, **modelagem estatística** e **machine learning**, busca-se identificar padrões que auxiliem na **classificação**, **logística** e **aproveitamento** das frutas.

---

## 🚀 Tecnologias Utilizadas

- **Python 3.x**
- **Pandas**, **NumPy**, **Scikit-learn**
- **Matplotlib** e **Seaborn**
- **Streamlit** (para visualizações interativas)
- **MySQL / Prisma** (opcional, se houver integração com banco de dados)
- **Git e GitHub**

---

## 🗂️ Estrutura do Repositório

Banana_Quality/
│
├── Data/ # Dados brutos e tratados
│
├── notebook/ # Notebooks de exploração, análise e modelagem
│
├── models/ # Modelos treinados (pickle, joblib, etc.)
│
├── outputs/ # Gráficos, relatórios e resultados
│
├── pages/ # Páginas de app (ex: Streamlit)
│
├── utils/ # Funções utilitárias de pré-processamento, métricas, etc.
│
├── main.py # Script principal (para execução local ou Streamlit)
├── requirements.txt # Dependências do projeto
└── README.md # Este arquivo

---

## 🎯 Objetivos do Projeto

1. Carregar e investigar os dados sobre qualidade de bananas.  
2. Executar análise **univariada, bivariada e multivariada**.  
3. Realizar **pré-processamento**:
   - Tratamento de valores faltantes  
   - Codificação de variáveis qualitativas  
   - Normalização/padronização de variáveis contínuas  
4. Treinar e comparar modelos de regressão:
   - Linear Regression  
   - Ridge Regression  
   - Random Forest  
   - XGBoost  
   - SVR  
5. Avaliar os modelos com métricas apropriadas.  
6. (Opcional) Realizar classificação: **bananas boas vs ruins**.  
7. Criar visualizações e/ou dashboard interativo em Streamlit.  

---

## ⚙️ Como Executar o Projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/Graciliana/Banana_Quality.git
cd Banana_Quality
```

### 2. Criar e ativar o ambiente virtual

```bash
python -m venv venv
# No Windows
venv\Scripts\activate
# No Linux/Mac
source venv/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Executar os notebooks

Abra os arquivos na pasta notebook/ e siga a ordem de execução (exploração → modelagem → avaliação).

### 5. Executar o dashboard (se aplicável)

```bash
streamlit run main.py

```

**Os resultados e gráficos estarão disponíveis na pasta outputs/.**

---

## 📊 Métricas de Avaliação

## Para Regressão

- MSE (Erro Quadrático Médio)

- RMSE (Raiz do Erro Quadrático Médio)

- MAE (Erro Absoluto Médio)

- R² (Coeficiente de Determinação)

## Para Classificação (opcional)

- Acurácia

- Precisão

- Recall

- F1-Score

- Matriz de Confusão

- Curva ROC / AUC

---

## 🤝 Contribuições

Contribuições são bem-vindas!
Sinta-se à vontade para:

- Abrir issues com sugestões ou bugs

- Criar pull requests com melhorias

- Propor novas visualizações ou modelos

## 📄 Licença

Este projeto está licenciado sob a MIT License.
Você pode usar, modificar e distribuir livremente, desde que mantenha os devidos créditos.

## 👩‍💻 Autora

Graciliana Kascher
[🔗 LinkedIn]<https://www.linkedin.com/in/gracilianakascher/>

💻 Ciência de Dados | Machine Learning | Visão Computacional | Análise de Dados
📊 Foco em análise de dados e aplicações práticas com Python e Streamlit

---

