# =================================================================================================
# 🏠 APP.PY — Página Inicial do Projeto Banana Quality (ou Orange Quality)
# =================================================================================================

import streamlit as st


# =================================================================================================
# CONFIGURAÇÃO INICIAL DO APP
# =================================================================================================
st.set_page_config(
    page_title="Banana Quality | Home",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =================================================================================================
# CABEÇALHO
# =================================================================================================
st.markdown(
    """
    <h1 style='color:#ffff00; text-align:center;'>
        🍌 <span style='color:#FFD700;'>Projeto Qualidade da Banana</span>
    </h1>
    """,
    unsafe_allow_html=True
)
# =================================================================================================
# SEÇÃO: Introdução
# =================================================================================================
st.header("📘 Introdução")

st.write("""
O **projeto Banana Quality** tem como objetivo **analisar e prever a qualidade das bananas** 
a partir de diversas variáveis físico-químicas, utilizando **técnicas de Análise Exploratória de Dados (EDA)**, 
**Pré-processamento**, **Modelagem Preditiva** e **Comparação de Modelos de Machine Learning**.

Com esse estudo, buscamos compreender **quais fatores mais influenciam a qualidade** e 
criar modelos que possam apoiar decisões em processos de **classificação, inspeção e controle de qualidade**.
""")

# =================================================================================================
# SEÇÃO: Estrutura do Projeto
# =================================================================================================
st.header("🧩 Estrutura do Projeto")

st.markdown("""
O sistema é composto por **múltiplas páginas**, cada uma com uma função específica no fluxo de análise:

1. **📊 EDA (Análise Exploratória de Dados)** – Visualizações e estatísticas descritivas do dataset.  
2. **⚙️ Pré-Processamento** – Limpeza, codificação, normalização e tratamento de outliers.  
3. **🤖 Modelagem** – Treinamento de diferentes modelos de Machine Learning.  
4. **📈 Comparação de Modelos** – Avaliação de desempenho e métricas comparativas.  
5. **🧩 Conclusões** – Resultados finais e insights obtidos durante o estudo.
""")

# =================================================================================================
# SEÇÃO: Navegação e Créditos
# =================================================================================================
st.markdown("---")
st.header("👩‍💻 Autora e Contato")

st.write("""
**Desenvolvido por [Graciliana Kascher](https://www.linkedin.com/in/gracilianakascher/)**  
Especialista em **Machine Learning, Visão Computacional e Ciência de Dados**.  

Este projeto foi estruturado com foco em **modularidade, reprodutibilidade e clareza**, 
utilizando **Streamlit** como framework interativo.
""")


