# ==============================================================
# 📊 Dashboard de Resultados — Banana Quality
# ==============================================================

import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================================================
# Configuração da Página
# ==============================================================
st.set_page_config(page_title="Dashboard de Resultados", page_icon="📊", layout="wide")

st.title("📊 Dashboard de Resultados — Comparação de Modelos")
st.markdown("""
Este painel apresenta a **síntese final dos resultados** dos modelos treinados, 
permitindo identificar **qual modelo apresentou o melhor desempenho geral** 
para o problema de classificação da **Qualidade das Bananas 🍌**.
""")

# ==============================================================
# Carregar métricas de todos os modelos
# ==============================================================
metrics_dir = os.path.join("outputs", "metrics")

model_files = {
    "Regressão Logística": "logistic_regression_metrics.json",
    "Random Forest": "random_forest_metrics.json",
    "KNN": "knn_classifier_metrics.json",
    "SVM": "svm_classifier_metrics.json",
    "Gradient Boosting": "gradient_boosting_metrics.json",
    "XGBoost": "xgboost_metrics.json",
}

metrics = {}
for model_name, file_name in model_files.items():
    path = os.path.join(metrics_dir, file_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            metrics[model_name] = json.load(f)

if not metrics:
    st.error("⚠️ Nenhum arquivo de métricas encontrado em `outputs/metrics`.")
    st.stop()

# ==============================================================
# Consolidar métricas
# ==============================================================
df_metrics = pd.DataFrame(metrics).T.round(4)
st.subheader("📋 Tabela Consolidada de Métricas")
st.dataframe(df_metrics, use_container_width=True)

# ==============================================================
# Gráficos de Comparação
# ==============================================================
st.subheader("📊 Comparação Visual das Métricas")
metrics_to_plot = ["Acurácia", "F1-score", "Precisão", "Recall", "RMSE", "R²"]

for metric in metrics_to_plot:
    if metric in df_metrics.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=df_metrics.index, y=df_metrics[metric], palette="viridis", ax=ax)
        ax.set_title(f"Comparação — {metric}")
        ax.set_ylabel(metric)
        ax.set_xlabel("Modelo")
        ax.bar_label(ax.containers[0], fmt="%.3f", padding=3)
        st.pyplot(fig)

# ==============================================================
# Gráfico Radar (Spider Chart)
# ==============================================================
st.subheader("🕸️ Gráfico Radar — Desempenho Multidimensional dos Modelos")

# Selecionar apenas métricas positivas para comparação visual
radar_metrics = ["Acurácia", "Precisão", "Recall", "F1-score"]
radar_df = df_metrics[radar_metrics]

# Normalizar valores entre 0 e 1 para melhor visualização
radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())

# Configurar ângulos do gráfico
labels = list(radar_norm.columns)
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # fechar o círculo

# Criar gráfico
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
for model in radar_norm.index:
    values = radar_norm.loc[model].tolist()
    values += values[:1]
    ax.plot(angles, values, label=model, linewidth=2)
    ax.fill(angles, values, alpha=0.15)

ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)
ax.set_title("Radar de Desempenho dos Modelos", size=14, pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
st.pyplot(fig)

# ==============================================================
# Identificar o melhor modelo
# ==============================================================
st.subheader("🏆 Conclusão dos Resultados")

best_accuracy = df_metrics["Acurácia"].idxmax()
best_f1 = df_metrics["F1-score"].idxmax()
best_r2 = df_metrics["R²"].idxmax()
best_rmse = df_metrics["RMSE"].idxmin()

col1, col2 = st.columns(2)
with col1:
    st.success(f"🎯 **Melhor Acurácia:** {best_accuracy}")
    st.info(f"💪 **Melhor F1-score:** {best_f1}")
with col2:
    st.warning(f"📈 **Melhor R²:** {best_r2}")
    st.error(f"📉 **Menor RMSE (erro):** {best_rmse}")

# ==============================================================
# Conclusão Automática
# ==============================================================
st.markdown("### 📚 Interpretação dos Resultados")

conclusao = f"""
Após a comparação de seis algoritmos de aprendizado supervisionado, observou-se que o modelo **{best_accuracy}**
apresentou o **melhor desempenho geral**, alcançando a maior acurácia e F1-score. 

Modelos como **{best_r2}** também mostraram bom poder de explicação da variabilidade (R² elevado), enquanto **{best_rmse}**
apresentou o menor erro médio quadrático (RMSE), indicando boa capacidade preditiva.

Esses resultados sugerem que o modelo **{best_accuracy}** é o mais indicado para uso prático
na **predição da qualidade das bananas**, equilibrando desempenho e estabilidade.
"""

st.markdown(conclusao)
