# ==============================================================
# 📈 Comparação de Modelos — Banana Quality
# ==============================================================

import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================
# Configuração da Página
# ==============================================================
st.set_page_config(page_title="Comparação de Modelos", page_icon="📊")

st.title("📈 Comparação de Modelos — Banana Quality")
st.markdown("""
Esta página apresenta uma **comparação entre os modelos treinados**, incluindo métricas de desempenho,
curvas ROC e Validação Cruzada.
""")
st.subheader("📈 Comparação de Modelos — Logistic Regression e Random Forest")
# ==============================================================
# Caminhos de arquivos
# ==============================================================
metrics_dir = os.path.join("outputs", "metrics")
figures_dir = os.path.join("outputs", "figures")

log_path = os.path.join(metrics_dir, "logistic_regression_metrics.json")
rf_path = os.path.join(metrics_dir, "random_forest_metrics.json")


roc_log_path = os.path.join(figures_dir, "roc_logistic_regression.png")
roc_rf_path = os.path.join(figures_dir, "roc_random_forest.png")

# ==============================================================
# Carregar métricas dos modelos
# ==============================================================
if os.path.exists(log_path) and os.path.exists(rf_path):
    with open(log_path, "r") as f:
        metrics_log = json.load(f)
    with open(rf_path, "r") as f:
        metrics_rf = json.load(f)

    # ==============================================================
    # Tabela Comparativa
    # ==============================================================
    st.subheader("📋 Tabela Comparativa de Métricas")
    df_comp = pd.DataFrame(
        {"Regressão Logística": metrics_log, "Random Forest": metrics_rf}
    )
    df_comp = df_comp.round(4)
    st.dataframe(df_comp, use_container_width=True)

    # ==============================================================
    # Gráfico Comparativo de Desempenho
    # ==============================================================
    st.subheader("📊 Comparação Visual das Métricas")
    df_plot = df_comp.reset_index().melt(
        id_vars="index", var_name="Modelo", value_name="Valor"
    )
    df_plot.rename(columns={"index": "Métrica"}, inplace=True)
    order = (
        df_plot.groupby("Métrica")["Valor"].mean().sort_values(ascending=False).index
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df_plot, x="Métrica", y="Valor", hue="Modelo", ax=ax, order=order)
    ax.set_title("Desempenho dos Modelos — Logistic vs Random Forest")
    ax.set_ylabel("Valor da Métrica")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    st.pyplot(fig)

    # ==============================================================
    # Curvas ROC
    # ==============================================================
    st.subheader("🧩 Curvas ROC dos Modelos")
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(roc_log_path):
            st.image(
                roc_log_path,
                caption="Curva ROC — Regressão Logística",
                use_container_width=True,
            )
        else:
            st.warning("Curva ROC da Regressão Logística não encontrada.")
    with col2:
        if os.path.exists(roc_rf_path):
            st.image(
                roc_rf_path,
                caption="Curva ROC — Random Forest",
                use_container_width=True,
            )
        else:
            st.warning("Curva ROC do Random Forest não encontrada.")

    # ==============================================================
    # Melhor Modelo (baseado na acurácia)
    # ==============================================================
    st.subheader("🏆 Melhor Modelo")
    if "Acurácia" in df_comp.index:
        if (
            df_comp.loc["Acurácia", "Regressão Logística"]
            > df_comp.loc["Acurácia", "Random Forest"]
        ):
            modelo_vencedor = "Regressão Logística"
        else:
            modelo_vencedor = "Random Forest"
        st.success(
            f"O modelo com **melhor desempenho geral** (baseado na Acurácia) é: **{modelo_vencedor}** 🎯"
        )
    else:
        st.warning("⚠️ A métrica 'Acurácia' não foi encontrada nas métricas carregadas.")
        
        
#******************************************************************************************
#---------------------------comparação dos tres modelos -----------------------------------
# ******************************************************************************************
st.subheader("📈 Comparação de Modelos — Logistic Regression, Random Forest e KNN Classifier")
# ==============================================================
# Caminhos de arquivos
# ==============================================================
metrics_dir = os.path.join("outputs", "metrics")
figures_dir = os.path.join("outputs", "figures")

log_path = os.path.join(metrics_dir, "logistic_regression_metrics.json")
rf_path = os.path.join(metrics_dir, "random_forest_metrics.json")
knn_path = os.path.join(metrics_dir, "knn_classifier_metrics.json")

roc_log_path = os.path.join(figures_dir, "roc_logistic_regression.png")
roc_rf_path = os.path.join(figures_dir, "roc_random_forest.png")
roc_knn_path = os.path.join(figures_dir, "roc_knn.png")

# ==============================================================
# Carregar métricas dos modelos
# ==============================================================
if os.path.exists(log_path) and os.path.exists(rf_path) and os.path.exists(knn_path):
    with open(log_path, "r") as f:
        metrics_log = json.load(f)
    with open(rf_path, "r") as f:
        metrics_rf = json.load(f)
    with open(knn_path, "r") as f:
        metrics_knn = json.load(f)

    # ==============================================================
    # Tabela Comparativa
    # ==============================================================
    st.subheader("📋 Tabela Comparativa de Métricas")
    df_comp = pd.DataFrame(
        {
            "Regressão Logística": metrics_log,
            "Random Forest": metrics_rf,
            "KNN": metrics_knn,
        }
    )
    df_comp = df_comp.round(4)
    st.dataframe(df_comp, use_container_width=True)

    # ==============================================================
    # Gráfico Comparativo de Desempenho
    # ==============================================================
    st.subheader("📊 Comparação Visual das Métricas")
    df_plot = df_comp.reset_index().melt(
        id_vars="index", var_name="Modelo", value_name="Valor"
    )
    df_plot.rename(columns={"index": "Métrica"}, inplace=True)
    order = (
        df_plot.groupby("Métrica")["Valor"].mean().sort_values(ascending=False).index
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_plot, x="Métrica", y="Valor", hue="Modelo", ax=ax, order=order)
    ax.set_title("Desempenho dos Modelos — Logistic vs Random Forest vs KNN")
    ax.set_ylabel("Valor da Métrica")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    st.pyplot(fig)

    # ==============================================================
    # Curvas ROC
    # ==============================================================
    st.subheader("🧩 Curvas ROC dos Modelos")
    col1, col2, col3 = st.columns(3)

    with col1:
        if os.path.exists(roc_log_path):
            st.image(
                roc_log_path,
                caption="Curva ROC — Regressão Logística",
                use_container_width=True,
            )
        else:
            st.warning("Curva ROC da Regressão Logística não encontrada.")

    with col2:
        if os.path.exists(roc_rf_path):
            st.image(
                roc_rf_path,
                caption="Curva ROC — Random Forest",
                use_container_width=True,
            )
        else:
            st.warning("Curva ROC do Random Forest não encontrada.")

    with col3:
        if os.path.exists(roc_knn_path):
            st.image(roc_knn_path, caption="Curva ROC — KNN", use_container_width=True)
        else:
            st.warning("Curva ROC do KNN não encontrada.")

    # ==============================================================
    # Melhor Modelo (baseado na Acurácia)
    # ==============================================================
    st.subheader("🏆 Melhor Modelo")
    if "Acurácia" in df_comp.index:
        melhor_modelo = df_comp.loc["Acurácia"].idxmax()
        st.success(
            f"O modelo com **melhor desempenho geral** (baseado na Acurácia) é: **{melhor_modelo}** 🎯"
        )
    else:
        st.warning("⚠️ A métrica 'Acurácia' não foi encontrada nas métricas carregadas.")
else:
    st.warning(
        "⚠️ Métricas de um ou mais modelos não foram encontradas. Execute o treinamento antes de comparar."
    )


# ******************************************************************************************
# ---------------------------comparação dos Quatro modelos -----------------------------------
# ******************************************************************************************
st.subheader(
    "Comparação de Modelos — Logistic Regression, Random Forest, KNN Classifier e SVM Classifier"
)
# ==============================================================
# Caminhos de arquivos
# ==============================================================
metrics_dir = os.path.join("outputs", "metrics")
figures_dir = os.path.join("outputs", "figures")

model_files = {
    "Regressão Logística": "logistic_regression_metrics.json",
    "Random Forest": "random_forest_metrics.json",
    "KNN": "knn_classifier_metrics.json",
    "SVM": "svm_classifier_metrics.json",
}
roc_files = {
    "Regressão Logística": "roc_logistic_regression.png",
    "Random Forest": "roc_random_forest.png",
    "KNN": "roc_knn.png",
    "SVM": "roc_svm.png",
}

# ==============================================================
# Carregar métricas
# ==============================================================
metrics = {}
for model_name, file_name in model_files.items():
    path = os.path.join(metrics_dir, file_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            metrics[model_name] = json.load(f)

if metrics:
    # ==============================================================
    # Tabela Comparativa
    # ==============================================================
    st.subheader("📋 Tabela Comparativa de Métricas")
    df_comp = pd.DataFrame(metrics).round(4)
    st.dataframe(df_comp, use_container_width=True)

    # ==============================================================
    # Gráfico Comparativo de Desempenho
    # ==============================================================
    st.subheader("📊 Comparação Visual das Métricas")
    df_plot = df_comp.reset_index().melt(
        id_vars="index", var_name="Modelo", value_name="Valor"
    )
    df_plot.rename(columns={"index": "Métrica"}, inplace=True)
    order = (
        df_plot.groupby("Métrica")["Valor"].mean().sort_values(ascending=False).index
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_plot, x="Métrica", y="Valor", hue="Modelo", ax=ax, order=order)
    ax.set_title("Desempenho dos Modelos")
    ax.set_ylabel("Valor da Métrica")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    st.pyplot(fig)

    # ==============================================================
    # Curvas ROC
    # ==============================================================
    st.subheader("🧩 Curvas ROC dos Modelos")
    cols = st.columns(len(roc_files))
    for i, (model_name, roc_file) in enumerate(roc_files.items()):
        with cols[i]:
            path = os.path.join(figures_dir, roc_file)
            if os.path.exists(path):
                st.image(
                    path, caption=f"Curva ROC — {model_name}", use_container_width=True
                )
            else:
                st.warning(f"Curva ROC do {model_name} não encontrada.")

    # ==============================================================
    # Melhor Modelo (baseado na Acurácia)
    # ==============================================================
    st.subheader("🏆 Melhor Modelo")
    if "Acurácia" in df_comp.index:
        melhor_modelo = df_comp.loc["Acurácia"].idxmax()
        st.success(
            f"O modelo com **melhor desempenho geral** (baseado na Acurácia) é: **{melhor_modelo}** 🎯"
        )
    else:
        st.warning("⚠️ A métrica 'Acurácia' não foi encontrada nas métricas carregadas.")

else:
    st.warning(
        "⚠️ Nenhuma métrica encontrada. Execute o treinamento dos modelos antes de realizar a comparação."
    )


# ******************************************************************************************
# ---------------------------comparação dos Cinco modelos -----------------------------------
# ******************************************************************************************
st.subheader(
    "Comparação de Modelos — Logistic Regression, Random Forest, KNN Classifier, SVM Classifier e Gradient boosting"
)
# ==============================================================
# Caminhos de arquivos
# ==============================================================
metrics_dir = os.path.join("outputs", "metrics")
figures_dir = os.path.join("outputs", "figures")

model_files = {
    "Regressão Logística": "logistic_regression_metrics.json",
    "Random Forest": "random_forest_metrics.json",
    "KNN": "knn_classifier_metrics.json",
    "SVM": "svm_classifier_metrics.json",
    "Gradient Boosting": "gradient_boosting_metrics.json",
}
roc_files = {
    "Regressão Logística": "roc_logistic_regression.png",
    "Random Forest": "roc_random_forest.png",
    "KNN": "roc_knn.png",
    "SVM": "roc_svm.png",
    "Gradient Boosting": "roc_gradient_boosting.png",
}

# ==============================================================
# Carregar métricas
# ==============================================================
metrics = {}
for model_name, file_name in model_files.items():
    path = os.path.join(metrics_dir, file_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            metrics[model_name] = json.load(f)

if metrics:
    # ==============================================================
    # Tabela Comparativa
    # ==============================================================
    st.subheader("📋 Tabela Comparativa de Métricas")
    df_comp = pd.DataFrame(metrics).round(4)
    st.dataframe(df_comp, use_container_width=True)

    # ==============================================================
    # Gráfico Comparativo de Desempenho
    # ==============================================================
    st.subheader("📊 Comparação Visual das Métricas")
    df_plot = df_comp.reset_index().melt(
        id_vars="index", var_name="Modelo", value_name="Valor"
    )
    df_plot.rename(columns={"index": "Métrica"}, inplace=True)
    order = (
        df_plot.groupby("Métrica")["Valor"].mean().sort_values(ascending=False).index
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_plot, x="Métrica", y="Valor", hue="Modelo", ax=ax, order=order)
    ax.set_title("Desempenho dos Modelos")
    ax.set_ylabel("Valor da Métrica")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    st.pyplot(fig)

    # ==============================================================
    # Curvas ROC
    # ==============================================================
    st.subheader("🧩 Curvas ROC dos Modelos")
    cols = st.columns(len(roc_files))
    for i, (model_name, roc_file) in enumerate(roc_files.items()):
        with cols[i]:
            path = os.path.join(figures_dir, roc_file)
            if os.path.exists(path):
                st.image(
                    path, caption=f"Curva ROC — {model_name}", use_container_width=True
                )
            else:
                st.warning(f"Curva ROC do {model_name} não encontrada.")

    # ==============================================================
    # Melhor Modelo (baseado na Acurácia)
    # ==============================================================
    st.subheader("🏆 Melhor Modelo")
    if "Acurácia" in df_comp.index:
        melhor_modelo = df_comp.loc["Acurácia"].idxmax()
        st.success(
            f"O modelo com **melhor desempenho geral** (baseado na Acurácia) é: **{melhor_modelo}** 🎯"
        )
    else:
        st.warning("⚠️ A métrica 'Acurácia' não foi encontrada nas métricas carregadas.")

else:
    st.warning(
        "⚠️ Nenhuma métrica encontrada. Execute o treinamento dos modelos antes de realizar a comparação."
    )


# ******************************************************************************************
# ---------------------------comparação dos SEIS modelos -----------------------------------
# ******************************************************************************************
st.subheader(
    "Comparação de Modelos — Logistic Regression, Random Forest, KNN Classifier, SVM Classifier, Gradient Boosting e XGboost Classifier"
)
# ==============================================================
# Caminhos de arquivos
# ==============================================================
metrics_dir = os.path.join("outputs", "metrics")
figures_dir = os.path.join("outputs", "figures")

model_files = {
    "Regressão Logística": "logistic_regression_metrics.json",
    "Random Forest": "random_forest_metrics.json",
    "KNN": "knn_classifier_metrics.json",
    "SVM": "svm_classifier_metrics.json",
    "Gradient Boosting": "gradient_boosting_metrics.json",
    "XGBoost": "xgboost_metrics.json",
}
roc_files = {
    "Regressão Logística": "roc_logistic_regression.png",
    "Random Forest": "roc_random_forest.png",
    "KNN": "roc_knn.png",
    "SVM": "roc_svm.png",
    "Gradient Boosting": "roc_gradient_boosting.png",
    "XGBoost": "roc_xgboost.png",
}

# ==============================================================
# Carregar métricas
# ==============================================================
metrics = {}
for model_name, file_name in model_files.items():
    path = os.path.join(metrics_dir, file_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            metrics[model_name] = json.load(f)

if metrics:
    # ==============================================================
    # Tabela Comparativa
    # ==============================================================
    st.subheader("📋 Tabela Comparativa de Métricas")
    df_comp = pd.DataFrame(metrics).round(4)
    st.dataframe(df_comp, use_container_width=True)

    # ==============================================================
    # Gráfico Comparativo de Desempenho
    # ==============================================================
    st.subheader("📊 Comparação Visual das Métricas")
    df_plot = df_comp.reset_index().melt(
        id_vars="index", var_name="Modelo", value_name="Valor"
    )
    df_plot.rename(columns={"index": "Métrica"}, inplace=True)
    order = (
        df_plot.groupby("Métrica")["Valor"].mean().sort_values(ascending=False).index
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_plot, x="Métrica", y="Valor", hue="Modelo", ax=ax, order=order)
    ax.set_title("Desempenho dos Modelos")
    ax.set_ylabel("Valor da Métrica")
    ax.set_xlabel("")
    plt.xticks(rotation=0)
    st.pyplot(fig)

    # ==============================================================
    # Curvas ROC
    # ==============================================================
    st.subheader("🧩 Curvas ROC dos Modelos")
    cols = st.columns(len(roc_files))
    for i, (model_name, roc_file) in enumerate(roc_files.items()):
        with cols[i]:
            path = os.path.join(figures_dir, roc_file)
            if os.path.exists(path):
                st.image(
                    path, caption=f"Curva ROC — {model_name}", use_container_width=True
                )
            else:
                st.warning(f"Curva ROC do {model_name} não encontrada.")

    # ==============================================================
    # Melhor Modelo (baseado na Acurácia)
    # ==============================================================
    st.subheader("🏆 Melhor Modelo")
    if "Acurácia" in df_comp.index:
        melhor_modelo = df_comp.loc["Acurácia"].idxmax()
        st.success(
            f"O modelo com **melhor desempenho geral** (baseado na Acurácia) é: **{melhor_modelo}** 🎯"
        )
    else:
        st.warning("⚠️ A métrica 'Acurácia' não foi encontrada nas métricas carregadas.")

else:
    st.warning(
        "⚠️ Nenhuma métrica encontrada. Execute o treinamento dos modelos antes de realizar a comparação."
    )
