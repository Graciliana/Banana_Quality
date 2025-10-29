# src/pages/3_⚙️_Modelagem.py
import os
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from models.logistic_regression import treinar_logistic_regression
from models.random_forest import treinar_random_forest

# ======================================================
# Configurações da Página
# ======================================================
st.set_page_config(page_title="Modelagem — Banana Quality", page_icon="🍌")

st.title("⚙️ Etapa de Modelagem")
st.markdown("""
Agora que os dados já foram limpos e codificados, podemos partir para a modelagem.
O primeiro passo é **dividir o dataset em treino e teste**, garantindo que os modelos
possam ser avaliados de forma justa e confiável.
""")

# ======================================================
# 1️⃣ Carregar o dataset pré-processado
# ======================================================
data_path = os.path.join("outputs", "data", "dataset_preprocessado.csv")

if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    st.success("✅ Dataset pré-processado carregado com sucesso!")
    st.dataframe(df.head())
else:
    st.error(
        "❌ Dataset pré-processado não encontrado! Execute primeiro o passo de Pré-processamento."
    )
    st.stop()

# ======================================================
# 2️⃣ Divisão dos Dados (Treino e Teste)
# ======================================================
st.subheader("🔹 Divisão dos Dados em Treino e Teste")

# Identificar a variável alvo
target_col = "Qualidade"  # ajuste o nome se necessário
X = df.drop(columns=[target_col])
y = df[target_col]

# Divisão fixa (80% treino / 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Exibir informações da divisão
st.markdown(f"""
✅ **Divisão concluída com sucesso!**
- **Treino:** {X_train.shape[0]} amostras  
- **Teste:** {X_test.shape[0]} amostras
""")

# Opcional: visualizar pequenas amostras
# st.write("Exemplo de dados de treino:")
# st.dataframe(X_train.head())

# st.write("Exemplo de rótulos (y_train):")
# st.dataframe(y_train.head())

# ======================================================
# 3️⃣ Salvar os dados divididos
# ======================================================
output_dir = os.path.join("outputs", "data")
os.makedirs(output_dir, exist_ok=True)

X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

st.success("💾 Dados de treino e teste salvos em `outputs/data/`.")


# *************************************************************************************************
# *************************************************************************************************
# ======================================================
# Regressão Logística
# ======================================================
st.title("🧠 Modelagem — Regressão Logística")

st.subheader("⚙️ Treinamento e Avaliação — Regressão Logística")

if st.button("🚀 Treinar Modelo Regressão Logística"):
    with st.spinner("Treinando modelo de Regressão Logística..."):
        # ✅ Correção: função retorna 9 valores, não 7
        modelo, metrics, cm, report, y_pred, y_pred_proba, fpr, tpr, roc_auc = (
            treinar_logistic_regression(
                X_train, X_test, y_train, y_test, save_path="outputs/"
            )
        )

    st.success("✅ Modelo treinado com sucesso!")

    # ======================================================
    # Métricas
    # ======================================================
    st.markdown("### 📊 Métricas do Modelo")
    st.json(metrics)

    metrics_df = pd.DataFrame(metrics.items(), columns=["Métrica", "Valor"])
    metrics_df["Valor"] = metrics_df["Valor"].apply(
        lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x)
    )
    # CORREÇÃO DE AVISO: use_container_width=True -> width='stretch'
    st.dataframe(metrics_df, width="stretch")

    # ======================================================
    # Matriz de Confusão
    # ======================================================
    st.markdown("### 🔹 Matriz de Confusão")
    cm_df = pd.DataFrame(
        cm,
        index=[f"Real {cls}" for cls in modelo.classes_],
        columns=[f"Previsto {cls}" for cls in modelo.classes_],
    )
    # CORREÇÃO DE AVISO: use_container_width=True -> width='stretch'
    st.dataframe(cm_df, width="stretch")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão - Regressão Logística")
    st.pyplot(fig)

    # ======================================================
    # Relatório
    # ======================================================
    st.markdown("### 📋 Relatório de Classificação")
    st.dataframe(pd.DataFrame(report).transpose())

    # ======================================================
    # Curva ROC
    # ======================================================
    if fpr is not None and tpr is not None:
        st.markdown("### 🧩 Curva ROC")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(
            fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
        )
        ax_roc.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Curva ROC - Regressão Logística")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
    else:
        st.warning("⚠️ Curva ROC não disponível para múltiplas classes.")

    # ======================================================
    # Arquivos
    # ======================================================
    st.markdown("### 💾 Arquivos salvos")
    st.markdown("""
    - `outputs/models/logistic_regression.pkl` → modelo treinado  
    - `outputs/metrics/logistic_regression_metrics.json` → métricas salvas  
    - `outputs/figures/roc_logistic_regression.png` → curva ROC  
    """)

else:
    st.info(
        "Clique em **'🚀 Treinar Modelo Regressão Logística'** para iniciar o treinamento."
    )


# *************************************************************************************************
# *************************************************************************************************

# ======================================================
# 🧠 Modelagem — Random Forest
# ======================================================
st.title("🌳 Modelagem — Random Forest")

st.subheader("⚙️ Treinamento e Avaliação — Random Forest")


if st.button("🚀 Treinar Modelo Random Forest"):
    with st.spinner("Treinando modelo Random Forest..."):
        modelo, metrics, cm, report, y_pred, y_pred_proba, fpr, tpr, roc_auc = (
            treinar_random_forest(
                X_train, X_test, y_train, y_test, save_path="outputs/"
            )
        )

    st.success("✅ Modelo treinado com sucesso!")

    # ==========================================
    # Exibir Métricas
    # ==========================================
    st.markdown("### 📊 Métricas do Modelo")
    st.json(metrics)

    metrics_df = pd.DataFrame(metrics.items(), columns=["Métrica", "Valor"])
    metrics_df["Valor"] = metrics_df["Valor"].apply(
        lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x)
    )
    # CORREÇÃO DE AVISO: use_container_width=True -> width='stretch'
    st.dataframe(metrics_df, width="stretch")

    # ==========================================
    # Matriz de Confusão
    # ==========================================
    st.markdown("### 🔹 Matriz de Confusão")
    cm_df = pd.DataFrame(
        cm,
        index=[f"Real {cls}" for cls in modelo.classes_],
        columns=[f"Previsto {cls}" for cls in modelo.classes_],
    )
    # CORREÇÃO DE AVISO: use_container_width=True -> width='stretch'
    st.dataframe(cm_df, width="stretch")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão - Random Forest")
    st.pyplot(fig)

    # ==========================================
    # Relatório de Classificação
    # ==========================================
    st.markdown("### 📋 Relatório de Classificação")
    st.dataframe(pd.DataFrame(report).transpose())

    # ==========================================
    # Curva ROC
    # ==========================================
    if fpr is not None and tpr is not None:
        st.markdown("### 🧩 Curva ROC")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Curva ROC - Random Forest")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
    else:
        st.warning("⚠️ Curva ROC não disponível para múltiplas classes.")

    # ==========================================
    # Arquivos salvos
    # ==========================================
    st.markdown("### 💾 Arquivos salvos")
    st.markdown("""
    - `outputs/models/random_forest.pkl` → modelo treinado  
    - `outputs/metrics/random_forest_metrics.json` → métricas salvas  
    - `outputs/figures/roc_random_forest.png` → curva ROC  
    """)
else:
    st.info(
        "Clique em **'🚀 Treinar Modelo Random Forest'** para iniciar o treinamento."
    )
    # **********************************************************************************************************************************************************************************************
# ======================================================
# 🧠 Modelagem — KNN
# ======================================================
st.title("🤖 Modelagem — KNN Classifier")

st.subheader("⚙️ Treinamento e Avaliação — KNN")

if st.button("🚀 Treinar Modelo KNN"):
    with st.spinner("Treinando modelo KNN..."):
        from models.knn_classifier import treinar_knn  # type: ignore

        modelo, metrics, cm, report, y_pred, y_pred_proba, fpr, tpr, roc_auc = (
            treinar_knn(X_train, X_test, y_train, y_test, save_path="outputs/")
        )

    st.success("✅ Modelo treinado com sucesso!")

    # ==========================================
    # Exibir Métricas
    # ==========================================
    st.markdown("### 📊 Métricas do Modelo")
    st.json(metrics)

    metrics_df = pd.DataFrame(metrics.items(), columns=["Métrica", "Valor"])
    metrics_df["Valor"] = metrics_df["Valor"].apply(
        lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x)
    )
    # CORREÇÃO DE AVISO: use_container_width=True -> width='stretch'
    st.dataframe(metrics_df, width="stretch")

    # ==========================================
    # Matriz de Confusão
    # ==========================================
    st.markdown("### 🔹 Matriz de Confusão")
    cm_df = pd.DataFrame(
        cm,
        index=[f"Real {cls}" for cls in modelo.classes_],
        columns=[f"Previsto {cls}" for cls in modelo.classes_],
    )
    # CORREÇÃO DE AVISO: use_container_width=True -> width='stretch'
    st.dataframe(cm_df, width="stretch")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão - KNN")
    st.pyplot(fig)

    # ==========================================
    # Relatório de Classificação
    # ==========================================
    st.markdown("### 📋 Relatório de Classificação")
    st.dataframe(pd.DataFrame(report).transpose())

    # ==========================================
    # Curva ROC
    # ==========================================
    if fpr is not None and tpr is not None:
        st.markdown("### 🧩 Curva ROC")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Curva ROC - KNN")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
    else:
        st.warning("⚠️ Curva ROC não disponível para múltiplas classes.")

    # ==========================================
    # Arquivos salvos
    # ==========================================
    st.markdown("### 💾 Arquivos salvos")
    st.markdown("""
    - `outputs/models/knn_classifier.pkl` → modelo treinado  
    - `outputs/metrics/knn_classifier_metrics.json` → métricas salvas  
    - `outputs/figures/roc_knn.png` → curva ROC  
    """)
else:
    st.info("Clique em **'🚀 Treinar Modelo KNN'** para iniciar o treinamento.")

# ======================================================
# 🧠 Modelagem — SVM
# ======================================================
st.title("🛡️ Modelagem — SVM Classifier")

st.subheader("⚙️ Treinamento e Avaliação — SVM")

if st.button("🚀 Treinar Modelo SVM"):
    with st.spinner("Treinando modelo SVM..."):
        from models.svm_classifier import treinar_svm  # type: ignore

        modelo, metrics, cm, report, y_pred, y_pred_proba, fpr, tpr, roc_auc = (
            treinar_svm(X_train, X_test, y_train, y_test, save_path="outputs/")
        )

    st.success("✅ Modelo treinado com sucesso!")

    # ==========================================
    # Exibir Métricas
    # ==========================================
    st.markdown("### 📊 Métricas do Modelo")
    st.json(metrics)

    metrics_df = pd.DataFrame(metrics.items(), columns=["Métrica", "Valor"])
    metrics_df["Valor"] = metrics_df["Valor"].apply(
        lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x)
    )
    # CORREÇÃO DE AVISO: use_container_width=True -> width='stretch'
    st.dataframe(metrics_df, width="stretch")

    # ==========================================
    # Matriz de Confusão
    # ==========================================
    st.markdown("### 🔹 Matriz de Confusão")
    cm_df = pd.DataFrame(
        cm,
        index=[f"Real {cls}" for cls in modelo.classes_],
        columns=[f"Previsto {cls}" for cls in modelo.classes_],
    )
    # CORREÇÃO DE AVISO: use_container_width=True -> width='stretch'
    st.dataframe(cm_df, width="stretch")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão - SVM")
    st.pyplot(fig)

    # ==========================================
    # Relatório de Classificação
    # ==========================================
    st.markdown("### 📋 Relatório de Classificação")
    st.dataframe(pd.DataFrame(report).transpose())

    # ==========================================
    # Curva ROC
    # ==========================================
    if fpr is not None and tpr is not None:
        st.markdown("### 🧩 Curva ROC")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", color="#00CEC8")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Curva ROC - SVM")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
    else:
        st.warning("⚠️ Curva ROC não disponível para múltiplas classes.")

    # ==========================================
    # Arquivos salvos
    # ==========================================
    st.markdown("### 💾 Arquivos salvos")
    st.markdown("""
    - `outputs/models/svm_classifier.pkl` → modelo treinado  
    - `outputs/metrics/svm_classifier_metrics.json` → métricas salvas  
    - `outputs/figures/roc_svm.png` → curva ROC  
    """)
else:
    st.info("Clique em **'🚀 Treinar Modelo SVM'** para iniciar o treinamento.")

# ======================================================
# 🧠 Modelagem — Gradient Boosting
# ======================================================
st.title("🔥 Modelagem — Gradient Boosting")

st.subheader("⚙️ Treinamento e Avaliação — Gradient Boosting")

if st.button("🚀 Treinar Modelo Gradient Boosting"):
    with st.spinner("Treinando modelo Gradient Boosting..."):
        from models.gradient_boosting_classifier import treinar_gradient_boosting  # type: ignore

        modelo, metrics, cm, report, y_pred, y_pred_proba, fpr, tpr, roc_auc = (
            treinar_gradient_boosting(
                X_train, X_test, y_train, y_test, save_path="outputs/"
            )
        )

    st.success("✅ Modelo treinado com sucesso!")

    # ==========================================
    # Exibir Métricas
    # ==========================================
    st.markdown("### 📊 Métricas do Modelo")
    st.json(metrics)

    metrics_df = pd.DataFrame(metrics.items(), columns=["Métrica", "Valor"])
    metrics_df["Valor"] = metrics_df["Valor"].apply(
        lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x)
    )
    # CORREÇÃO DE AVISO: use_container_width=True -> width='stretch'
    st.dataframe(metrics_df, width="stretch")

    # ==========================================
    # Matriz de Confusão
    # ==========================================
    st.markdown("### 🔹 Matriz de Confusão")
    cm_df = pd.DataFrame(
        cm,
        index=[f"Real {cls}" for cls in modelo.classes_],
        columns=[f"Previsto {cls}" for cls in modelo.classes_],
    )
    # CORREÇÃO DE AVISO: use_container_width=True -> width='stretch'
    st.dataframe(cm_df, width="stretch")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão - Gradient Boosting")
    st.pyplot(fig)

    # ==========================================
    # Relatório de Classificação
    # ==========================================
    st.markdown("### 📋 Relatório de Classificação")
    st.dataframe(pd.DataFrame(report).transpose())

    # ==========================================
    # Curva ROC
    # ==========================================
    if fpr is not None and tpr is not None:
        st.markdown("### 🧩 Curva ROC")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", color="#E74C3C")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Curva ROC - Gradient Boosting")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
    else:
        st.warning("⚠️ Curva ROC não disponível para múltiplas classes.")

    # ==========================================
    # Arquivos salvos
    # ==========================================
    st.markdown("### 💾 Arquivos salvos")
    st.markdown("""
    - `outputs/models/gradient_boosting.pkl` → modelo treinado  
    - `outputs/metrics/gradient_boosting_metrics.json` → métricas salvas  
    - `outputs/figures/roc_gradient_boosting.png` → curva ROC  
    """)
else:
    st.info(
        "Clique em **'🚀 Treinar Modelo Gradient Boosting'** para iniciar o treinamento."
    )


# ======================================================
# 🧠 Modelagem — XGBoost
# ======================================================
st.title("🚀 Modelagem — XGBoost Classifier")

st.subheader("⚙️ Treinamento e Avaliação — XGBoost")

if st.button("🚀 Treinar Modelo XGBoost"):
    with st.spinner("Treinando modelo XGBoost..."):
        from models.xgboost_classifier import treinar_xgboost  # type: ignore

        modelo, metrics, cm, report, y_pred, y_pred_proba, fpr, tpr, roc_auc = (
            treinar_xgboost(X_train, X_test, y_train, y_test, save_path="outputs/")
        )

    st.success("✅ Modelo treinado com sucesso!")

    # ==========================================
    # Exibir Métricas
    # ==========================================
    st.markdown("### 📊 Métricas do Modelo")
    st.json(metrics)

    metrics_df = pd.DataFrame(metrics.items(), columns=["Métrica", "Valor"])
    metrics_df["Valor"] = metrics_df["Valor"].apply(
        lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x)
    )
    # CORREÇÃO DE AVISO: use_container_width=True -> width='stretch'
    st.dataframe(metrics_df, width="stretch")

    # ==========================================
    # Matriz de Confusão
    # ==========================================
    st.markdown("### 🔹 Matriz de Confusão")
    cm_df = pd.DataFrame(
        cm,
        index=[f"Real {cls}" for cls in modelo.classes_],
        columns=[f"Previsto {cls}" for cls in modelo.classes_],
    )
    # CORREÇÃO DE AVISO: use_container_width=True -> width='stretch'
    st.dataframe(cm_df, width="stretch")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão - XGBoost")
    st.pyplot(fig)

    # ==========================================
    # Relatório de Classificação
    # ==========================================
    st.markdown("### 📋 Relatório de Classificação")
    st.dataframe(pd.DataFrame(report).transpose())

    # ==========================================
    # Curva ROC
    # ==========================================
    if fpr is not None and tpr is not None:
        st.markdown("### 🧩 Curva ROC")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", color="#8E44AD")
        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("Curva ROC - XGBoost")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
    else:
        st.warning("⚠️ Curva ROC não disponível para múltiplas classes.")

    # ==========================================
    # Arquivos salvos
    # ==========================================
    st.markdown("### 💾 Arquivos salvos")
    st.markdown("""
    - `outputs/models/xgboost.pkl` → modelo treinado  
    - `outputs/metrics/xgboost_metrics.json` → métricas salvas  
    - `outputs/figures/roc_xgboost.png` → curva ROC  
    """)
else:
    st.info("Clique em **'🚀 Treinar Modelo XGBoost'** para iniciar o treinamento.")
