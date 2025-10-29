# models/svm_classifier.py
import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    r2_score,
)


def treinar_svm(
    X_train,
    X_test,
    y_train,
    y_test,
    save_path="outputs/",
    kernel="rbf",
    C=1.0,
    gamma="scale",
):
    # Criar diretórios
    os.makedirs(os.path.join(save_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "figures"), exist_ok=True)

    # =====================================================
    # Treinar o modelo SVM
    # =====================================================
    model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
    model.fit(X_train, y_train)

    # =====================================================
    # Previsões
    # =====================================================
    y_pred = model.predict(X_test)
    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        y_pred_proba = (
            model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else None
        )

    # =====================================================
    # Avaliação do modelo
    # =====================================================
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "Acurácia": acc,
        "Precisão": prec,
        "Recall": rec,
        "F1-score": f1,
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
    }

    # Matriz de confusão e relatório
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # =====================================================
    # Curva ROC (apenas binária)
    # =====================================================
    roc_fig_path = None
    fpr, tpr, roc_auc = None, None, None

    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        plt.figure(figsize=(6, 5))
        plt.plot(
            fpr, tpr, color="#00CEC8", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
        )
        plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Curva ROC — SVM")
        plt.legend(loc="lower right")

        roc_fig_path = os.path.join(save_path, "figures", "roc_svm.png")
        plt.savefig(roc_fig_path)
        plt.close()

    # =====================================================
    # Salvar modelo e métricas
    # =====================================================
    joblib.dump(model, os.path.join(save_path, "models", "svm_classifier.pkl"))

    with open(
        os.path.join(save_path, "metrics", "svm_classifier_metrics.json"), "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    return model, metrics, cm, report, y_pred, y_pred_proba, fpr, tpr, roc_auc
