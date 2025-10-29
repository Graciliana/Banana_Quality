import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
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


def treinar_gradient_boosting(X_train, X_test, y_train, y_test, save_path="outputs/"):
    os.makedirs(os.path.join(save_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "figures"), exist_ok=True)

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = (
        model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else None
    )

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

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Curva ROC
    roc_fig_path = None
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="orange", lw=2, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.title("Curva ROC — Gradient Boosting")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        roc_fig_path = os.path.join(save_path, "figures", "roc_gradient_boosting.png")
        plt.savefig(roc_fig_path)
        plt.close()

    joblib.dump(model, os.path.join(save_path, "models", "gradient_boosting.pkl"))
    with open(
        os.path.join(save_path, "metrics", "gradient_boosting_metrics.json"), "w"
    ) as f:
        json.dump(metrics, f, indent=4)

    return (
        model,
        metrics,
        cm,
        report,
        y_pred,
        y_pred_proba,
        fpr if y_pred_proba is not None else None,
        tpr if y_pred_proba is not None else None,
        roc_auc if y_pred_proba is not None else None,
    )
