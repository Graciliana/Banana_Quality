import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
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


def treinar_xgboost(X_train, X_test, y_train, y_test, save_path="outputs/"):
    os.makedirs(os.path.join(save_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "figures"), exist_ok=True)

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
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

    roc_fig_path = None
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="red", lw=2, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
        plt.title("Curva ROC — XGBoost")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        roc_fig_path = os.path.join(save_path, "figures", "roc_xgboost.png")
        plt.savefig(roc_fig_path)
        plt.close()

    joblib.dump(model, os.path.join(save_path, "models", "xgboost_classifier.pkl"))
    with open(os.path.join(save_path, "metrics", "xgboost_metrics.json"), "w") as f:
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
