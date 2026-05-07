import mlflow
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

EVAL_THRESHOLD = 0.70

# Bonus 2: params valid cho từng loại model
_VALID_PARAMS = {
    "random_forest":      {"n_estimators", "max_depth", "min_samples_split",
                           "class_weight", "max_features", "min_samples_leaf", "bootstrap"},
    "gradient_boosting":  {"n_estimators", "max_depth", "min_samples_split",
                           "learning_rate", "subsample"},
    "logistic_regression":{"C", "max_iter", "class_weight", "solver"},
}


def _build_model(model_type: str, params: dict):
    valid = _VALID_PARAMS.get(model_type, set())
    filtered = {k: v for k, v in params.items() if k in valid}
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(random_state=42, **filtered)
    if model_type == "logistic_regression":
        return LogisticRegression(random_state=42, **filtered)
    return RandomForestClassifier(random_state=42, **filtered)


def _check_data_drift(y_train: pd.Series) -> dict:
    # Bonus 5: cảnh báo nếu lớp nào < 10%
    counts = y_train.value_counts()
    total = len(y_train)
    dist = {}
    for cls in sorted(counts.index):
        ratio = counts[cls] / total
        dist[str(cls)] = round(ratio, 4)
        if ratio < 0.10:
            print(f"WARNING: Class {cls} chi chiem {ratio:.1%} tap train (<10%)")
    return dist


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    params = dict(params)
    # Bonus 2: lay model_type, mac dinh random_forest
    model_type = str(params.pop("model_type", "random_forest"))

    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    # Bonus 5: kiem tra phan phoi nhan
    class_dist = _check_data_drift(y_train)

    with mlflow.start_run():
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)

        # Bonus 2: chon thuat toan theo model_type
        model = _build_model(model_type, params)
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        os.makedirs("outputs", exist_ok=True)

        # Bonus 5: them class_distribution vao metrics.json
        with open("outputs/metrics.json", "w") as f:
            json.dump({
                "accuracy": acc,
                "f1_score": f1,
                "class_distribution": class_dist,
            }, f, indent=2)

        # Bonus 3: bao cao hieu suat chi tiet
        cm     = confusion_matrix(y_eval, preds)
        report = classification_report(
            y_eval, preds, target_names=["thap", "trung_binh", "cao"]
        )
        with open("outputs/report.txt", "w", encoding="utf-8") as f:
            f.write(f"Model     : {model_type}\n")
            f.write(f"Accuracy  : {acc:.4f}\n")
            f.write(f"F1 Score  : {f1:.4f}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n\n")
            f.write("Classification Report:\n")
            f.write(report + "\n")
            f.write("Class Distribution (train):\n")
            for cls, ratio in class_dist.items():
                f.write(f"  Class {cls}: {float(ratio):.1%}\n")
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
