
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib, os
import mlflow
import mlflow.sklearn


def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def load_data(path):
    X_train = pd.read_csv(path + "/X_train.csv")
    y_train = pd.read_csv(path + "/y_train.csv").squeeze()
    X_test = pd.read_csv(path + "/X_test.csv")
    y_test = pd.read_csv(path + "/y_test.csv").squeeze()
    return X_train, y_train, X_test, y_test

def build_model(params):
    n = params["train"]["n_estimators"]
    rs = params["train"]["random_state"]
    return RandomForestClassifier(n_estimators=n, random_state=rs)

def main():
    params = load_params()

    X_train, y_train, X_test, y_test = load_data(params["train"]["data_path"])

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("mlops_hw1_wine")

    with mlflow.start_run():
        model = build_model(params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        # Логирование параметров
        mlflow.log_param("n_estimators", params["train"]["n_estimators"])
        mlflow.log_param("random_state", params["train"]["random_state"])

        # Логирование метрик
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        print("Эксперимент завершён и записан в MLflow")
        print(f"Accuracy: {acc:.4f}")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        print("Сохранено в models/model.pkl")


if __name__ == "__main__":
    main()

