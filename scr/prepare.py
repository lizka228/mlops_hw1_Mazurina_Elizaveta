
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    params = load_params()
    test_size = params["prepare"]["test_size"]
    random_state = params["prepare"]["random_state"]

    df = pd.read_csv("data/raw/wine.csv")
    df = df.dropna(axis=0)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    os.makedirs("data/prepared", exist_ok=True)
    X_train.to_csv("data/prepared/X_train.csv", index=False)
    X_test.to_csv("data/prepared/X_test.csv", index=False)
    y_train.to_csv("data/prepared/y_train.csv", index=False)
    y_test.to_csv("data/prepared/y_test.csv", index=False)

    print("Обработанные данные сохранены в data/prepared")

if __name__ == "__main__":
    main()

