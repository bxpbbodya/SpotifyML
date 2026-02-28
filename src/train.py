import argparse
import os
import json
import subprocess
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"


def main(data_dir, n_estimators, max_depth, seed):

    np.random.seed(seed)

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=["popularity"])
    y_train = df_train["popularity"]

    X_test = df_test.drop(columns=["popularity"])
    y_test = df_test["popularity"]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1
    )

    mlflow.set_experiment("Spotify_Popularity_DVC")

    with mlflow.start_run():

        mlflow.set_tag("git_commit", get_git_commit())
        mlflow.set_tag("author", "student")

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("seed", seed)

        mlflow.log_metric("rmse_train", rmse_train)
        mlflow.log_metric("rmse_test", rmse_test)
        mlflow.log_metric("r2_train", r2_train)
        mlflow.log_metric("r2_test", r2_test)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        metrics = {
            "rmse_train": float(rmse_train),
            "rmse_test": float(rmse_test),
            "r2_train": float(r2_train),
            "r2_test": float(r2_test),
        }

        with open("metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        mlflow.log_artifact("metrics.json")

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="SpotifyModel"
        )

        print(metrics)

        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        # після predictions
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_test_pred.round()
        )

        os.makedirs("artifacts", exist_ok=True)
        plt.savefig("artifacts/confusion_matrix.png")
        plt.close()

        mlflow.log_artifact("artifacts/confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main(args.data_dir, args.n_estimators, args.max_depth, args.seed)

