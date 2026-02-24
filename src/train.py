import argparse
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def main(data_dir, n_estimators, max_depth):

    # === Load prepared data ===
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    print("Training started...")

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=["popularity"])
    y_train = df_train["popularity"]

    X_test = df_test.drop(columns=["popularity"])
    y_test = df_test["popularity"]

    # === Model ===
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )

    mlflow.set_experiment("Spotify_Popularity_DVC")

    with mlflow.start_run():

        # === Train ===
        model.fit(X_train, y_train)

        # === Predictions ===
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # === Metrics ===
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        # === Log params ===
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # === Log metrics ===
        mlflow.log_metric("rmse_train", rmse_train)
        mlflow.log_metric("rmse_test", rmse_test)
        mlflow.log_metric("r2_train", r2_train)
        mlflow.log_metric("r2_test", r2_test)

        # === Tags ===
        mlflow.set_tag("author", "student")
        mlflow.set_tag("model_type", "RandomForestRegressor")
        mlflow.set_tag("pipeline_stage", "train")

        # === Feature Importance ===
        importances = model.feature_importances_
        fi = pd.Series(importances, index=X_train.columns)\
               .sort_values(ascending=False)[:15]

        os.makedirs("artifacts", exist_ok=True)
        fi_path = "artifacts/feature_importance.png"

        plt.figure(figsize=(8, 5))
        fi.plot(kind="barh")
        plt.title("Top 15 Feature Importances")
        plt.tight_layout()
        plt.savefig(fi_path)
        plt.close()

        mlflow.log_artifact(fi_path)

        import json

        # === Save model locally ===
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pkl"

        import joblib
        joblib.dump(model, model_path)

        # === Save metrics.json ===
        metrics = {
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "r2_train": r2_train,
            "r2_test": r2_test
        }

        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # === Log model ===
        mlflow.sklearn.log_model(model, "model")

        print(f"RMSE train: {rmse_train:.4f}")
        print(f"RMSE test: {rmse_test:.4f}")
        print(f"R2 train: {r2_train:.4f}")
        print(f"R2 test: {r2_test:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to prepared data directory")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)

    args = parser.parse_args()

    main(args.data_dir, args.n_estimators, args.max_depth)
