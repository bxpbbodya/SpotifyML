import argparse
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def main(args):

    # === Load dataset ===
    df = pd.read_csv(args.data_path)

    # === Drop non-numeric / useless columns ===
    df = df.drop(columns=[
        "index",
        "track_id",
        "artists",
        "album_name",
        "track_name"
    ], errors="ignore")

    # Encode genre
    df = pd.get_dummies(df, columns=["track_genre"], drop_first=True)

    # Convert boolean
    df["explicit"] = df["explicit"].astype(int)

    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))

    # === Split features / target ===
    X = df.drop(columns=["popularity"])
    y = df["popularity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # === Model ===
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )

    mlflow.set_experiment("Spotify_Popularity_Regression")

    with mlflow.start_run():

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        # === Log params ===
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)

        # === Log metrics ===
        mlflow.log_metric("rmse_train", rmse_train)
        mlflow.log_metric("rmse_test", rmse_test)
        mlflow.log_metric("r2_train", r2_train)
        mlflow.log_metric("r2_test", r2_test)

        # === Tags ===
        mlflow.set_tag("author", "student")
        mlflow.set_tag("model_type", "RandomForestRegressor")
        mlflow.set_tag("task", "regression")

        # === Feature Importance ===
        importances = model.feature_importances_
        fi = pd.Series(importances, index=X.columns).sort_values(ascending=False)[:15]

        plt.figure(figsize=(8,5))
        fi.plot(kind="barh")
        plt.title("Top 15 Feature Importances")
        plt.tight_layout()

        os.makedirs("artifacts", exist_ok=True)
        fi_path = "artifacts/feature_importance.png"
        plt.savefig(fi_path)
        plt.close()

        mlflow.log_artifact(fi_path)

        # === Log model ===
        mlflow.sklearn.log_model(model, "model")

        print(f"RMSE test: {rmse_test:.4f}")
        print(f"R2 test: {r2_test:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/dataset.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)

    args = parser.parse_args()
    main(args)
