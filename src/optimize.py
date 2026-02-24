import os
import random
import joblib
import mlflow
import optuna
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import hydra
from omegaconf import DictConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_data(path: str):
    df = pd.read_csv(path)

    X = df.drop(columns=["popularity"])
    y = df["popularity"]

    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )


def objective_factory(cfg, X_train, X_val, y_train, y_val):

    def objective(trial: optuna.Trial):

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                cfg.search_space.n_estimators.low,
                cfg.search_space.n_estimators.high
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                cfg.search_space.max_depth.low,
                cfg.search_space.max_depth.high
            ),
        }

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):

            mlflow.log_params(params)
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("model_type", "RandomForestRegressor")

            model = RandomForestRegressor(
                random_state=cfg.seed,
                **params
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            rmse = mean_squared_error(y_val, preds, squared=False)
            mlflow.log_metric("rmse", rmse)

            return rmse

    return objective


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):

    set_seed(cfg.seed)

    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X_train, X_val, y_train, y_val = load_data(cfg.data.train_path)

    sampler = optuna.samplers.TPESampler(seed=cfg.seed)

    with mlflow.start_run(run_name="HPO_Study"):

        study = optuna.create_study(
            direction=cfg.hpo.direction,
            sampler=sampler
        )

        objective = objective_factory(cfg, X_train, X_val, y_train, y_val)
        study.optimize(objective, n_trials=cfg.hpo.n_trials)

        best_params = study.best_trial.params
        mlflow.log_dict(best_params, "best_params.json")

        best_model = RandomForestRegressor(
            random_state=cfg.seed,
            **best_params
        )
        best_model.fit(X_train, y_train)

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_model.pkl")
        mlflow.log_artifact("models/best_model.pkl")

        mlflow.log_metric("best_rmse", study.best_value)


if __name__ == "__main__":
    main()
