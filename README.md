Spotify Popularity Prediction (MLOps Lab 1)
Objective

The goal of this project is to predict song popularity using machine learning and apply MLOps principles including experiment tracking with MLflow.

Dataset

Spotify Tracks Attributes and Popularity (Kaggle)

Target variable:
popularity

Technologies

Python 3.10+

Scikit-learn

MLflow

Pandas

Matplotlib

How to run

Create virtual environment:

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows


Install dependencies:

pip install -r requirements.txt


Run experiments:

python src/train.py --n_estimators 100 --max_depth 6


Start MLflow UI:

mlflow ui


Open:

127.0.0.1:5000

Experiments

Hyperparameter tuning was performed on:

n_estimators

max_depth

Metrics logged:

RMSE (train/test)

R² (train/test)

Results

Model performance was compared using MLflow UI.
Overfitting was analyzed by comparing train and test metrics.