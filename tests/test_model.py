# tests/test_model.py
import os
import pandas as pd

def test_data_schema_basic():
    data_path = os.getenv("DATA_PATH", "data/train.csv")
    assert os.path.exists(data_path), f"Data not found: {data_path}"

    df = pd.read_csv(data_path)

    required_cols = {"target"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    assert df["target"].notna().all(), "Target has missing values"
    assert df.shape[0] >= 100, "Dataset too small"