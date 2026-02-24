import os
import pandas as pd


def test_data_schema_basic():
    data_path = os.getenv("DATA_PATH", "data/prepared/train.csv")
    assert os.path.exists(data_path), f"Data not found: {data_path}"

    df = pd.read_csv(data_path)

    # Перевірка наявності target
    required_cols = {"popularity"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

    # Немає пропусків у target
    assert df["popularity"].notna().all(), "Target has missing values"

    # Мінімальний розмір вибірки
    assert df.shape[0] >= 100, "Dataset too small"
