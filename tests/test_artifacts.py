import os
import json


def test_model_exists():
    assert os.path.exists("models/model.pkl"), "Model not saved"


def test_metrics_exists():
    assert os.path.exists("metrics.json"), "metrics.json not found"


def test_quality_gate():
    with open("metrics.json") as f:
        metrics = json.load(f)

    # Quality Gate для регресії
    assert metrics["r2_test"] >= 0.5, "Quality Gate failed"
