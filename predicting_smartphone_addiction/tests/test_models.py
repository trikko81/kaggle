from typing import Any, Dict
import numpy as np
import pandas as pd
import pytest

from src.models import ModelFactory


@pytest.fixture
def dummy_train_data() -> tuple[pd.DataFrame, np.ndarray]:
    np.random.seed(42)
    n_samples = 100
    df = pd.DataFrame({
        "num_1": np.random.randn(n_samples),
        "num_2": np.random.rand(n_samples) * 10,
        "cat_1": pd.Categorical(np.random.choice(["A", "B", "C"], size=n_samples)),
        "cat_2": pd.Categorical(np.random.choice(["Low", "High"], size=n_samples)),
    })
    y = np.random.choice([0, 1], size=n_samples)
    return df, y


@pytest.mark.parametrize("model_name", ["lightgbm", "xgboost", "catboost", "tabular_mlp"])
def test_model_zoo_fit_and_predict(
    model_name: str, dummy_train_data: tuple[pd.DataFrame, np.ndarray]
) -> None:
    X, y = dummy_train_data
    model = ModelFactory.get_model(model_name)
    model.fit(X, y)
    preds = model.predict_proba(X)

    assert preds.shape == (len(X), 2)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)
    assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-4)
