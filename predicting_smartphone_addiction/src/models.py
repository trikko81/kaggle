from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import CONFIG


class PyTorchTabularMLP(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_dims: List[int] | None = None,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 512,
        epochs: int = 10,
        random_state: int = CONFIG.seed,
    ) -> None:
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.classes_ = np.array([0, 1])

        self.num_imputer_ = SimpleImputer(strategy="median")
        self.scaler_ = StandardScaler()
        self.model_: Optional[nn.Module] = None
        self.num_cols_: List[str] = []
        self.cat_cols_: List[str] = []
        self.cat_maps_: Dict[str, Dict[Any, int]] = {}

    def _prepare_inputs(self, X: pd.DataFrame, is_train: bool = False) -> torch.Tensor:
        df = X.copy()
        if is_train:
            self.cat_cols_ = [c for c in df.columns if df[c].dtype.name in ["category", "object"]]
            self.num_cols_ = [c for c in df.columns if c not in self.cat_cols_]

            for c in self.cat_cols_:
                uniques = df[c].dropna().unique().tolist()
                mapping = {val: idx + 1 for idx, val in enumerate(uniques)}
                self.cat_maps_[c] = mapping

        num_data = df[self.num_cols_].to_numpy(dtype=np.float32)
        if is_train:
            num_data = self.num_imputer_.fit_transform(num_data)
            num_scaled = self.scaler_.fit_transform(num_data)
        else:
            num_data = self.num_imputer_.transform(num_data)
            num_scaled = self.scaler_.transform(num_data)

        tensors = [torch.tensor(num_scaled, dtype=torch.float32)]

        for c in self.cat_cols_:
            mapping = self.cat_maps_[c]
            raw_vals = df[c].astype(object).map(mapping).to_numpy()
            encoded = np.nan_to_num(raw_vals, nan=0.0).astype(np.float32)
            tensors.append(torch.tensor(encoded, dtype=torch.float32).unsqueeze(1))

        return torch.cat(tensors, dim=1)

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "PyTorchTabularMLP":
        torch.manual_seed(self.random_state)
        y_arr = np.asarray(y, dtype=np.float32)

        X_tensor = self._prepare_inputs(X, is_train=True)
        y_tensor = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        in_dim = X_tensor.shape[1]
        layers: List[nn.Module] = []
        curr_dim = in_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(self.dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, 1))

        self.model_ = nn.Sequential(*layers)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        self.model_.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self.model_(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted.")
        self.model_.eval()
        X_tensor = self._prepare_inputs(X, is_train=False)

        with torch.no_grad():
            logits = self.model_(X_tensor).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

        probs_pos = np.clip(probs, 0.0, 1.0)
        return np.column_stack([1.0 - probs_pos, probs_pos])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **params: Any) -> None:
        self.params = params
        self.model_: Optional[Any] = None
        self.classes_ = np.array([0, 1])

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series, **fit_params: Any) -> "CatBoostWrapper":
        import catboost as cb
        cat_features = [c for c in X.columns if X[c].dtype.name in ["category", "object"]]
        X_copy = X.copy()
        for c in cat_features:
            X_copy[c] = X_copy[c].astype(str).fillna("missing")

        defaults = {"random_state": CONFIG.seed, "iterations": 1000, "verbose": 0}
        combined_params = {**defaults, **self.params}
        self.model_ = cb.CatBoostClassifier(**combined_params)
        self.model_.fit(X_copy, y, cat_features=cat_features, **fit_params)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        X_copy = X.copy()
        for c in X_copy.columns:
            if X_copy[c].dtype.name in ["category", "object"]:
                X_copy[c] = X_copy[c].astype(str).fillna("missing")
        return self.model_.predict_proba(X_copy)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


class ModelFactory:
    @staticmethod
    def get_model(model_name: str, params: Dict[str, Any] | None = None) -> BaseEstimator:
        params = params or {}
        name = model_name.lower()

        if name == "hist_gb":
            return HistGradientBoostingClassifier(random_state=CONFIG.seed, **params)
        elif name == "rf":
            return RandomForestClassifier(random_state=CONFIG.seed, **params)
        elif name == "lightgbm":
            try:
                import lightgbm as lgb
                defaults = {
                    "random_state": CONFIG.seed,
                    "n_estimators": 1000,
                    "learning_rate": 0.03,
                    "max_depth": 7,
                    "num_leaves": 63,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "verbose": -1,
                }
                combined = {**defaults, **params}
                return lgb.LGBMClassifier(**combined)
            except ImportError as exc:
                raise ImportError("lightgbm is not installed.") from exc
        elif name == "xgboost":
            try:
                import xgboost as xgb
                defaults = {
                    "random_state": CONFIG.seed,
                    "n_estimators": 1000,
                    "learning_rate": 0.03,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "eval_metric": "auc",
                    "tree_method": "hist",
                    "enable_categorical": True,
                }
                combined = {**defaults, **params}
                return xgb.XGBClassifier(**combined)
            except ImportError as exc:
                raise ImportError("xgboost is not installed.") from exc
        elif name == "catboost":
            try:
                return CatBoostWrapper(**params)
            except ImportError as exc:
                raise ImportError("catboost is not installed.") from exc
        elif name in ["tabular_mlp", "mlp"]:
            return PyTorchTabularMLP(**params)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
