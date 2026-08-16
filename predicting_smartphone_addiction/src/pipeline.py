from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import CONFIG, PATHS
from src.dataset import DatasetLoader
from src.features import FeaturePipeline
from src.models import ModelFactory


class CrossValidationPipeline:
    def __init__(
        self,
        model_name: str = "hist_gb",
        model_params: Dict[str, Any] | None = None,
        n_splits: int = CONFIG.n_splits,
        seed: int = CONFIG.seed,
    ) -> None:
        self.model_name: str = model_name
        self.model_params: Dict[str, Any] = model_params or {}
        self.n_splits: int = n_splits
        self.seed: int = seed

    def run_cv(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = CONFIG.target_col,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        oof_preds = np.zeros(len(train_df))
        test_preds = np.zeros(len(test_df))

        X = train_df[feature_cols].copy()
        y = train_df[target_col].to_numpy()
        X_test = test_df[feature_cols].copy()

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            model = ModelFactory.get_model(self.model_name, self.model_params)
            model.fit(X_train, y_train)

            val_probs = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_probs

            fold_auc = roc_auc_score(y_val, val_probs)
            test_preds += model.predict_proba(X_test)[:, 1] / self.n_splits

        overall_auc = roc_auc_score(y, oof_preds)
        return oof_preds, test_preds, float(overall_auc)

    def generate_submission(
        self,
        test_df: pd.DataFrame,
        test_preds: np.ndarray,
        filename: str = "submission.csv",
        id_col: str = CONFIG.id_col,
        target_col: str = CONFIG.target_col,
    ) -> Path:
        PATHS.submissions_dir.mkdir(parents=True, exist_ok=True)
        sub_path = PATHS.submissions_dir / filename

        submission_df = pd.DataFrame({
            id_col: test_df[id_col],
            target_col: test_preds,
        })
        submission_df.to_csv(sub_path, index=False)
        return sub_path


def main() -> None:
    loader = DatasetLoader()
    try:
        train_df, test_df, _ = loader.load_raw_data()
    except (FileNotFoundError, RuntimeError) as exc:
        return

    feature_pipeline = FeaturePipeline()
    train_feat, test_feat, feature_names = feature_pipeline.fit_transform_train_test(train_df, test_df)

    pipeline = CrossValidationPipeline(model_name="lightgbm")
    _, test_preds, _ = pipeline.run_cv(train_feat, test_feat, feature_names)
    pipeline.generate_submission(test_df, test_preds)


if __name__ == "__main__":
    main()
