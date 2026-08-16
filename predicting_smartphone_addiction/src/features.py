from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import CONFIG


class FeaturePipeline(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        id_col: str = CONFIG.id_col,
        target_col: str = CONFIG.target_col,
        add_missing_indicators: bool = True,
    ) -> None:
        self.id_col: str = id_col
        self.target_col: str = target_col
        self.add_missing_indicators: bool = add_missing_indicators
        self.feature_names_: List[str] = []
        self.cat_freq_maps_: dict[str, dict[str, float]] = {}

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        raw_cols = [c for c in out.columns if c not in [self.id_col, self.target_col]]
        out["null_count"] = out[raw_cols].isnull().sum(axis=1).astype(np.float64)

        if self.add_missing_indicators:
            for col in raw_cols:
                if out[col].isnull().any():
                    out[f"{col}_isna"] = out[col].isnull().astype(np.float64)

        social = out["social_media_hours"]
        gaming = out["gaming_hours"]
        work = out["work_study_hours"]
        daily_screen = out["daily_screen_time_hours"]
        weekend_screen = out["weekend_screen_time"]
        sleep = out["sleep_hours"]
        opens = out["app_opens_per_day"]
        notifs = out["notifications_per_day"]

        out["breakdown_sum"] = social.fillna(0) + gaming.fillna(0) + work.fillna(0)
        out["unaccounted_screen_time"] = daily_screen - out["breakdown_sum"]
        out["breakdown_to_screen_ratio"] = out["breakdown_sum"] / (daily_screen + 1e-5)

        out["social_to_screen_ratio"] = social / (daily_screen + 1e-5)
        out["gaming_to_screen_ratio"] = gaming / (daily_screen + 1e-5)
        out["work_to_screen_ratio"] = work / (daily_screen + 1e-5)
        out["entertainment_hours"] = social.fillna(0) + gaming.fillna(0)
        out["entertainment_to_work_ratio"] = out["entertainment_hours"] / (work.fillna(0) + 1e-5)

        out["weekend_to_daily_ratio"] = weekend_screen / (daily_screen + 1e-5)
        out["weekend_daily_diff"] = weekend_screen - daily_screen

        out["waking_hours"] = 24.0 - sleep
        out["screen_fraction_of_waking_day"] = daily_screen / (out["waking_hours"] + 1e-5)
        out["non_screen_waking_hours"] = out["waking_hours"] - daily_screen

        out["time_per_open"] = (daily_screen * 60.0) / (opens + 1e-5)
        out["notifications_per_open"] = notifs / (opens + 1e-5)
        out["notifications_per_screen_hour"] = notifs / (daily_screen + 1e-5)
        out["opens_per_screen_hour"] = opens / (daily_screen + 1e-5)

        stress_numeric = out["stress_level"].map({"Low": 0.0, "Medium": 1.0, "High": 2.0})
        impact_numeric = out["academic_work_impact"].map({"No": 0.0, "Yes": 1.0})
        out["stress_numeric"] = stress_numeric
        out["academic_impact_numeric"] = impact_numeric
        out["stress_x_screen"] = stress_numeric * daily_screen
        out["impact_x_screen"] = impact_numeric * daily_screen

        return out

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeaturePipeline":
        transformed = self._engineer_features(X)
        cat_cols = [c for c in CONFIG.categorical_features if c in X.columns]
        for c in cat_cols:
            freq = X[c].value_counts(normalize=True, dropna=True).to_dict()
            self.cat_freq_maps_[c] = freq

        feature_cols = [c for c in transformed.columns if c not in [self.id_col, self.target_col]]
        self.feature_names_ = feature_cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            df_trans = self._engineer_features(X)
            for c, freq_map in self.cat_freq_maps_.items():
                if c in df_trans.columns:
                    df_trans[f"{c}_freq"] = df_trans[c].map(freq_map).fillna(0.0).astype(np.float64)

            for c in CONFIG.categorical_features:
                if c in df_trans.columns:
                    df_trans[c] = df_trans[c].astype("category")

            new_features = [c for c in df_trans.columns if c not in [self.id_col, self.target_col]]
            self.feature_names_ = new_features
            return df_trans
        except Exception as exc:
            raise RuntimeError(f"Error during feature transformation: {exc}") from exc

    def fit_transform_train_test(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        self.fit(train_df)
        train_trans = self.transform(train_df)
        test_trans = self.transform(test_df)
        return train_trans, test_trans, self.feature_names_
