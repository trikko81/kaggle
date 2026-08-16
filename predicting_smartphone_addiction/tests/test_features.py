from typing import List
import numpy as np
import pandas as pd
import pytest

from src.features import FeaturePipeline


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data = {
        "id": [0, 1, 2, 3, 4],
        "gender": ["Male", "Female", np.nan, "Other", "Male"],
        "stress_level": ["High", "Low", "Medium", np.nan, "High"],
        "academic_work_impact": ["Yes", "No", "Yes", "No", np.nan],
        "age": [20.0, 25.0, 30.0, np.nan, 22.0],
        "daily_screen_time_hours": [8.0, 4.0, 10.0, 6.0, np.nan],
        "social_media_hours": [3.0, 1.0, 4.0, np.nan, 2.0],
        "gaming_hours": [2.0, 0.5, 1.5, 1.0, np.nan],
        "work_study_hours": [2.0, 2.0, 3.0, 2.0, 1.0],
        "sleep_hours": [7.0, 8.0, 6.0, 7.5, 6.5],
        "notifications_per_day": [150.0, 80.0, 200.0, 100.0, np.nan],
        "app_opens_per_day": [100.0, 50.0, 120.0, 80.0, 60.0],
        "weekend_screen_time": [10.0, 5.0, 12.0, 7.0, 8.0],
        "addicted_label": [1, 0, 1, 0, 1],
    }

    test_data = {
        "id": [5, 6],
        "gender": ["Female", "Male"],
        "stress_level": ["Low", "High"],
        "academic_work_impact": ["No", "Yes"],
        "age": [21.0, 28.0],
        "daily_screen_time_hours": [5.0, 9.0],
        "social_media_hours": [2.0, 3.5],
        "gaming_hours": [1.0, 2.0],
        "work_study_hours": [1.5, 2.5],
        "sleep_hours": [8.0, 6.0],
        "notifications_per_day": [90.0, 180.0],
        "app_opens_per_day": [60.0, 110.0],
        "weekend_screen_time": [6.0, 11.0],
    }

    return pd.DataFrame(train_data), pd.DataFrame(test_data)


def test_feature_pipeline_transformation(sample_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train_df, test_df = sample_data
    pipeline = FeaturePipeline()

    train_trans, test_trans, feature_names = pipeline.fit_transform_train_test(train_df, test_df)

    assert len(train_trans) == len(train_df)
    assert len(test_trans) == len(test_df)
    assert "addicted_label" not in feature_names
    assert "id" not in feature_names

    expected_new_cols: List[str] = [
        "breakdown_sum",
        "unaccounted_screen_time",
        "social_to_screen_ratio",
        "weekend_to_daily_ratio",
        "time_per_open",
        "notifications_per_open",
        "screen_fraction_of_waking_day",
        "null_count",
    ]
    for col in expected_new_cols:
        assert col in feature_names
        assert col in train_trans.columns
        assert col in test_trans.columns
