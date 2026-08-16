from pathlib import Path
import pandas as pd
from src.config import PATHS
from src.dataset import DatasetLoader


def run_quick_eda() -> None:
    loader = DatasetLoader()
    try:
        train_df, test_df, sample_sub = loader.load_raw_data()
    except Exception:
        return

    print("Train Shape:", train_df.shape)
    print("Test Shape: ", test_df.shape)
    if sample_sub is not None:
        print("Sample Sub: ", sample_sub.shape)

    train_nulls = train_df.isnull().sum()
    print("Train Nulls:\n", train_nulls[train_nulls > 0] if (train_nulls > 0).any() else "None")

    target_col = "addicted_label"
    if target_col in train_df.columns:
        print(f"Target Distribution:\n{train_df[target_col].value_counts(normalize=True)}")


if __name__ == "__main__":
    run_quick_eda()
