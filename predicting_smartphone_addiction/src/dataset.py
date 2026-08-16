from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.config import PATHS


class DatasetLoader:
    def __init__(
        self,
        train_path: Optional[Path] = None,
        test_path: Optional[Path] = None,
        sample_sub_path: Optional[Path] = None,
    ) -> None:
        self.train_path: Path = train_path or PATHS.train_path
        self.test_path: Path = test_path or PATHS.test_path
        self.sample_sub_path: Path = sample_sub_path or PATHS.sample_sub_path

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        try:
            if not self.train_path.exists():
                raise FileNotFoundError(f"Training dataset not found: {self.train_path}")
            if not self.test_path.exists():
                raise FileNotFoundError(f"Test dataset not found: {self.test_path}")

            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            sample_sub_df = pd.read_csv(self.sample_sub_path) if self.sample_sub_path.exists() else None

            return train_df, test_df, sample_sub_df
        except Exception as exc:
            raise RuntimeError(f"Error loading datasets: {exc}") from exc
