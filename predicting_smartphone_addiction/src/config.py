from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ProjectPaths:
    root_dir: Path = Path(__file__).resolve().parent.parent
    raw_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "raw")
    processed_data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "processed")
    submissions_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "submissions")
    train_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "raw" / "train.csv")
    test_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "raw" / "test.csv")
    sample_sub_path: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "raw" / "sample_submission.csv")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    n_splits: int = 5
    target_col: str = "addicted_label"
    id_col: str = "id"
    metric: str = "roc_auc"
    categorical_features: List[str] = field(
        default_factory=lambda: ["gender", "stress_level", "academic_work_impact"]
    )
    numerical_features: List[str] = field(
        default_factory=lambda: [
            "age",
            "daily_screen_time_hours",
            "social_media_hours",
            "gaming_hours",
            "work_study_hours",
            "sleep_hours",
            "notifications_per_day",
            "app_opens_per_day",
            "weekend_screen_time",
        ]
    )


PATHS = ProjectPaths()
CONFIG = ExperimentConfig()
