from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split

from src.utils.config import load_config


DEFAULT_RANDOM_STATE = 42
DEFAULT_DURATION_COL = "time"
DEFAULT_EVENT_COL = "event"


@dataclass
class SurvivalSplit:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame


def load_featurized_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load the Cox-ready featurized survival dataset.

    Args:
        file_path: Path to the processed CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    return df


def split_survival_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
    event_col: str = DEFAULT_EVENT_COL,
) -> SurvivalSplit:
    """
    Split survival data into train and validation sets with event stratification.

    Args:
        df: Cox-ready dataset.
        test_size: Validation ratio.
        random_state: Seed for reproducibility.
        event_col: Event indicator column.

    Returns:
        SurvivalSplit: Dataclass containing train and validation DataFrames.
    """
    if event_col not in df.columns:
        raise KeyError(f"Missing required event column: {event_col}")

    train_df, valid_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[event_col],
    )

    return SurvivalSplit(train_df=train_df, valid_df=valid_df)


def fit_baseline_cox(
    train_df: pd.DataFrame,
    duration_col: str = DEFAULT_DURATION_COL,
    event_col: str = DEFAULT_EVENT_COL,
    penalizer: float = 0.0,
) -> CoxPHFitter:
    """
    Fit a baseline Cox Proportional Hazards model.

    Args:
        train_df: Training dataset.
        duration_col: Survival duration column.
        event_col: Event indicator column.
        penalizer: Penalizer value for baseline model.

    Returns:
        CoxPHFitter: Fitted Cox model.
    """
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(train_df, duration_col=duration_col, event_col=event_col)
    return cph


def fit_penalized_cox(
    train_df: pd.DataFrame,
    penalizer: float = 0.1,
    duration_col: str = DEFAULT_DURATION_COL,
    event_col: str = DEFAULT_EVENT_COL,
) -> CoxPHFitter:
    """
    Fit a penalized Cox model.

    Args:
        train_df: Training dataset.
        penalizer: Regularization strength.
        duration_col: Survival duration column.
        event_col: Event indicator column.

    Returns:
        CoxPHFitter: Fitted penalized Cox model.
    """
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(train_df, duration_col=duration_col, event_col=event_col)
    return cph


def evaluate_concordance(
    model: CoxPHFitter,
    valid_df: pd.DataFrame,
) -> float:
    """
    Evaluate validation concordance index.

    Args:
        model: Fitted Cox model.
        valid_df: Validation dataset.

    Returns:
        float: Validation concordance index.
    """
    return float(model.score(valid_df, scoring_method="concordance_index"))


def build_model_performance_table(
    baseline_model: CoxPHFitter,
    penalized_model: CoxPHFitter,
    baseline_valid_cindex: float,
    penalized_valid_cindex: float,
) -> pd.DataFrame:
    """
    Build a comparison table for baseline vs penalized Cox models.

    Args:
        baseline_model: Fitted baseline Cox model.
        penalized_model: Fitted penalized Cox model.
        baseline_valid_cindex: Validation concordance for baseline model.
        penalized_valid_cindex: Validation concordance for penalized model.

    Returns:
        pd.DataFrame: Sorted performance comparison table.
    """
    model_performance = pd.DataFrame(
        {
            "model": ["baseline_cox", "penalized_cox"],
            "train_cindex": [
                baseline_model.concordance_index_,
                penalized_model.concordance_index_,
            ],
            "valid_cindex": [
                baseline_valid_cindex,
                penalized_valid_cindex,
            ],
        }
    )

    model_performance["overfitting_gap"] = (
        model_performance["train_cindex"] - model_performance["valid_cindex"]
    )

    model_performance = model_performance.sort_values(
        by="valid_cindex",
        ascending=False,
    ).reset_index(drop=True)

    return model_performance


def get_best_model_row(model_performance: pd.DataFrame) -> pd.Series:
    """
    Select the best model based on validation concordance.

    Args:
        model_performance: Model comparison table.

    Returns:
        pd.Series: Best model row.
    """
    if model_performance.empty:
        raise ValueError("Model performance table is empty.")

    return model_performance.loc[model_performance["valid_cindex"].idxmax()]


def build_coefficient_comparison(
    baseline_model: CoxPHFitter,
    penalized_model: CoxPHFitter,
) -> pd.DataFrame:
    """
    Build coefficient comparison table between baseline and penalized Cox models.

    Args:
        baseline_model: Fitted baseline Cox model.
        penalized_model: Fitted penalized Cox model.

    Returns:
        pd.DataFrame: Feature-wise coefficient comparison.
    """
    baseline_summary = baseline_model.summary
    penalized_summary = penalized_model.summary.reindex(baseline_summary.index)

    comparison_df = pd.DataFrame(
        {
            "feature": baseline_summary.index,
            "baseline_coef": baseline_summary["coef"].values,
            "baseline_hr": baseline_summary["exp(coef)"].values,
            "penalized_coef": penalized_summary["coef"].values,
            "penalized_hr": penalized_summary["exp(coef)"].values,
        }
    )

    return comparison_df


def save_model_outputs(
    model_performance: pd.DataFrame,
    comparison_df: pd.DataFrame,
    model_performance_path: str | Path,
    coefficient_comparison_path: str | Path,
) -> tuple[Path, Path]:
    """
    Save model refinement outputs to CSV.

    Args:
        model_performance: Comparison table for model performance.
        comparison_df: Coefficient comparison table.
        model_performance_path: Output path for performance CSV.
        coefficient_comparison_path: Output path for coefficient CSV.

    Returns:
        tuple[Path, Path]: Saved file paths.
    """
    performance_path = Path(model_performance_path)
    coefficient_path = Path(coefficient_comparison_path)

    performance_path.parent.mkdir(parents=True, exist_ok=True)
    coefficient_path.parent.mkdir(parents=True, exist_ok=True)

    model_performance.to_csv(performance_path, index=False)
    comparison_df.to_csv(coefficient_path, index=False)

    return performance_path, coefficient_path


def resolve_project_paths(
    config: dict[str, Any],
    config_path: str | Path,
) -> tuple[Path, Path, Path]:
    """
    Resolve processed data and output paths relative to the project root.

    Args:
        config: Parsed YAML config.
        config_path: Path to config.yaml.

    Returns:
        tuple[Path, Path, Path]:
            - processed data path
            - model performance output path
            - coefficient comparison output path
    """
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent

    processed_data_path = project_root / config["data"]["processed_data_path"]
    model_performance_path = project_root / config["output"]["model_performance_path"]
    coefficient_comparison_path = (
        project_root / config["output"]["coefficient_comparison_path"]
    )

    return processed_data_path, model_performance_path, coefficient_comparison_path


def run_refinement_pipeline(
    file_path: str | Path,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
    penalizer: float = 0.1,
    duration_col: str = DEFAULT_DURATION_COL,
    event_col: str = DEFAULT_EVENT_COL,
    baseline_penalizer: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end refinement pipeline.

    Flow:
    load data -> split -> fit baseline/penalized -> compare models.

    Args:
        file_path: Path to featurized dataset.
        test_size: Validation split ratio.
        random_state: Random seed.
        penalizer: Penalizer strength for penalized Cox model.
        duration_col: Survival duration column.
        event_col: Event indicator column.
        baseline_penalizer: Penalizer for baseline Cox model.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - model_performance table
            - coefficient comparison table
    """
    df = load_featurized_data(file_path)
    split_data = split_survival_data(
        df=df,
        test_size=test_size,
        random_state=random_state,
        event_col=event_col,
    )

    baseline_model = fit_baseline_cox(
        split_data.train_df,
        duration_col=duration_col,
        event_col=event_col,
        penalizer=baseline_penalizer,
    )
    penalized_model = fit_penalized_cox(
        split_data.train_df,
        penalizer=penalizer,
        duration_col=duration_col,
        event_col=event_col,
    )

    baseline_valid_cindex = evaluate_concordance(
        baseline_model,
        split_data.valid_df,
    )
    penalized_valid_cindex = evaluate_concordance(
        penalized_model,
        split_data.valid_df,
    )

    model_performance = build_model_performance_table(
        baseline_model=baseline_model,
        penalized_model=penalized_model,
        baseline_valid_cindex=baseline_valid_cindex,
        penalized_valid_cindex=penalized_valid_cindex,
    )

    comparison_df = build_coefficient_comparison(
        baseline_model=baseline_model,
        penalized_model=penalized_model,
    )

    return model_performance, comparison_df


def run_refinement_pipeline_from_config(
    config_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end refinement pipeline driven by config.yaml.

    Args:
        config_path: Path to YAML config.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - model_performance table
            - coefficient comparison table
    """
    config = load_config(config_path)
    (
        processed_data_path,
        model_performance_path,
        coefficient_comparison_path,
    ) = resolve_project_paths(config, config_path)

    duration_col = config["target"]["duration_col"]
    event_col = config["target"]["event_col"]
    test_size = config["split"]["test_size"]
    random_state = config["split"]["random_state"]
    baseline_penalizer = config["model"]["baseline"]["penalizer"]
    penalized_penalizer = config["model"]["penalized"]["penalizer"]

    model_performance, comparison_df = run_refinement_pipeline(
        file_path=processed_data_path,
        test_size=test_size,
        random_state=random_state,
        penalizer=penalized_penalizer,
        duration_col=duration_col,
        event_col=event_col,
        baseline_penalizer=baseline_penalizer,
    )

    save_model_outputs(
        model_performance=model_performance,
        comparison_df=comparison_df,
        model_performance_path=model_performance_path,
        coefficient_comparison_path=coefficient_comparison_path,
    )

    return model_performance, comparison_df