from __future__ import annotations

import pickle
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
    """
    performance_path = Path(model_performance_path)
    coefficient_path = Path(coefficient_comparison_path)

    performance_path.parent.mkdir(parents=True, exist_ok=True)
    coefficient_path.parent.mkdir(parents=True, exist_ok=True)

    model_performance.to_csv(performance_path, index=False)
    comparison_df.to_csv(coefficient_path, index=False)

    return performance_path, coefficient_path


def save_pickle_artifact(obj: Any, artifact_path: str | Path) -> Path:
    """
    Save a Python object as a pickle artifact.
    """
    path = Path(artifact_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)

    return path


def load_pickle_artifact(artifact_path: str | Path) -> Any:
    """
    Load a Python object from a pickle artifact.
    """
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def build_inference_bundle(
    df: pd.DataFrame,
    model: CoxPHFitter,
    duration_col: str = DEFAULT_DURATION_COL,
    event_col: str = DEFAULT_EVENT_COL,
) -> dict[str, Any]:
    """
    Build a production inference bundle containing the trained model,
    feature columns, and precomputed risk cutoffs.
    """
    feature_columns = [col for col in df.columns if col not in [duration_col, event_col]]
    feature_df = df[feature_columns].copy()
    train_scores = model.predict_partial_hazard(feature_df).astype(float)

    low_cutoff = float(train_scores.quantile(0.33))
    high_cutoff = float(train_scores.quantile(0.67))

    return {
        "model": model,
        "feature_columns": feature_columns,
        "low_cutoff": low_cutoff,
        "high_cutoff": high_cutoff,
        "duration_col": duration_col,
        "event_col": event_col,
        "training_rows": int(df.shape[0]),
    }


def resolve_project_paths(
    config: dict[str, Any],
    config_path: str | Path,
) -> tuple[Path, Path, Path, Path]:
    """
    Resolve processed data and output/artifact paths relative to the project root.
    """
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent

    processed_data_path = project_root / config["data"]["processed_data_path"]
    model_performance_path = project_root / config["output"]["model_performance_path"]
    coefficient_comparison_path = (
        project_root / config["output"]["coefficient_comparison_path"]
    )
    model_artifact_path = project_root / config["output"]["model_artifact_path"]

    return (
        processed_data_path,
        model_performance_path,
        coefficient_comparison_path,
        model_artifact_path,
    )


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


def train_final_penalized_model(
    file_path: str | Path,
    penalizer: float = 0.1,
    duration_col: str = DEFAULT_DURATION_COL,
    event_col: str = DEFAULT_EVENT_COL,
) -> CoxPHFitter:
    """
    Train the final penalized Cox model on the full processed dataset.
    """
    df = load_featurized_data(file_path)
    model = fit_penalized_cox(
        df,
        penalizer=penalizer,
        duration_col=duration_col,
        event_col=event_col,
    )
    return model


def train_and_save_inference_bundle_from_config(
    config_path: str | Path,
) -> Path:
    """
    Train the final penalized Cox model on full data and save a production
    inference bundle as a pickle artifact.
    """
    config = load_config(config_path)
    (
        processed_data_path,
        _,
        _,
        model_artifact_path,
    ) = resolve_project_paths(config, config_path)

    duration_col = config["target"]["duration_col"]
    event_col = config["target"]["event_col"]
    penalized_penalizer = config["model"]["penalized"]["penalizer"]

    df = load_featurized_data(processed_data_path)
    model = fit_penalized_cox(
        df,
        penalizer=penalized_penalizer,
        duration_col=duration_col,
        event_col=event_col,
    )
    inference_bundle = build_inference_bundle(
        df=df,
        model=model,
        duration_col=duration_col,
        event_col=event_col,
    )

    save_pickle_artifact(inference_bundle, model_artifact_path)
    return model_artifact_path


def run_refinement_pipeline_from_config(
    config_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end refinement pipeline driven by config.yaml.
    """
    config = load_config(config_path)
    (
        processed_data_path,
        model_performance_path,
        coefficient_comparison_path,
        _,
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