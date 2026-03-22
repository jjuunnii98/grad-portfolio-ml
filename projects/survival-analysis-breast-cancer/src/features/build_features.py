from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any

import pandas as pd

from utils.config import load_config


DEFAULT_DURATION_SOURCE_COL = "Overall Survival (Months)"
DEFAULT_VITAL_STATUS_COL = "Patient's Vital Status"
DEFAULT_DURATION_COL = "time"
DEFAULT_EVENT_COL = "event"

DEFAULT_SELECTED_FEATURES = [
    "Age at Diagnosis",
    "Neoplasm Histologic Grade",
    "Tumor Stage",
    "ER Status",
    "HER2 Status",
    "Lymph nodes examined positive",
    "Tumor Size",
]

DEFAULT_CATEGORICAL_COLS = [
    "Neoplasm Histologic Grade",
    "Tumor Stage",
    "ER Status",
    "HER2 Status",
]

DEFAULT_NUMERIC_COLS = [
    "Age at Diagnosis",
    "Lymph nodes examined positive",
    "Tumor Size",
]


class FeatureEngineeringError(Exception):
    """Raised when survival feature engineering fails."""


def load_raw_clinical_data(file_path: str | Path, sep: str = "\t") -> pd.DataFrame:
    """
    Load raw METABRIC clinical data.

    Args:
        file_path: Path to raw clinical dataset.
        sep: File separator. METABRIC TSV uses tab.

    Returns:
        Loaded raw DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    df = pd.read_csv(path, sep=sep)
    if df.empty:
        raise ValueError("Loaded raw dataset is empty.")

    return df


def create_survival_targets(
    df: pd.DataFrame,
    duration_source_col: str = DEFAULT_DURATION_SOURCE_COL,
    vital_status_col: str = DEFAULT_VITAL_STATUS_COL,
    duration_col: str = DEFAULT_DURATION_COL,
    event_col: str = DEFAULT_EVENT_COL,
) -> pd.DataFrame:
    """
    Create survival duration and event columns.

    Event rule:
    - 1 if Patient's Vital Status == 'Died of Disease'
    - 0 otherwise
    """
    required_cols = [duration_source_col, vital_status_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns for target creation: {missing_cols}")

    out = df.copy()
    out[event_col] = out[vital_status_col].apply(
        lambda x: 1 if x == "Died of Disease" else 0
    )
    out[duration_col] = pd.to_numeric(out[duration_source_col], errors="coerce")

    return out


def select_modeling_features(
    df: pd.DataFrame,
    selected_features: Iterable[str] = DEFAULT_SELECTED_FEATURES,
    duration_col: str = DEFAULT_DURATION_COL,
    event_col: str = DEFAULT_EVENT_COL,
) -> pd.DataFrame:
    """
    Select core modeling variables plus survival targets.
    """
    selected_features = list(selected_features)
    required_cols = selected_features + [duration_col, event_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required feature columns: {missing_cols}")

    return df[required_cols].copy()


def clean_categorical_columns(
    df: pd.DataFrame,
    categorical_cols: Iterable[str] = DEFAULT_CATEGORICAL_COLS,
) -> pd.DataFrame:
    """
    Strip whitespace and coerce categorical feature columns to string.
    """
    out = df.copy()
    for col in categorical_cols:
        if col not in out.columns:
            raise KeyError(f"Missing categorical column: {col}")
        out[col] = out[col].astype(str).str.strip()
    return out


def clean_numeric_columns(
    df: pd.DataFrame,
    numeric_cols: Iterable[str] = DEFAULT_NUMERIC_COLS,
    duration_col: str = DEFAULT_DURATION_COL,
    event_col: str = DEFAULT_EVENT_COL,
) -> pd.DataFrame:
    """
    Convert numeric feature columns and survival targets to numeric dtype.
    """
    out = df.copy()
    cols_to_convert = list(numeric_cols) + [duration_col, event_col]

    for col in cols_to_convert:
        if col not in out.columns:
            raise KeyError(f"Missing numeric column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with any missing values in the current modeling table.
    """
    out = df.dropna().copy()
    if out.empty:
        raise FeatureEngineeringError("All rows were removed after dropna().")
    return out


def remove_rare_tumor_stage_zero(
    df: pd.DataFrame,
    tumor_stage_col: str = "Tumor Stage",
) -> pd.DataFrame:
    """
    Remove clinically ambiguous rare Tumor Stage == '0.0' rows.
    """
    if tumor_stage_col not in df.columns:
        raise KeyError(f"Missing tumor stage column: {tumor_stage_col}")

    out = df[df[tumor_stage_col] != "0.0"].copy()
    if out.empty:
        raise FeatureEngineeringError(
            "All rows were removed when filtering Tumor Stage == '0.0'."
        )
    return out


def one_hot_encode_features(
    df: pd.DataFrame,
    categorical_cols: Iterable[str] = DEFAULT_CATEGORICAL_COLS,
) -> pd.DataFrame:
    """
    One-hot encode categorical features with drop_first=True.
    """
    missing_cols = [col for col in categorical_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing categorical columns for encoding: {missing_cols}")

    return pd.get_dummies(
        df,
        columns=list(categorical_cols),
        drop_first=True,
        dtype=int,
    )


def reorder_final_columns(
    df: pd.DataFrame,
    duration_col: str = DEFAULT_DURATION_COL,
    event_col: str = DEFAULT_EVENT_COL,
) -> pd.DataFrame:
    """
    Move survival target columns to the end for modeling clarity.
    """
    required_cols = [duration_col, event_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing final target columns: {missing_cols}")

    feature_cols = [col for col in df.columns if col not in required_cols]
    return df[feature_cols + required_cols].copy()


def build_cox_ready_dataset(
    raw_df: pd.DataFrame,
    selected_features: Iterable[str] = DEFAULT_SELECTED_FEATURES,
    categorical_cols: Iterable[str] = DEFAULT_CATEGORICAL_COLS,
    numeric_cols: Iterable[str] = DEFAULT_NUMERIC_COLS,
    remove_stage_zero: bool = True,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline for Cox-ready METABRIC clinical data.

    Pipeline:
    1. Create survival targets
    2. Select modeling columns
    3. Drop missing rows
    4. Clean categorical columns
    5. Remove rare Tumor Stage == '0.0' if configured
    6. Clean numeric columns
    7. Drop missing rows again after numeric coercion
    8. One-hot encode categorical columns
    9. Reorder columns so time/event are at the end
    """
    df = create_survival_targets(raw_df)
    df = select_modeling_features(df, selected_features=selected_features)
    df = drop_missing_rows(df)
    df = clean_categorical_columns(df, categorical_cols=categorical_cols)

    if remove_stage_zero:
        df = remove_rare_tumor_stage_zero(df)

    df = clean_numeric_columns(
        df,
        numeric_cols=numeric_cols,
    )
    df = drop_missing_rows(df)
    df = one_hot_encode_features(df, categorical_cols=categorical_cols)
    df = reorder_final_columns(df)

    return df


def save_featurized_dataset(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Save featurized Cox-ready dataset to CSV.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def resolve_project_paths(config: dict[str, Any], config_path: str | Path) -> tuple[Path, Path]:
    """
    Resolve raw and processed data paths relative to the project root.

    Args:
        config: Parsed YAML config.
        config_path: Path to config.yaml.

    Returns:
        Tuple of (raw_data_path, processed_data_path).
    """
    config_file = Path(config_path).resolve()
    project_root = config_file.parent.parent

    raw_data_path = project_root / config["data"]["raw_data_path"]
    processed_data_path = project_root / config["data"]["processed_data_path"]

    return raw_data_path, processed_data_path


def run_feature_pipeline_from_config(
    config_path: str | Path,
) -> pd.DataFrame:
    """
    End-to-end feature engineering pipeline driven by config.yaml.

    Args:
        config_path: Path to YAML config.

    Returns:
        Final featurized DataFrame.
    """
    config = load_config(config_path)
    raw_data_path, processed_data_path = resolve_project_paths(config, config_path)

    selected_features = config["features"]["selected_features"]
    categorical_cols = config["features"]["categorical_cols"]
    numeric_cols = config["features"]["numeric_cols"]
    remove_stage_zero = config["features"]["remove_rare_stage_zero"]

    raw_df = load_raw_clinical_data(raw_data_path)
    featurized_df = build_cox_ready_dataset(
        raw_df=raw_df,
        selected_features=selected_features,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        remove_stage_zero=remove_stage_zero,
    )
    save_featurized_dataset(featurized_df, processed_data_path)

    return featurized_df


def run_feature_pipeline(
    raw_file_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Backward-compatible end-to-end feature engineering pipeline:
    raw TSV -> Cox-ready CSV.

    Returns:
        Final featurized DataFrame.
    """
    raw_df = load_raw_clinical_data(raw_file_path)
    featurized_df = build_cox_ready_dataset(raw_df)
    save_featurized_dataset(featurized_df, output_path)
    return featurized_df