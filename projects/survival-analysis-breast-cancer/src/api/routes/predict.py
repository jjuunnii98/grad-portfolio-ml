from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import RiskPredictionRequest, RiskPredictionResponse
from src.models.survival_model import fit_penalized_cox, load_featurized_data
from src.utils.config import load_config

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def get_inference_bundle() -> dict:
    """
    Load config, train the penalized Cox model once, and cache inference assets.
    """
    config = load_config(CONFIG_PATH)

    processed_data_path = PROJECT_ROOT / config["data"]["processed_data_path"]
    duration_col = config["target"]["duration_col"]
    event_col = config["target"]["event_col"]
    penalizer = config["model"]["penalized"]["penalizer"]

    df = load_featurized_data(processed_data_path)
    model = fit_penalized_cox(
        df,
        penalizer=penalizer,
        duration_col=duration_col,
        event_col=event_col,
    )

    feature_columns = [col for col in df.columns if col not in [duration_col, event_col]]
    train_features = df[feature_columns].copy()
    train_scores = model.predict_partial_hazard(train_features).astype(float)

    low_cutoff = float(train_scores.quantile(0.33))
    high_cutoff = float(train_scores.quantile(0.67))

    return {
        "config": config,
        "model": model,
        "feature_columns": feature_columns,
        "train_scores": train_scores,
        "low_cutoff": low_cutoff,
        "high_cutoff": high_cutoff,
    }


def build_feature_frame(
    payload: RiskPredictionRequest,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Convert API payload into a single-row Cox-ready feature frame.
    """
    row = {col: 0.0 for col in feature_columns}

    row.update(
        {
            "Age at Diagnosis": float(payload.age_at_diagnosis),
            "Lymph nodes examined positive": float(
                payload.lymph_nodes_examined_positive
            ),
            "Tumor Size": float(payload.tumor_size),
            "Neoplasm Histologic Grade_2.0": float(payload.grade_2),
            "Neoplasm Histologic Grade_3.0": float(payload.grade_3),
            "Tumor Stage_2.0": float(payload.tumor_stage_2),
            "Tumor Stage_3.0": float(payload.tumor_stage_3),
            "Tumor Stage_4.0": float(payload.tumor_stage_4),
            "ER Status_Positive": float(payload.er_status_positive),
            "HER2 Status_Positive": float(payload.her2_status_positive),
        }
    )

    expected_engineered_columns = {
        "Age at Diagnosis",
        "Lymph nodes examined positive",
        "Tumor Size",
        "Neoplasm Histologic Grade_2.0",
        "Neoplasm Histologic Grade_3.0",
        "Tumor Stage_2.0",
        "Tumor Stage_3.0",
        "Tumor Stage_4.0",
        "ER Status_Positive",
        "HER2 Status_Positive",
    }

    missing_required_columns = [
        col for col in expected_engineered_columns if col not in feature_columns
    ]
    if missing_required_columns:
        raise HTTPException(
            status_code=500,
            detail=f"Missing expected training feature columns: {missing_required_columns}",
        )

    return pd.DataFrame([row], columns=feature_columns)


@router.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "breast-cancer-survival-risk-api",
        "model": "penalized_cox_cached_inference",
    }


@router.post("/predict-risk", response_model=RiskPredictionResponse)
def predict_risk(payload: RiskPredictionRequest) -> RiskPredictionResponse:
    """
    Run actual penalized Cox inference using the Cox-ready feature space.
    """
    bundle = get_inference_bundle()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    train_scores = bundle["train_scores"]
    low_cutoff = bundle["low_cutoff"]
    high_cutoff = bundle["high_cutoff"]

    input_df = build_feature_frame(payload, feature_columns)
    risk_score = float(model.predict_partial_hazard(input_df).iloc[0])
    risk_percentile = float((train_scores <= risk_score).mean() * 100)

    if risk_score < low_cutoff:
        risk_group = "Low Risk"
    elif risk_score < high_cutoff:
        risk_group = "Intermediate Risk"
    else:
        risk_group = "High Risk"

    return RiskPredictionResponse(
        model_used="penalized_cox_actual_inference_v1",
        risk_score=round(risk_score, 6),
        risk_group=risk_group,
        risk_percentile=round(risk_percentile, 2),
    )