from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import RiskPredictionRequest, RiskPredictionResponse
from src.models.survival_model import load_pickle_artifact
from src.utils.config import load_config

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def get_inference_bundle() -> dict[str, Any]:
    """
    Load the pre-trained penalized Cox inference bundle from pickle.

    This removes request-time retraining and reduces cold-start overhead.
    """
    config = load_config(CONFIG_PATH)
    artifact_path = PROJECT_ROOT / config["output"]["model_artifact_path"]
    return load_pickle_artifact(artifact_path)


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
            detail=(
                "Missing expected training feature columns: "
                f"{missing_required_columns}"
            ),
        )

    return pd.DataFrame([row], columns=feature_columns)


def assign_risk_group(
    risk_score: float,
    low_cutoff: float,
    high_cutoff: float,
) -> str:
    """
    Map a continuous Cox partial hazard score to a discrete risk group.
    """
    if risk_score < low_cutoff:
        return "Low Risk"
    if risk_score < high_cutoff:
        return "Intermediate Risk"
    return "High Risk"


def assign_risk_percentile(
    risk_score: float,
    low_cutoff: float,
    high_cutoff: float,
) -> float:
    """
    Return a simple percentile-style bucket aligned with the risk thresholds.
    """
    if risk_score <= low_cutoff:
        return 33.0
    if risk_score <= high_cutoff:
        return 67.0
    return 90.0


@router.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "breast-cancer-survival-risk-api",
        "model": "penalized_cox_pickle_inference",
    }


@router.post("/predict-risk", response_model=RiskPredictionResponse)
def predict_risk(payload: RiskPredictionRequest) -> RiskPredictionResponse:
    """
    Run penalized Cox inference using a pre-trained pickle artifact.
    """
    bundle = get_inference_bundle()
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    low_cutoff = float(bundle["low_cutoff"])
    high_cutoff = float(bundle["high_cutoff"])

    input_df = build_feature_frame(payload, feature_columns)
    risk_score = float(model.predict_partial_hazard(input_df).iloc[0])
    risk_group = assign_risk_group(risk_score, low_cutoff, high_cutoff)
    risk_percentile = assign_risk_percentile(risk_score, low_cutoff, high_cutoff)

    return RiskPredictionResponse(
        model_used="penalized_cox_pickle_inference_v1",
        risk_score=round(risk_score, 6),
        risk_group=risk_group,
        risk_percentile=round(risk_percentile, 2),
    )