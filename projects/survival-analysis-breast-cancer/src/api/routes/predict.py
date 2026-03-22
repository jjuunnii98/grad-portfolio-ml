from fastapi import APIRouter

from src.api.schemas import RiskPredictionRequest, RiskPredictionResponse

router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/predict-risk", response_model=RiskPredictionResponse)
def predict_risk(payload: RiskPredictionRequest) -> RiskPredictionResponse:
    """
    Temporary risk scoring endpoint.

    This is a placeholder inference layer that mimics
    survival risk scoring using weighted clinical factors.

    Later, this should be replaced by:
    - loading trained model artifacts
    - transforming request payload into Cox-ready feature vector
    - running model-based inference
    """

    risk_score = (
        payload.age_at_diagnosis * 0.004
        + payload.lymph_nodes_examined_positive * 0.05
        + payload.tumor_size * 0.008
        + payload.grade_2 * 0.10
        + payload.grade_3 * 0.30
        + payload.tumor_stage_2 * 0.15
        + payload.tumor_stage_3 * 0.40
        + payload.tumor_stage_4 * 0.80
        - payload.er_status_positive * 0.20
        + payload.her2_status_positive * 0.25
    )

    if risk_score < 0.5:
        risk_group = "Low Risk"
    elif risk_score < 1.0:
        risk_group = "Intermediate Risk"
    else:
        risk_group = "High Risk"

    return RiskPredictionResponse(
        model_used="mock_penalized_cox_api_v1",
        risk_score=round(risk_score, 4),
        risk_group=risk_group,
    )