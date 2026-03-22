from pydantic import BaseModel, Field


class RiskPredictionRequest(BaseModel):
    age_at_diagnosis: float = Field(..., gt=0, description="Age at diagnosis")
    lymph_nodes_examined_positive: float = Field(
        ..., ge=0, description="Number of positive lymph nodes"
    )
    tumor_size: float = Field(..., gt=0, description="Tumor size")
    grade_2: int = Field(..., ge=0, le=1, description="Grade 2 indicator")
    grade_3: int = Field(..., ge=0, le=1, description="Grade 3 indicator")
    tumor_stage_2: int = Field(..., ge=0, le=1, description="Tumor Stage 2 indicator")
    tumor_stage_3: int = Field(..., ge=0, le=1, description="Tumor Stage 3 indicator")
    tumor_stage_4: int = Field(..., ge=0, le=1, description="Tumor Stage 4 indicator")
    er_status_positive: int = Field(..., ge=0, le=1, description="ER positive indicator")
    her2_status_positive: int = Field(..., ge=0, le=1, description="HER2 positive indicator")


class RiskPredictionResponse(BaseModel):
    model_used: str
    risk_score: float
    risk_group: str