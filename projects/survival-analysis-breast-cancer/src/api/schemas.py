from pydantic import BaseModel, ConfigDict, Field, model_validator


class RiskPredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age_at_diagnosis": 55,
                "lymph_nodes_examined_positive": 2,
                "tumor_size": 22,
                "grade_2": 0,
                "grade_3": 1,
                "tumor_stage_2": 1,
                "tumor_stage_3": 0,
                "tumor_stage_4": 0,
                "er_status_positive": 1,
                "her2_status_positive": 0,
            }
        }
    )

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

    @model_validator(mode="after")
    def validate_one_hot_groups(self) -> "RiskPredictionRequest":
        if self.grade_2 + self.grade_3 > 1:
            raise ValueError("Only one of grade_2 or grade_3 can be 1.")

        if self.tumor_stage_2 + self.tumor_stage_3 + self.tumor_stage_4 > 1:
            raise ValueError(
                "Only one of tumor_stage_2, tumor_stage_3, tumor_stage_4 can be 1."
            )

        return self


class RiskPredictionResponse(BaseModel):
    model_used: str
    risk_score: float
    risk_group: str
    risk_percentile: float