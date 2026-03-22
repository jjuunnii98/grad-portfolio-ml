# Survival Analysis for Breast Cancer (METABRIC)

## Abstract

This project presents a full survival analysis pipeline using the METABRIC breast cancer clinical dataset. We implement Cox Proportional Hazards models to identify clinically interpretable risk factors associated with patient survival.

The pipeline is designed to be reproducible, modular, and deployable, transitioning from exploratory analysis to production-ready modeling.

---

## Overview
👉 Production-ready survival analysis system with real-time inference via FastAPI and Docker

This project builds an end-to-end survival analysis system covering:

- Problem definition
- Exploratory Data Analysis (EDA)
- Feature engineering
- Survival modeling (Cox PH)
- Model refinement (regularization)
- Evaluation and interpretation

The goal is to bridge statistical survival analysis with modern ML engineering practices.

---

# 프로젝트 개요

본 프로젝트는 METABRIC 유방암 임상 데이터를 기반으로
생존 분석(Survival Analysis)을 end-to-end로 구현한 프로젝트이다.

단순 분석을 넘어서 재현 가능한 ML 파이프라인과
모델 해석까지 포함한 실무 수준의 ML 시스템 구조를 구축하였다.

---

## Dataset

- METABRIC Breast Cancer Clinical Dataset
- Patient-level clinical features with survival outcomes

### Key Variables

- Age at Diagnosis
- Tumor Size
- Tumor Stage
- Histologic Grade
- ER Status / HER2 Status
- Lymph Node Positivity
- Overall Survival Time (Months)
- Vital Status (event)

---

## Survival Problem Definition

We define the survival task as:

- Duration (T): Overall Survival (Months)
- Event (E):
  - 1 → Died of Disease
  - 0 → Censored

This formulation enables right-censored survival modeling.

---

## Methodology

### 1. Feature Engineering

- Missing value removal (`dropna`)
- Numeric coercion
- One-hot encoding of categorical variables
- Removal of rare/ambiguous categories (e.g., Tumor Stage 0.0)

Final dataset:
- Shape: (1353, 12)
- Cox-ready format

---

### 2. Survival Modeling

We implement two Cox models:

#### Baseline Cox
- No regularization

#### Penalized Cox
- L2 regularization (penalizer)

---

### 3. Evaluation Metrics

- Concordance Index (C-index)
- Hazard Ratio (exp(coef))
- Overfitting Gap (train - validation)

---

## Results

### Model Performance

| Model | Train C-index | Valid C-index | Overfitting Gap |
|------|--------------|--------------|----------------|
| Penalized Cox | 0.718 | 0.638 | 0.079 |
| Baseline Cox | 0.716 | 0.629 | 0.086 |

### Key Insight

- Penalized Cox improves generalization
- Reduces overfitting gap
- Produces more stable coefficients

---

## Clinical Interpretation

### High-Risk Factors (HR > 1)

- Tumor Stage ↑ → Hazard ↑
- Histologic Grade 3 → High risk
- HER2 Positive → Increased hazard
- Tumor Size ↑ → Increased risk
- Lymph Node Positivity ↑ → Increased risk

### Protective Factor (HR < 1)

- ER Positive → Reduced hazard

---

## Coefficient Stability (Regularization Effect)

Penalized Cox reduces extreme coefficients:

Example:

- Grade 2: 0.52 → 0.02
- Grade 3: 0.82 → 0.32

→ Improves robustness and interpretability

---

## Project Structure

```
projects/survival-analysis-breast-cancer/
├── notebooks/
├── src/
│   ├── features/
│   ├── models/
│   ├── api/
│   └── utils/
├── configs/
├── data/
├── artifacts/        # trained model (.pkl)
└── main.py
```

---

## Pipeline Architecture

```
config.yaml
   ↓
build_features.py
   ↓
survival_model.py
   ↓
main.py
   ↓
pickle artifact (.pkl)
   ↓
FastAPI inference
```

---

## Run the Pipeline

```bash
python main.py
```

### Outputs

- `data/processed/metabric_clinical_featurized.csv`
- `data/processed/model_refinement_comparison.csv`
- `data/processed/model_refinement_coefficients.csv`
- `artifacts/model/penalized_cox_inference_bundle.pkl`

---

## API Deployment (FastAPI)

This project includes a FastAPI service for survival risk prediction using a trained penalized Cox model (production-ready inference).

### Run the API Server

```bash
uvicorn src.api.main:app --reload
```

### Access Swagger UI

```text
http://127.0.0.1:8000/docs
```

### Available Endpoints

- `GET /health` → API health check
- `POST /predict-risk` → Predict survival risk

### Example Request

```json
{
  "age_at_diagnosis": 55,
  "lymph_nodes_examined_positive": 2,
  "tumor_size": 22,
  "grade_2": 0,
  "grade_3": 1,
  "tumor_stage_2": 1,
  "tumor_stage_3": 0,
  "tumor_stage_4": 0,
  "er_status_positive": 1,
  "her2_status_positive": 0
}
```

### Example Response
👉 Risk score is computed using partial hazard from Cox model and mapped to risk groups via quantile thresholds

```json
{
  "model_used": "penalized_cox_v1",
  "risk_score": 0.786,
  "risk_group": "Intermediate Risk"
}
```

---

## API 실행 (FastAPI)

본 프로젝트는 survival risk 예측을 위한 FastAPI 서버를 포함한다.

### 서버 실행

```bash
uvicorn src.api.main:app --reload
```

### Swagger 접속

```text
http://127.0.0.1:8000/docs
```

### 주요 기능

- `/health` → 서버 상태 확인
- `/predict-risk` → 생존 위험 예측

→ 현재는 학습된 Cox 모델(pickle artifact)을 기반으로 실제 inference 수행

---

## 🐳 Docker Deployment

### Build

```bash
docker build -t survival-api .
```

### Run

```bash
docker run -p 8000:8000 survival-api
```

### Access API

```text
http://127.0.0.1:8000/docs
```

This container serves the trained Cox model using FastAPI with preloaded inference artifacts.

---

## Limitations

- No external validation dataset
- Proportional hazards assumption not fully tested
- Static features only (no time-dependent covariates)

---

## Future Work

- Time-dependent Cox model
- Random Survival Forest (RSF)
- DeepSurv / Neural Survival Models
- External dataset validation
- Deployment via FastAPI

---

## Tech Stack

- Python
- pandas / numpy
- lifelines
- scikit-learn
- matplotlib / seaborn

---

## Author

### Junyeong Song  
#### AI Engineer | Survival Analysis | Healthcare AI