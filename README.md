> 🚀 Production-ready Survival Analysis System with API Deployment

# Survival Analysis for Breast Cancer (METABRIC)

## 🚀 Featured Project

### Survival Analysis API (Production)

👉 https://survival-api.onrender.com/docs

- Cox Survival Model
- FastAPI Inference
- Docker Deployment
- Cloud (Render)

---

## 🌐 Live API

👉 https://survival-api.onrender.com/docs  

> 🚀 **Production Deployment (Render)**
> - Fully containerized with Docker
> - Real-time inference via FastAPI
> - Health-monitored cloud service

### Available Endpoints

- `GET /health` → API health check  
- `POST /predict-risk` → Survival risk prediction  

---

## 🧠 Abstract

This project presents an end-to-end survival analysis pipeline using the METABRIC breast cancer clinical dataset.

It integrates statistical survival modeling (Cox Proportional Hazards) with production-level ML engineering, including:

- Feature engineering pipeline
- Model training and refinement
- Evaluation and interpretation
- FastAPI-based inference service
- Docker containerization
- Cloud deployment (Render)

👉 This project demonstrates the transition from research-grade modeling to production-ready AI systems.

---

## 📌 Overview

This project covers the full ML lifecycle:

1. Problem definition  
2. Data preprocessing  
3. Feature engineering  
4. Survival modeling (Cox PH)  
5. Model refinement (regularization)  
6. Evaluation  
7. API deployment (FastAPI + Docker + Cloud)

👉 Goal:  
**Bridge classical survival analysis with modern AI engineering and deployment**

---

# 🇰🇷 프로젝트 개요

본 프로젝트는 METABRIC 유방암 임상 데이터를 기반으로  
생존 분석(Survival Analysis)을 end-to-end로 구현한 프로젝트이다.

단순 분석이 아닌:

- 재현 가능한 ML pipeline  
- 모델 해석 가능성  
- API 서비스화  
- Docker 기반 배포  
- Cloud (Render) 운영  

까지 포함한 **실무 수준 AI 시스템**을 구축하였다.

---

## 📊 Dataset

- METABRIC Breast Cancer Clinical Dataset  
- 환자 단위 임상 데이터 + 생존 정보 포함  

### 주요 변수

- Age at Diagnosis  
- Tumor Size  
- Tumor Stage  
- Histologic Grade  
- ER Status / HER2 Status  
- Lymph Node Positivity  
- Survival Time (Months)  
- Vital Status  

---

## ⚙️ Survival Problem Definition

- Duration (T): Overall Survival (Months)  
- Event (E):  
  - 1 → Died of Disease  
  - 0 → Censored  

👉 Right-censored survival modeling 적용

---

## 🏗️ Methodology

### 1️⃣ Feature Engineering

- 결측값 제거 (`dropna`)
- 수치형 변환
- 범주형 one-hot encoding
- 이상값 제거 (Tumor Stage 0.0)

👉 Final dataset: 
```
Shape: (1353, 12)
```

---

### 2️⃣ Survival Modeling

#### Baseline Cox
- No regularization

#### Penalized Cox
- L2 regularization (penalizer)

---

### 3️⃣ Evaluation Metrics

- Concordance Index (C-index)
- Hazard Ratio (exp(coef))
- Overfitting Gap

---

## 📈 Results

| Model | Train C-index | Valid C-index | Overfitting Gap |
|------|--------------|--------------|----------------|
| Penalized Cox | 0.718 | 0.638 | 0.079 |
| Baseline Cox | 0.716 | 0.629 | 0.086 |

---

## 🔍 Key Insights

- Penalized Cox improves generalization
- Overfitting 감소
- Coefficient 안정성 증가

---

## 🧬 Clinical Interpretation

### High Risk Factors (HR > 1)

- Tumor Stage ↑ → Hazard 증가  
- Histologic Grade 3 → 높은 위험  
- HER2 Positive → 위험 증가  
- Tumor Size ↑ → 위험 증가  
- Lymph Node Positivity ↑ → 위험 증가  

### Protective Factor (HR < 1)

- ER Positive → 생존율 증가  

---

## 🔧 Engineering Architecture

### Pipeline Flow
```
config.yaml
↓
build_features.py
↓
survival_model.py
↓
main.py
↓
FastAPI (inference)
```

---

## 🚀 Run Pipeline

```bash
python main.py
```

---

## 📦 Outputs

- `data/processed/metabric_clinical_featurized.csv`  
- `data/processed/model_refinement_comparison.csv`  
- `data/processed/model_refinement_coefficients.csv`  
- `artifacts/model/penalized_cox_inference_bundle.pkl`  

---

## 🌐 API (FastAPI)

### Run locally

```bash
uvicorn src.api.main:app --reload
```

### Swagger

```text
http://127.0.0.1:8000/docs
```

---

## 🧪 Example Request

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

---

## 🧪 Example Response

```json
{
  "model_used": "penalized_cox_pickle_inference",
  "risk_score": 0.786,
  "risk_group": "Intermediate Risk"
}
```

---

## 🐳 Deployment

### Docker

```bash
docker build -t survival-api .
docker run -p 8000:8000 survival-api
```

---

## ☁️ Cloud Deployment (Render)
- Docker 기반 배포  
- FastAPI 서비스 운영  
- Health check 기반 자동 관리  

👉 Live API:  
https://survival-api.onrender.com/docs

---

## ⚠️ Limitations
- External validation 없음  
- PH assumption 완전 검증 X  
- Time-dependent 변수 미포함  

---

## 🔮 Future Work
- Time-dependent Cox model  
- Random Survival Forest  
- DeepSurv / Deep Learning  
- External dataset validation  
- MLOps (CI/CD, monitoring)  

---

## 🧰 Tech Stack
- Python  
- pandas / numpy  
- lifelines  
- scikit-learn  
- FastAPI  
- Docker  
- Render  

---

## 👨‍💻 Author

### Junyeong Song

#### AI Engineer | Survival Analysis | Healthcare AI

- GitHub: https://github.com/jjuunnii98  
- LinkedIn: https://www.linkedin.com/in/jun-yeong-song/  