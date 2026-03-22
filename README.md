# 📊 Grad Portfolio — Machine Learning & Survival Analysis

## Overview
👉 Production-ready survival analysis system with Cox model serving via FastAPI

This repository represents a **research-oriented machine learning portfolio**  
focused on **statistical modeling, survival analysis, and interpretable AI**.

It complements my engineering projects by emphasizing:
- model interpretability
- statistical reasoning
- rigorous evaluation methodologies

---

## Objective
To develop a deeper understanding of machine learning models through  
**statistical foundations and research-driven experimentation**.

This repository focuses on:

- Survival Analysis (time-to-event modeling)
- Statistical Modeling & Inference
- Interpretable Machine Learning
- Reproducible analytical workflows

---

# 🚀 Highlight Project

## Breast Cancer Survival Analysis (METABRIC)

An end-to-end survival analysis project using the METABRIC breast cancer clinical dataset.

### Key Components
- Problem definition and clinical target design  
- Exploratory Data Analysis (EDA)  
- Cox-ready feature engineering  
- Baseline Cox Proportional Hazards modeling  
- Model evaluation (Concordance Index)  
- Model refinement with train-validation split  
- Penalized Cox model comparison  

### Project Link
👉 https://github.com/jjuunnii98/grad-portfolio-ml/tree/main/projects/survival-analysis-breast-cancer

### 🔥 Project Highlights
- Cox Proportional Hazards modeling with clinical interpretability
- Penalized Cox model to improve generalization and reduce overfitting
- Fully reproducible pipeline using config-driven architecture
- Modular ML system (`features → models → pipeline → main.py`)
- FastAPI-based inference service (`/predict-risk`)

### Run Pipeline
```bash
python projects/survival-analysis-breast-cancer/main.py
```

### Run API
```bash
uvicorn src.api.main:app --reload
```

### API Docs
```text
http://127.0.0.1:8000/docs
```

### Key Result
- Validation C-index: **0.638 (Penalized Cox)**
- Improved generalization vs baseline model

### Workflow

```text
01_problem_definition.ipynb
→ 02_eda.ipynb
→ 03_feature_engineering.ipynb
→ 04_modeling.ipynb
→ 05_evaluation.ipynb
→ 06_model_refinement.ipynb
```

## ▶️ Run the Pipeline

You can execute the full survival analysis pipeline from a single entrypoint.

### What `main.py` does
- Runs feature engineering from the centralized config
- Builds the Cox-ready dataset
- Executes baseline vs penalized Cox model refinement
- Saves model comparison outputs to configured paths

### Command

```bash
python projects/survival-analysis-breast-cancer/main.py
```

### Expected Outputs
- `data/processed/metabric_clinical_featurized.csv`
- `data/processed/model_refinement_comparison.csv`
- `data/processed/model_refinement_coefficients.csv`

---

## ▶️ 파이프라인 실행

하나의 entrypoint로 전체 survival analysis 파이프라인을 실행할 수 있다.

### `main.py`가 수행하는 작업
- config 기반 feature engineering 실행
- Cox 모델 입력용 데이터셋 생성
- baseline Cox vs penalized Cox 비교
- 결과 CSV 자동 저장

### 실행 명령어

```bash
python projects/survival-analysis-breast-cancer/main.py
```

### 생성 결과물
- `data/processed/metabric_clinical_featurized.csv`
- `data/processed/model_refinement_comparison.csv`
- `data/processed/model_refinement_coefficients.csv`


---

## Repository Structure
```
projects/
└── survival-analysis-breast-cancer/
    ├── README.md
    ├── configs/
    ├── data/
    ├── notebooks/
    │   ├── 01_problem_definition.ipynb
    │   ├── 02_eda.ipynb
    │   ├── 03_feature_engineering.ipynb
    │   ├── 04_modeling.ipynb
    │   ├── 05_evaluation.ipynb
    │   └── 06_model_refinement.ipynb
    ├── results/
    └── src/
```

---

## Methodological Focus

This repository emphasizes:

	•	Statistical rigor over black-box performance
	•	Interpretability over complexity
	•	Reproducibility over ad-hoc experimentation


---

## Relationshipto Engineering Projects

This repository complements my main ML engineering projects:

	•	End-to-End ML Analytics System
	•	Healthcare Risk Prediction API
	•	Crypto Risk Intelligence Pipeline

While those focus on system implementation, this repository focuses on model understanding and research depth.

---

## Tools & Technologies
	•	Python
	•	pandas · NumPy
	•	scikit-learn
	•	lifelines (survival analysis)
	•	statsmodels
	•	Jupyter Notebook
	•	Git / GitHub

---

## Future Work
	•	Advanced survival models (time-varying covariates)
	•	Calibration and model validation techniques
	•	Penalized and regularized survival models
	•	Research paper replication studies
	•	Transition of notebook logic into reusable src/ modules
	•	API-oriented deployment experiments for survival risk inference

---

## Author

### Junyeong Song
#### AI Engineer | Machine Learning Systems | Survival Analysis
# 📊 Grad Portfolio — Machine Learning & Survival Analysis

## Overview
This repository represents a **research-driven machine learning portfolio** focused on:

- Survival Analysis (time-to-event modeling)
- Statistical Modeling & Inference
- Interpretable Machine Learning
- Reproducible ML pipelines
- Production-level model serving

It bridges the gap between **statistical rigor** and **engineering implementation**, evolving from notebook-based research into a deployable ML system.

---

## 🎯 Objective

To build **interpretable and production-ready ML systems** grounded in statistical principles.

This repository emphasizes:

- Model interpretability over black-box complexity
- Reproducibility via config-driven pipelines
- Transition from research → engineering → deployment

---

# 🚀 Highlight Project

## Breast Cancer Survival Analysis (METABRIC)

An end-to-end survival analysis project using the **METABRIC clinical dataset**, extended into a **production-grade inference system**.

### 🔥 Core Capabilities

- Cox Proportional Hazards modeling (clinical interpretability)
- Penalized Cox model for generalization improvement
- Config-driven ML pipeline (YAML-based reproducibility)
- Modular architecture (`features → models → pipeline → API`)
- FastAPI inference service with real model serving
- Pickle-based model artifact (cold-start optimized)

---

## 🧠 Modeling Summary

| Model            | Train C-index | Validation C-index | Overfitting Gap |
|-----------------|--------------|-------------------|-----------------|
| Baseline Cox    | 0.715        | 0.629             | 0.086           |
| Penalized Cox   | **0.718**    | **0.638**         | **0.079**       |

👉 Penalization improves generalization and reduces overfitting.

---

## ⚙️ End-to-End Pipeline

```text
Raw Clinical Data
→ Feature Engineering
→ Cox-ready Dataset
→ Model Training (Baseline vs Penalized)
→ Model Evaluation
→ Model Refinement
→ Final Model Training
→ Pickle Artifact Generation
→ FastAPI Inference Serving
```

---

## ▶️ Run the Pipeline

```bash
python projects/survival-analysis-breast-cancer/main.py
```

### What `main.py` does

- Feature engineering from config
- Model training & validation
- Penalized Cox refinement
- Final model training on full dataset
- **Inference artifact generation (.pkl)**

### Expected Outputs

```text
data/processed/metabric_clinical_featurized.csv
data/processed/model_refinement_comparison.csv
data/processed/model_refinement_coefficients.csv
artifacts/model/penalized_cox_inference_bundle.pkl
```

---

## 🌐 Run API (Production Inference)

```bash
uvicorn src.api.main:app --reload
```

### API Docs

```text
http://127.0.0.1:8000/docs
```

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

👉 Returns risk score + risk group using trained Cox model

---

## 🐳 Docker (Deployment Ready)

### Build

```bash
docker build -t survival-api projects/survival-analysis-breast-cancer
```

### Run

```bash
docker run -p 8000:8000 survival-api
```

---

## 📂 Project Structure

```text
projects/
└── survival-analysis-breast-cancer/
    ├── configs/          # YAML-based configuration
    ├── data/             # Processed datasets
    ├── artifacts/        # Trained model (.pkl)
    ├── notebooks/        # Research workflow
    ├── src/
    │   ├── features/     # Feature engineering
    │   ├── models/       # Survival models
    │   ├── api/          # FastAPI service
    │   └── utils/        # Config loader
    ├── main.py           # Pipeline entrypoint
    └── README.md
```

---

## 🔬 Methodological Focus

- Statistical rigor over black-box modeling
- Survival-specific evaluation (C-index)
- Explicit overfitting diagnostics
- Feature-level interpretability (hazard ratios)

---

## 🔗 Relationship to Engineering Projects

This repository complements engineering-focused systems such as:

- End-to-End ML Pipelines
- Healthcare Risk Prediction APIs
- Risk Intelligence Systems (Finance / Crypto)

👉 This repo = **"Model Understanding Layer"**
👉 Engineering repos = **"System Implementation Layer"**

---

## 🛠 Tech Stack

- Python
- pandas / NumPy
- scikit-learn
- lifelines (survival analysis)
- FastAPI
- Docker
- Git / GitHub

---

## 🚀 Future Work

- Time-varying Cox models
- Survival calibration techniques
- Deep survival models (DeepSurv)
- CI/CD + cloud deployment
- Real-time inference pipelines

---

## 👨‍💻 Author

### Junyeong Song
AI Engineer | Survival Analysis | ML Systems

---

## 💡 One-line Summary

**From statistical survival analysis → to production-ready ML inference system.**