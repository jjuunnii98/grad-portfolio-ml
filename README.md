# 📊 Grad Portfolio — Machine Learning & Survival Analysis

## Overview
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