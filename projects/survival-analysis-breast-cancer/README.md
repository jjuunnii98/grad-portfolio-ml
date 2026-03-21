# Survival Analysis for Breast Cancer (METABRIC)

## Overview

This project builds an end-to-end survival analysis pipeline using the METABRIC breast cancer clinical dataset.

The objective is to identify clinically interpretable risk factors associated with patient survival using Cox Proportional Hazards modeling.

This project demonstrates a complete workflow covering:
- problem definition
- exploratory data analysis (EDA)
- feature engineering
- survival modeling
- model evaluation and interpretation

---

# 프로젝트 개요

본 프로젝트는 METABRIC 유방암 임상 데이터를 활용하여  
생존 분석(Survival Analysis) 파이프라인을 end-to-end로 구축한 것이다.

Cox Proportional Hazards 모델을 기반으로  
환자의 생존 위험과 관련된 임상적 요인을 해석 가능한 형태로 도출하는 것을 목표로 한다.

전체 분석 과정은 다음 단계를 포함한다:
- 문제 정의
- 데이터 탐색(EDA)
- feature engineering
- survival modeling
- 모델 평가 및 해석

---

## Objective

- Build a reproducible survival analysis pipeline
- Identify statistically significant clinical risk factors
- Interpret hazard ratios in a clinically meaningful way
- Provide an interpretable baseline survival model

---

# 분석 목표

- 재현 가능한 survival analysis pipeline 구축
- 통계적으로 유의한 임상 변수 식별
- hazard ratio를 임상적으로 해석
- 해석 가능한 baseline survival 모델 구축

---

## Dataset

- METABRIC Breast Cancer Clinical Dataset
- Includes patient-level clinical variables and survival outcomes

Key variables:
- Age at Diagnosis
- Tumor Size
- Tumor Stage
- Histologic Grade
- ER Status / HER2 Status
- Lymph Node Positivity
- Overall Survival Time
- Vital Status (event)

---

# 데이터셋

- METABRIC 유방암 임상 데이터
- 환자 단위 임상 변수 및 생존 정보 포함

주요 변수:
- 진단 시 나이
- 종양 크기
- 종양 병기 (Tumor Stage)
- 조직학적 Grade
- ER / HER2 상태
- 양성 림프절 수
- 생존 기간
- 생존 여부 (event)

---

## Project Structure
```
notebooks/
├── 01_problem_definition.ipynb
├── 02_eda.ipynb
├── 03_feature_engineering.ipynb
├── 04_modeling.ipynb
└── 05_evaluation.ipynb
```

---

## Workflow
```
Raw Data
→ Problem Definition
→ EDA
→ Feature Engineering
→ Cox PH Modeling
→ Evaluation & Interpretation
```

---

## Methodology

### 1. Survival Modeling

- Model: Cox Proportional Hazards
- Target:
  - time = Overall Survival (Months)
  - event = death indicator

### 2. Feature Engineering

- Missing value handling (dropna)
- Categorical encoding (one-hot encoding)
- Removal of rare/ambiguous categories (e.g., Tumor Stage 0.0)
- Construction of Cox-ready dataset

### 3. Evaluation

- Hazard Ratio (exp(coef))
- 95% Confidence Intervals
- Statistical significance (p-value)
- Concordance Index

---

# 분석 방법

### 1. 생존 모델

- 모델: Cox Proportional Hazards
- 목표 변수:
  - time: 생존 기간
  - event: 사망 여부

### 2. Feature Engineering

- 결측치 제거
- 범주형 변수 one-hot encoding
- 희귀/모호한 범주 제거 (예: Tumor Stage 0.0)
- Cox 모델 입력 데이터셋 구성

### 3. 평가 방법

- Hazard Ratio
- 95% 신뢰구간
- p-value
- Concordance Index

---

## Key Results

### Significant Risk Factors

The model identified several clinically meaningful risk factors:

- Tumor Stage (higher stage → higher risk)
- Histologic Grade (Grade 3 → higher risk)
- HER2 Positive → increased hazard
- Tumor Size → increased hazard
- Lymph Node Positivity → increased hazard

### Protective Factor

- ER Positive → associated with lower hazard

---

# 주요 결과

다음과 같은 임상적으로 의미 있는 위험 요인이 확인되었다:

- 종양 병기 증가 → 위험 증가
- Grade 3 → 위험 증가
- HER2 양성 → 위험 증가
- 종양 크기 증가 → 위험 증가
- 양성 림프절 수 증가 → 위험 증가

보호 효과:
- ER 양성 → 위험 감소

---

## Interpretation

- Hazard Ratio > 1 → Increased risk
- Hazard Ratio < 1 → Protective effect
- Confidence intervals used for robustness
- Results align with known clinical patterns

---

# 해석

- HR > 1 → 위험 증가
- HR < 1 → 보호 효과
- 신뢰구간을 통해 안정성 확인
- 결과는 임상적으로 타당한 방향성과 일치

---

## Limitations

- No external validation
- Model evaluated on same dataset
- Small subgroup sizes for some categories
- Proportional hazards assumptions not fully optimized

---

# 한계점

- 외부 검증 미실시
- 동일 데이터셋에서 평가
- 일부 subgroup 표본 수 부족
- PH 가정 최적화 미완료

---

## Future Work

- Train / validation split
- Regularized Cox models
- Random Survival Forest / DeepSurv comparison
- Time-dependent evaluation
- External dataset validation

---

# 향후 개선 방향

- train / validation 분리
- 정규화 Cox 모델
- RSF / DeepSurv 비교
- 시간 기반 평가
- 외부 데이터 검증

---

## Tech Stack

- Python
- pandas / numpy
- lifelines
- matplotlib / seaborn
- Jupyter Notebook

---

## Author

Junyeong Song  

AI Engineer | Survival Analysis | Healthcare AI