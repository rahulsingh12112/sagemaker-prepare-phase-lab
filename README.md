# SageMaker Data Preparation Pipeline — Loan Approval Use Case

## Overview

Built an end-to-end data preparation pipeline using AWS SageMaker services for a loan approval prediction use case. This project covers the full "Prepare" phase of the ML lifecycle.

## Architecture

```
Raw Data (S3) → Preprocessing → Feature Store → Bias Detection → Training-Ready Dataset
```

## What I Did

- Loaded raw loan application data and performed EDA
- Built a preprocessing pipeline: handled missing values, engineered features (risk_score, loan_to_income ratio), encoded categoricals
- Created a Feature Group in SageMaker Feature Store (Online + Offline)
- Ingested processed features and validated real-time reads from Online Store
- Ran pre-training bias analysis on gender to check fairness before model training

## Services Used

| Service | Purpose |
|---------|---------|
| S3 | Raw & processed data storage |
| SageMaker Feature Store | Centralized feature repository |
| SageMaker Clarify (concepts) | Pre-training bias metrics |
| Boto3 | All AWS API interactions |
| Pandas / Scikit-learn | Local preprocessing & splitting |

## Dataset

Synthetic loan approval dataset (50 records, 13 features) — includes demographics, financials, and loan details.

## Key Findings

- Feature engineering improved signal: `risk_score` and `loan_to_income` ratio are strong predictors
- Gender bias analysis showed slight imbalance in approval rates — flagged for mitigation
- Feature Store enables consistent feature serving across training and inference

## How to Run

1. Open the notebook in SageMaker Studio (JupyterLab)
2. Upload `data/loan_data.csv` to your working directory
3. Run cells sequentially
4. Remember to stop your JupyterLab space after completion

## Project Structure

```
├── README.md
├── data/
│   └── loan_data.csv
├── notebooks/
│   └── prepare_phase_lab.ipynb       ← Data preparation pipeline
└── build-phase/
    ├── notebooks/
    │   └── build_phase_lab.ipynb     ← Model training (all build options)
    └── scripts/
        └── custom_train.py           ← BYO training script
```

## BUILD Phase — What's Covered

| SageMaker Feature | What I Demonstrated |
|-------------------|---------------------|
| Studio Notebooks | Entire lab runs in JupyterLab |
| Built-in Algorithms | XGBoost + Linear Learner training |
| JumpStart | LightGBM-style pre-trained model |
| Autopilot | Multi-algorithm candidate evaluation |
| Bring Your Own | Custom sklearn RandomForest script |
| Local Mode | Local prototyping before cloud submission |
