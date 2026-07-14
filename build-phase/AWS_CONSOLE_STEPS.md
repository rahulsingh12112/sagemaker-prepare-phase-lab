# BUILD Phase — AWS Console Steps (No Code)

This document covers all the AWS Console (GUI) steps for the BUILD phase of the SageMaker ML lifecycle.

---

## 1. Verify Training Jobs

**Path:** AWS Console → SageMaker → Model training & customization → Training & tuning jobs

- All submitted training jobs appear here
- Each job shows: Status (InProgress / Completed / Failed), Duration, Instance type
- Click on a job name to see: Hyperparameters, Input/Output paths, Metrics, Logs

---

## 2. Verify Data in S3

**Path:** AWS Console → S3 → `sagemaker-{region}-{account_id}` → `loan-approval-lab/`

Expected folder structure after running the lab:

```
loan-approval-lab/
├── raw/
│   └── loan_data.csv                  (original dataset)
├── processed/
│   ├── train/
│   │   ├── train.csv                  (training data, no headers)
│   │   └── train_with_headers.csv     (training data with column names)
│   └── test/
│       └── test.csv                   (test data, no headers)
├── models/
│   └── model.tar.gz                   (trained model artifact)
├── feature-store/
│   └── (parquet files)                (offline feature store data)
└── autopilot-output/
    └── (autopilot generated files)
```

---

## 3. JumpStart — Use Pre-trained Models

**Path:** AWS Console → SageMaker → Studio → Left menu → JumpStart / Models

**Steps:**

1. Open SageMaker Studio
2. Click "JumpStart / Models" in the left navigation
3. Search for "XGBoost" or "LightGBM" in the search bar
4. Select a model card (e.g., "LightGBM Classification")
5. Click "Train" to fine-tune on your own data
6. Provide S3 path to your training data: `s3://{bucket}/loan-approval-lab/processed/train/`
7. Configure instance type (ml.m5.large is sufficient)
8. Click "Train" to start the training job
9. Monitor progress in Training & tuning jobs

Alternatively, click "Deploy" to deploy a pre-trained model directly to an endpoint without fine-tuning.

---

## 4. Autopilot — Automatic Model Building (via Canvas)

**Path:** AWS Console → SageMaker → Applications and IDEs → Canvas → Open Canvas

**Steps:**

1. In Canvas, go to "Datasets" in the left menu
2. Click "Import" or "Create dataset"
3. Upload `loan_data.csv` from your local machine
4. Give it a name: `loan-approval-data`
5. Wait for import to complete
6. Go to "My Models" in the left menu
7. Click "New Model" or "+ Create"
8. Enter model name: `loan-approval-model`
9. Select dataset: `loan-approval-data`
10. Select target column: `loan_approved`
11. Canvas auto-detects problem type: Binary Classification
12. Choose build type:
    - **Quick build** — 2 to 5 minutes, tries fewer candidates (recommended for testing)
    - **Standard build** — 1 to 2 hours, tries 250+ candidates (full Autopilot)
13. Click "Build"
14. Wait for completion

**Results page shows:**
- Overall accuracy and F1 score
- Feature importance chart (which features matter most)
- Confusion matrix (true positives, false positives, etc.)
- Option to deploy the model or run batch predictions

---

## 5. Feature Store — Verify Features

**Path:** AWS Console → SageMaker → Studio → Left menu → Feature Store

**What you can see:**

- List of all Feature Groups created
- Click on a Feature Group to see:
  - Feature definitions (column names and types)
  - Record count
  - Online store status
  - Offline store S3 location
- Run sample queries against the offline store using Athena

---

## 6. Processing Jobs — Verify Data Preprocessing

**Path:** AWS Console → SageMaker → Data preparation → Processing jobs

- Lists all processing jobs (data preprocessing runs)
- Each job shows: Status, Duration, Instance type, Input/Output S3 paths
- Click a job to see CloudWatch logs for debugging

---

## 7. MLflow Experiments — Track Training Runs

**Path:** AWS Console → SageMaker → Studio → Left menu → Experiments

- Requires an MLflow Tracking Server (create one if not exists)
- Once set up, all training metrics, parameters, and artifacts are logged
- Compare multiple runs side-by-side
- Register best models for deployment

---

## Summary — Console Navigation Quick Reference

| Task | Console Path |
|------|-------------|
| Check training jobs | SageMaker → Training & tuning jobs |
| Browse S3 data | S3 → your bucket → loan-approval-lab/ |
| Use JumpStart models | SageMaker Studio → JumpStart / Models |
| Run Autopilot (AutoML) | SageMaker → Canvas → My Models → New Model |
| View Feature Store | SageMaker Studio → Feature Store |
| Check processing jobs | SageMaker → Data preparation → Processing jobs |
| Track experiments | SageMaker Studio → Experiments (MLflow) |
