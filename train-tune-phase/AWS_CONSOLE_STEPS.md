# Train & Tune Phase — AWS Console Steps

## 1. Fully Managed Training — Submit a Training Job

**Path:** AWS Console → SageMaker → Model training & customization → Training & tuning jobs → Create training job

1. Job name: `loan-xgboost-training`
2. IAM role: Select your SageMaker execution role
3. Algorithm options → Choose: Built-in algorithm → XGBoost
4. Input data configuration:
   - Channel name: `train`
   - S3 location: `s3://{bucket}/loan-approval-lab/processed/train/`
   - Content type: `text/csv`
5. Output data configuration:
   - S3 output path: `s3://{bucket}/loan-approval-lab/models/`
6. Resource configuration:
   - Instance type: `ml.m5.large`
   - Instance count: 1
   - Volume size: 10 GB
7. Hyperparameters:
   - objective: `binary:logistic`
   - num_round: `100`
   - max_depth: `4`
   - eta: `0.1`
8. Stopping condition: Max runtime 600 seconds
9. Click "Create training job"
10. Monitor: Status changes InProgress → Completed

---

## 2. Automatic Model Tuning (Hyperparameter Optimization)

**Path:** AWS Console → SageMaker → Model training & customization → Training & tuning jobs → Create hyperparameter tuning job

1. Job name: `loan-xgboost-tuning`
2. Select algorithm: XGBoost
3. Objective metric: `validation:auc` (maximize)
4. Hyperparameter ranges:
   - max_depth: Integer [3, 10]
   - eta: Continuous [0.01, 0.3]
   - num_round: Integer [50, 300]
   - subsample: Continuous [0.5, 1.0]
5. Resource limits:
   - Max training jobs: 10
   - Max parallel jobs: 2
6. Input/Output same as training job
7. Early stopping: Enable
8. Click "Create"
9. Monitor: Each candidate runs, best one highlighted

---

## 3. Managed Spot Training

Same as step 1, but in Resource configuration:
- Check "Managed spot training" checkbox
- Max wait time: 1200 seconds
- Max runtime: 600 seconds
- Gives up to 90% cost saving on training compute

---

## 4. Debugger and Profiler

**Path:** AWS Console → SageMaker → Profiler (under Applications and IDEs)

- When creating a training job, enable "SageMaker Debugger" in advanced options
- After job completes: Console → SageMaker → Profiler → Select your job
- View: CPU/GPU utilization, memory usage, bottleneck detection
- Debugger rules auto-detect: vanishing gradients, overfitting, etc.

---

## 5. Experiments (MLflow)

**Path:** AWS Console → SageMaker → Studio → Experiments

1. Create MLflow Tracking Server (one-time setup)
2. All training jobs automatically logged
3. Compare metrics across runs
4. Register best model for deployment

---

## 6. JumpStart (for Distributed Training / Training Compiler)

**Path:** SageMaker Studio → JumpStart / Models

- For large models: JumpStart provides pre-configured distributed training
- Training Compiler: Available with PyTorch/TensorFlow models
- Select model → Train → Enable "Training Compiler" in advanced options

---

## Console Navigation Quick Reference

| Task | Console Path |
|------|-------------|
| Submit Training Job | SageMaker → Training & tuning jobs → Create |
| HPO / Tuning | SageMaker → Training & tuning jobs → Create HPO job |
| Spot Training | Same as training → Check "Managed Spot" box |
| Debugger/Profiler | SageMaker → Profiler |
| Experiments | SageMaker Studio → Experiments (MLflow) |
| Monitor Jobs | SageMaker → Training & tuning jobs → Click job |
| View Logs | CloudWatch → Log groups → /aws/sagemaker/TrainingJobs |
| Model Artifacts | S3 → bucket → loan-approval-lab/models/ |
