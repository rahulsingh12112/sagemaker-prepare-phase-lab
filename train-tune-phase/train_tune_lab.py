"""
Train & Tune Phase — SageMaker Lab
Covers: Fully Managed Training, Distributed Training, Training Compiler,
        Automatic Model Tuning, Managed Spot Training, Debugger/Profiler,
        Experiments, Customization Support
"""

import boto3
import pandas as pd
import numpy as np
import time
import json
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.base import clone
import tracemalloc

# ─── AWS SETUP ───────────────────────────────────────────────────────────────

region = boto3.session.Session().region_name
sts = boto3.client('sts')
caller = sts.get_caller_identity()
account_id = caller['Account']
arn = caller['Arn']

if ':assumed-role/' in arn:
    role_name = arn.split(':assumed-role/')[1].split('/')[0]
    role = f"arn:aws:iam::{account_id}:role/{role_name}"
else:
    role = arn

bucket = f"sagemaker-{region}-{account_id}"
prefix = "loan-approval-lab"
s3 = boto3.client('s3')
sm = boto3.client('sagemaker')

print(f"Region: {region}")
print(f"Bucket: {bucket}")

# ─── DATA PREP ───────────────────────────────────────────────────────────────

df = pd.read_csv('../data/loan_data.csv')
df = df.drop('customer_id', axis=1)
df['income_per_exp_year'] = df['income'] / (df['employment_years'] + 1)
df['loan_to_income'] = df['loan_amount'] / df['income']
df['risk_score'] = df['late_payments'] * 10 + df['debt_to_income'] * 100

home_dummies = pd.get_dummies(df['home_ownership'], prefix='home').astype(int)
purpose_dummies = pd.get_dummies(df['purpose'], prefix='purpose').astype(int)
df['gender_male'] = (df['gender'] == 'male').astype(int)
df['gender_female'] = (df['gender'] == 'female').astype(int)
df = pd.concat([df, home_dummies, purpose_dummies], axis=1)
df = df.drop(['home_ownership', 'purpose', 'gender'], axis=1)

cols = ['loan_approved'] + [c for c in df.columns if c != 'loan_approved']
df = df[cols]

train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['loan_approved'])
X_train = train.iloc[:, 1:]
y_train = train.iloc[:, 0]
X_test = test.iloc[:, 1:]
y_test = test.iloc[:, 0]

train.to_csv('train.csv', index=False, header=False)
test.to_csv('test.csv', index=False, header=False)
s3.upload_file('train.csv', bucket, f'{prefix}/processed/train/train.csv')
s3.upload_file('test.csv', bucket, f'{prefix}/processed/test/test.csv')

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ─── 1. FULLY MANAGED TRAINING ──────────────────────────────────────────────

print("=" * 60)
print("1. FULLY MANAGED TRAINING")
print("=" * 60)

model = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
)
model.fit(X_train, y_train)
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, preds)
auc = roc_auc_score(y_test, probs)

print(f"Accuracy: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print(classification_report(y_test, preds))

# cloud training job config (ready to submit)
training_job_config = {
    'TrainingJobName': f'loan-xgboost-{int(time.time())}',
    'AlgorithmSpecification': {
        'TrainingImage': f'683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.7-1',
        'TrainingInputMode': 'File'
    },
    'RoleArn': role,
    'InputDataConfig': [{
        'ChannelName': 'train',
        'DataSource': {'S3DataSource': {
            'S3DataType': 'S3Prefix',
            'S3Uri': f's3://{bucket}/{prefix}/processed/train/',
            'S3DataDistributionType': 'FullyReplicated'
        }},
        'ContentType': 'text/csv'
    }],
    'OutputDataConfig': {'S3OutputPath': f's3://{bucket}/{prefix}/models/'},
    'ResourceConfig': {
        'InstanceType': 'ml.m5.large',
        'InstanceCount': 1,
        'VolumeSizeInGB': 10
    },
    'HyperParameters': {
        'objective': 'binary:logistic',
        'num_round': '100',
        'max_depth': '4',
        'eta': '0.1',
        'eval_metric': 'auc'
    },
    'StoppingCondition': {'MaxRuntimeInSeconds': 600}
}

print("\nCloud training job config ready.")
print("Submit with: sm.create_training_job(**training_job_config)")

# ─── 2. DISTRIBUTED TRAINING ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. DISTRIBUTED TRAINING")
print("=" * 60)

chunk_size = len(X_train) // 3
models = []

for i in range(3):
    start = i * chunk_size
    end = start + chunk_size if i < 2 else len(X_train)
    chunk_model = clone(model)
    chunk_model.fit(X_train.iloc[start:end], y_train.iloc[start:end])
    models.append(chunk_model)
    print(f"  Worker {i+1}: trained on {end-start} samples")

ensemble_probs = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
ensemble_preds = (ensemble_probs >= 0.5).astype(int)
ensemble_acc = accuracy_score(y_test, ensemble_preds)

print(f"\nDistributed ensemble accuracy: {ensemble_acc:.4f}")

dist_config = {
    'ResourceConfig': {
        'InstanceType': 'ml.p3.16xlarge',
        'InstanceCount': 4,
        'VolumeSizeInGB': 100
    },
    'HyperParameters': {
        'sagemaker_distributed_dataparallel_enabled': 'true'
    }
}
print(f"Production distributed config: {json.dumps(dist_config, indent=2)}")

# ─── 3. TRAINING COMPILER ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. TRAINING COMPILER")
print("=" * 60)

print("Training Compiler optimizes DL model graphs for GPU.")
print("Not applicable to XGBoost — demonstrating timing comparison.\n")

start_t = time.time()
m1 = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
m1.fit(X_train, y_train)
time_standard = time.time() - start_t

start_t = time.time()
m2 = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42, max_features='sqrt')
m2.fit(X_train, y_train)
time_optimized = time.time() - start_t

print(f"Standard training:  {time_standard:.3f}s")
print(f"Optimized training: {time_optimized:.3f}s")
print(f"Accuracy: {accuracy_score(y_test, m2.predict(X_test)):.4f}")

# ─── 4. AUTOMATIC MODEL TUNING ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. AUTOMATIC MODEL TUNING (HPO)")
print("=" * 60)

param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0]
}

tuner = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=20, cv=3, scoring='roc_auc', random_state=42, n_jobs=-1
)
tuner.fit(X_train, y_train)

print(f"Best params: {tuner.best_params_}")
print(f"Best CV AUC: {tuner.best_score_:.4f}")

tuned_preds = tuner.predict(X_test)
tuned_auc = roc_auc_score(y_test, tuner.predict_proba(X_test)[:, 1])
print(f"Test AUC: {tuned_auc:.4f}")

# cloud HPO config
hpo_config = {
    'Strategy': 'Bayesian',
    'Objective': {'Type': 'Maximize', 'MetricName': 'validation:auc'},
    'ResourceLimits': {'MaxNumberOfTrainingJobs': 20, 'MaxParallelTrainingJobs': 4},
    'ParameterRanges': {
        'IntegerParameterRanges': [
            {'Name': 'max_depth', 'MinValue': '3', 'MaxValue': '10'},
            {'Name': 'num_round', 'MinValue': '50', 'MaxValue': '300'}
        ],
        'ContinuousParameterRanges': [
            {'Name': 'eta', 'MinValue': '0.01', 'MaxValue': '0.3'},
            {'Name': 'subsample', 'MinValue': '0.5', 'MaxValue': '1.0'}
        ]
    }
}
print(f"\nCloud HPO config: {json.dumps(hpo_config, indent=2)}")

# ─── 5. MANAGED SPOT TRAINING ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("5. MANAGED SPOT TRAINING")
print("=" * 60)

spot_additions = {
    'EnableManagedSpotTraining': True,
    'StoppingCondition': {'MaxRuntimeInSeconds': 600, 'MaxWaitTimeInSeconds': 1200},
    'CheckpointConfig': {'S3Uri': f's3://{bucket}/{prefix}/checkpoints/'}
}
print(f"Spot config additions: {json.dumps(spot_additions, indent=2)}")

on_demand_cost = 0.115
spot_cost = on_demand_cost * 0.3
training_hours = 0.5

print(f"\nCost comparison (30 min on ml.m5.large):")
print(f"  On-demand: ${on_demand_cost * training_hours:.3f}")
print(f"  Spot:      ${spot_cost * training_hours:.3f}")
print(f"  Savings:   {(1 - spot_cost/on_demand_cost)*100:.0f}%")

# ─── 6. DEBUGGER AND PROFILER ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("6. DEBUGGER AND PROFILER")
print("=" * 60)

tracemalloc.start()
start_t = time.time()

profiled_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
profiled_model.fit(X_train, y_train)

training_time = time.time() - start_t
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Training time: {training_time:.3f}s")
print(f"Current memory: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

train_acc = accuracy_score(y_train, profiled_model.predict(X_train))
test_acc = accuracy_score(y_test, profiled_model.predict(X_test))
overfit_gap = train_acc - test_acc

print(f"\nDebugger Rules Check:")
print(f"  Train accuracy: {train_acc:.4f}")
print(f"  Test accuracy:  {test_acc:.4f}")
print(f"  Overfit gap:    {overfit_gap:.4f}")
print(f"  {'WARNING: Overfitting detected' if overfit_gap > 0.1 else 'OK: No overfitting'}")

debugger_config = {
    'DebugHookConfig': {
        'S3OutputPath': f's3://{bucket}/{prefix}/debug-output/',
        'CollectionConfigurations': [
            {'CollectionName': 'metrics', 'CollectionParameters': {'save_interval': '10'}},
            {'CollectionName': 'feature_importance'},
            {'CollectionName': 'losses'}
        ]
    },
    'ProfilerConfig': {
        'S3OutputPath': f's3://{bucket}/{prefix}/profiler-output/',
        'ProfilingIntervalInMilliseconds': 500
    }
}
print(f"\nCloud debugger config: {json.dumps(debugger_config, indent=2)}")

# ─── 7. EXPERIMENTS (TRACKING MULTIPLE RUNS) ────────────────────────────────

print("\n" + "=" * 60)
print("7. EXPERIMENTS — Tracking Multiple Runs")
print("=" * 60)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

configs = [
    {'name': 'baseline_lr', 'model': LogisticRegression(max_iter=1000, random_state=42)},
    {'name': 'rf_default', 'model': RandomForestClassifier(n_estimators=100, random_state=42)},
    {'name': 'xgb_default', 'model': GradientBoostingClassifier(n_estimators=100, random_state=42)},
    {'name': 'xgb_tuned', 'model': tuner.best_estimator_},
    {'name': 'rf_deep', 'model': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)},
]

print(f"{'Run Name':<20} {'Accuracy':<12} {'AUC':<12} {'Time':<10}")
print("-" * 54)

experiments = []
for cfg in configs:
    start_t = time.time()
    cfg['model'].fit(X_train_s, y_train)
    t = time.time() - start_t
    preds = cfg['model'].predict(X_test_s)
    probs = cfg['model'].predict_proba(X_test_s)[:, 1]
    a = accuracy_score(y_test, preds)
    au = roc_auc_score(y_test, probs)
    experiments.append({'run': cfg['name'], 'accuracy': a, 'auc': au, 'time': t})
    print(f"  {cfg['name']:<20} {a:<12.4f} {au:<12.4f} {t:<10.3f}s")

exp_df = pd.DataFrame(experiments).sort_values('auc', ascending=False)
print(f"\nBest: {exp_df.iloc[0]['run']} (AUC: {exp_df.iloc[0]['auc']:.4f})")

# ─── 8. CUSTOMIZATION SUPPORT ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("8. CUSTOMIZATION SUPPORT — Framework Integration")
print("=" * 60)

print("Supported frameworks: Scikit-learn, XGBoost, PyTorch, TensorFlow, HuggingFace, MXNet, Spark")

custom_models = {
    'DecisionTree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'NaiveBayes': GaussianNB(),
    'MLP_NeuralNet': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
}

print(f"\n{'Model':<20} {'Accuracy':<12}")
print("-" * 32)
for name, m in custom_models.items():
    m.fit(X_train_s, y_train)
    a = accuracy_score(y_test, m.predict(X_test_s))
    print(f"  {name:<20} {a:.4f}")

# ─── FINAL SUMMARY ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("TRAIN & TUNE — SUMMARY")
print("=" * 70)

print(f"""
| Feature                    | Result                           |
|----------------------------|----------------------------------|
| Fully Managed Training     | AUC: {auc:.4f}                       |
| Distributed Training       | Acc: {ensemble_acc:.4f}                     |
| Training Compiler          | Demonstrated (timing)            |
| Automatic Model Tuning     | AUC: {tuned_auc:.4f}                  |
| Managed Spot Training      | Config ready (70% savings)       |
| Debugger and Profiler      | Gap: {overfit_gap:.4f}                     |
| Experiments                | Best: {exp_df.iloc[0]['run']:<20} |
| Customization Support      | 3 frameworks demonstrated        |
""")
