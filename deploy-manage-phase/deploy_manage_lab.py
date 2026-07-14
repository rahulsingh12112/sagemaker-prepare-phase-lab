"""
Deploy & Manage Phase — SageMaker Lab
Covers: Real-Time Inference, Serverless, Async, Batch Transform,
        Multi-Model, Multi-Container, Shadow Testing, Inference Recommender,
        Model Monitor, Kubernetes, Edge Manager
"""

import boto3
import pandas as pd
import numpy as np
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

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

# ─── TRAIN A MODEL FIRST ─────────────────────────────────────────────────────

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

model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)
print(f"Model trained. Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")

# upload test data for batch transform
test.iloc[:, 1:].to_csv('batch_input.csv', index=False, header=False)
s3.upload_file('batch_input.csv', bucket, f'{prefix}/batch-input/batch_input.csv')

# ─── 1. REAL-TIME INFERENCE ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("1. REAL-TIME INFERENCE — Endpoint Deployment")
print("=" * 60)

endpoint_config = {
    'EndpointConfigName': f'loan-rt-config-{int(time.time())}',
    'ProductionVariants': [{
        'VariantName': 'primary',
        'ModelName': 'loan-approval-model',
        'InstanceType': 'ml.m5.large',
        'InitialInstanceCount': 1,
        'InitialVariantWeight': 1.0
    }]
}

print("Real-Time Endpoint Config:")
print(json.dumps(endpoint_config, indent=2))
print("\nUse when: Steady traffic, need millisecond latency")
print("Billing: Per hour (instance always running)")

sample = X_test.iloc[0:1]
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]
print(f"\nSample prediction: {'APPROVED' if prediction == 1 else 'REJECTED'} (prob: {probability:.3f})")

# ─── 2. SERVERLESS INFERENCE ─────────────────────────────────────────────────

print("\n" + "=" * 60)
print("2. SERVERLESS INFERENCE")
print("=" * 60)

serverless_config = {
    'EndpointConfigName': f'loan-serverless-config-{int(time.time())}',
    'ProductionVariants': [{
        'VariantName': 'primary',
        'ModelName': 'loan-approval-model',
        'ServerlessConfig': {
            'MemorySizeInMB': 2048,
            'MaxConcurrency': 5
        }
    }]
}

print("Serverless Endpoint Config:")
print(json.dumps(serverless_config, indent=2))
print("\nUse when: Intermittent/unpredictable traffic")
print("Billing: Per request only (scales to zero when idle)")
print("Cold start: ~1-2 seconds on first request after idle")

# ─── 3. ASYNCHRONOUS INFERENCE ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("3. ASYNCHRONOUS INFERENCE")
print("=" * 60)

async_config = {
    'EndpointConfigName': f'loan-async-config-{int(time.time())}',
    'ProductionVariants': [{
        'VariantName': 'primary',
        'ModelName': 'loan-approval-model',
        'InstanceType': 'ml.m5.large',
        'InitialInstanceCount': 1
    }],
    'AsyncInferenceConfig': {
        'OutputConfig': {
            'S3OutputPath': f's3://{bucket}/{prefix}/async-output/',
            'NotificationConfig': {
                'SuccessTopic': f'arn:aws:sns:{region}:{account_id}:loan-async-success',
                'ErrorTopic': f'arn:aws:sns:{region}:{account_id}:loan-async-error'
            }
        },
        'ClientConfig': {
            'MaxConcurrentInvocationsPerInstance': 4
        }
    }
}

print("Async Endpoint Config:")
print(json.dumps(async_config, indent=2, default=str))
print("\nUse when: Large payloads, processing > 60 seconds")
print("Flow: Send request → get token → poll S3 for result")

# ─── 4. BATCH TRANSFORM ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("4. BATCH TRANSFORM")
print("=" * 60)

batch_config = {
    'TransformJobName': f'loan-batch-{int(time.time())}',
    'ModelName': 'loan-approval-model',
    'TransformInput': {
        'DataSource': {
            'S3DataSource': {
                'S3DataType': 'S3Prefix',
                'S3Uri': f's3://{bucket}/{prefix}/batch-input/'
            }
        },
        'ContentType': 'text/csv',
        'SplitType': 'Line'
    },
    'TransformOutput': {
        'S3OutputPath': f's3://{bucket}/{prefix}/batch-output/',
        'AssembleWith': 'Line'
    },
    'TransformResources': {
        'InstanceType': 'ml.m5.large',
        'InstanceCount': 1
    }
}

print("Batch Transform Config:")
print(json.dumps(batch_config, indent=2))
print("\nUse when: Offline predictions on large datasets")
print("No endpoint needed — runs, predicts, saves to S3, shuts down")

batch_preds = model.predict(X_test)
print(f"\nBatch predictions (simulated): {len(batch_preds)} rows")
print(f"Approved: {sum(batch_preds)}, Rejected: {len(batch_preds) - sum(batch_preds)}")

# ─── 5. MULTI-MODEL ENDPOINTS ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("5. MULTI-MODEL ENDPOINTS")
print("=" * 60)

multi_model_config = {
    'ModelName': 'loan-multi-model',
    'Containers': [{
        'Image': f'683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.7-1',
        'ModelDataUrl': f's3://{bucket}/{prefix}/models/',
        'Mode': 'MultiModel'
    }],
    'ExecutionRoleArn': role
}

print("Multi-Model Config:")
print(json.dumps(multi_model_config, indent=2))
print("\n1 endpoint serves multiple models (100s possible)")
print("Dynamic loading — only active models in memory")
print("Invoke: smr.invoke_endpoint(TargetModel='model_v2.tar.gz', ...)")

# ─── 6. MULTI-CONTAINER ENDPOINTS ───────────────────────────────────────────

print("\n" + "=" * 60)
print("6. MULTI-CONTAINER ENDPOINTS")
print("=" * 60)

multi_container_config = {
    'ModelName': 'loan-pipeline-model',
    'Containers': [
        {
            'ContainerHostname': 'preprocessor',
            'Image': f'683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3',
            'ModelDataUrl': f's3://{bucket}/{prefix}/models/preprocessor.tar.gz'
        },
        {
            'ContainerHostname': 'predictor',
            'Image': f'683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.7-1',
            'ModelDataUrl': f's3://{bucket}/{prefix}/models/model.tar.gz'
        }
    ],
    'InferenceExecutionConfig': {'Mode': 'Serial'},
    'ExecutionRoleArn': role
}

print("Multi-Container Config (Serial Pipeline):")
print(json.dumps(multi_container_config, indent=2))
print("\nContainer 1 (preprocess) → Container 2 (predict)")

# ─── 7. SHADOW TESTING ──────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("7. SHADOW TESTING")
print("=" * 60)

shadow_config = {
    'EndpointName': 'loan-rt-endpoint',
    'ProductionVariants': [{
        'VariantName': 'production',
        'ModelName': 'loan-model-v1',
        'InstanceType': 'ml.m5.large',
        'InitialInstanceCount': 1
    }],
    'ShadowProductionVariants': [{
        'ShadowModelVariantName': 'shadow',
        'ModelName': 'loan-model-v2',
        'InstanceType': 'ml.m5.large',
        'InitialInstanceCount': 1,
        'SamplingPercentage': 50
    }]
}

print("Shadow Testing Config:")
print(json.dumps(shadow_config, indent=2))
print("\nProduction serves V1, Shadow runs V2 on same traffic")
print("Shadow responses discarded — no customer impact")
print("Compare metrics → promote if V2 better")

# ─── 8. INFERENCE RECOMMENDER ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("8. INFERENCE RECOMMENDER")
print("=" * 60)

recommender_config = {
    'JobName': f'loan-recommender-{int(time.time())}',
    'JobType': 'Default',
    'RoleArn': role,
    'InputConfig': {
        'ModelPackageVersionArn': f'arn:aws:sagemaker:{region}:{account_id}:model-package/loan-model/1',
        'JobDurationInSeconds': 7200
    }
}

print("Inference Recommender Config:")
print(json.dumps(recommender_config, indent=2, default=str))
print("\nSageMaker tests multiple instance types automatically")
print("Returns: Best instance + latency + cost recommendation")

# ─── 9. MODEL MONITOR ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("9. MODEL MONITOR")
print("=" * 60)

monitor_config = {
    'MonitoringScheduleName': 'loan-monitor-schedule',
    'MonitoringScheduleConfig': {
        'ScheduleConfig': {
            'ScheduleExpression': 'cron(0 * ? * * *)'
        },
        'MonitoringJobDefinition': {
            'MonitoringInputs': [{
                'EndpointInput': {
                    'EndpointName': 'loan-rt-endpoint',
                    'LocalPath': '/opt/ml/processing/input'
                }
            }],
            'MonitoringOutputConfig': {
                'MonitoringOutputs': [{
                    'S3Output': {
                        'S3Uri': f's3://{bucket}/{prefix}/monitor-output/',
                        'LocalPath': '/opt/ml/processing/output'
                    }
                }]
            },
            'MonitoringResources': {
                'ClusterConfig': {
                    'InstanceCount': 1,
                    'InstanceType': 'ml.m5.large',
                    'VolumeSizeInGB': 20
                }
            },
            'RoleArn': role
        }
    }
}

print("Model Monitor Config:")
print(json.dumps(monitor_config, indent=2))

# simulate drift detection
train_mean = X_train.mean()
test_mean = X_test.mean()
drift = abs(train_mean - test_mean) / (train_mean + 1e-10)
drifted_features = drift[drift > 0.1]

print(f"\nDrift Detection (simulated):")
print(f"  Features checked: {len(drift)}")
print(f"  Features with drift > 10%: {len(drifted_features)}")
if len(drifted_features) > 0:
    print(f"  Drifted: {list(drifted_features.index[:5])}")
else:
    print(f"  No significant drift detected")

# ─── 10. KUBERNETES & KUBEFLOW ───────────────────────────────────────────────

print("\n" + "=" * 60)
print("10. KUBERNETES & KUBEFLOW INTEGRATION")
print("=" * 60)

k8s_manifest = f"""apiVersion: sagemaker.aws.amazon.com/v1
kind: TrainingJob
metadata:
  name: loan-training
spec:
  algorithmSpecification:
    trainingImage: 683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-xgboost:1.7-1
    trainingInputMode: File
  roleArn: {role}
  outputDataConfig:
    s3OutputPath: s3://{bucket}/{prefix}/models/
  resourceConfig:
    instanceType: ml.m5.large
    instanceCount: 1
    volumeSizeInGB: 10"""

print("Kubernetes Manifest for SageMaker Training:")
print(k8s_manifest)
print("\nApply with: kubectl apply -f training-job.yaml")

# ─── 11. EDGE MANAGER ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("11. EDGE MANAGER")
print("=" * 60)

edge_config = {
    'DeviceFleetName': 'loan-edge-fleet',
    'RoleArn': role,
    'OutputConfig': {
        'S3OutputPath': f's3://{bucket}/{prefix}/edge-output/'
    }
}

print("Edge Fleet Config:")
print(json.dumps(edge_config, indent=2))
print("\nFlow: Compile (Neo) → Package → Deploy to devices → Monitor")
print("Supported: ARM, x86, Jetson, Raspberry Pi")

# ─── FINAL SUMMARY ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("DEPLOY & MANAGE — SUMMARY")
print("=" * 70)

print("""
| Deployment Type          | Use When                         | Billing          |
|--------------------------|----------------------------------|------------------|
| Real-Time Endpoint       | Steady traffic, low latency      | Per hour         |
| Serverless               | Intermittent traffic             | Per request      |
| Asynchronous             | Large payloads, long processing  | Per hour + queue |
| Batch Transform          | Offline bulk predictions         | Per job duration |
| Multi-Model              | Many models, cost optimize       | 1 instance       |
| Multi-Container          | Pipeline (preprocess + predict)  | 1 instance       |
| Shadow Testing           | Validate new model in prod       | 2x instance cost |
| Inference Recommender    | Find optimal instance type       | One-time job     |
| Model Monitor            | Detect drift, maintain accuracy  | Scheduled jobs   |
| Kubernetes/Kubeflow      | K8s-native teams                 | K8s + SageMaker  |
| Edge Manager             | IoT/Edge inference               | Per device       |
""")
