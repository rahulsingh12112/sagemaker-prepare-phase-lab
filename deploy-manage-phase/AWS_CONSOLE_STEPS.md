# Deploy & Manage Phase — AWS Console Steps

## 1. Fully Managed Deployment — Real-Time Endpoint

**Path:** AWS Console → SageMaker → Deployments & inference → Endpoints → Create endpoint

1. Endpoint name: `loan-approval-endpoint`
2. Create endpoint configuration:
   - Config name: `loan-approval-config`
   - Add model: Select your trained model (from Training jobs output)
   - Instance type: `ml.m5.large`
   - Initial instance count: 1
3. Click "Create endpoint"
4. Wait for status: InService
5. Test: Send a CSV payload to get predictions

---

## 2. Real-Time Inference

**Path:** Same endpoint as above — always running, low latency

- Use when: Steady, predictable traffic
- Latency: Milliseconds
- Billing: Per hour (instance always on)
- Scale: Auto-scaling policies can be attached

---

## 3. Serverless Inference

**Path:** AWS Console → SageMaker → Deployments & inference → Endpoints → Create endpoint

1. Same steps as above BUT in endpoint configuration:
   - Choose "Serverless" instead of real-time
   - Memory: 2048 MB
   - Max concurrency: 5
2. No instance runs when idle — scales to zero
3. Use when: Intermittent/unpredictable traffic
4. Billing: Per request (pay only when invoked)

---

## 4. Asynchronous Inference

**Path:** AWS Console → SageMaker → Deployments & inference → Endpoints → Create endpoint

1. Endpoint configuration → Select "Asynchronous"
2. Configure:
   - S3 output path for results
   - SNS notification topic (optional — get notified when done)
   - Max concurrent invocations
3. Use when: Large payloads, long processing (>60 sec)
4. Client sends request → gets back a token → polls for result in S3

---

## 5. Batch Transform

**Path:** AWS Console → SageMaker → Deployments & inference → Batch transform jobs → Create

1. Job name: `loan-batch-predict`
2. Model: Select trained model
3. Input:
   - S3 path: `s3://{bucket}/loan-approval-lab/processed/test/test.csv`
   - Content type: `text/csv`
   - Split type: Line
4. Output:
   - S3 path: `s3://{bucket}/loan-approval-lab/batch-output/`
5. Instance: `ml.m5.large`, count: 1
6. Click "Create job"
7. No endpoint needed — runs, predicts, saves to S3, shuts down

---

## 6. Multi-Model Endpoints

**Path:** Same as endpoint creation, but model configuration different

- Upload multiple model.tar.gz files to same S3 prefix
- Single endpoint serves ALL models
- SageMaker dynamically loads/unloads models based on traffic
- Cost saving: 1 instance serves 100s of models

---

## 7. Multi-Container Endpoints

**Path:** Endpoint configuration → Add multiple containers

- Multiple containers on same instance (e.g., preprocessing + model)
- Serial inference pipeline OR direct invocation
- Use when: Need different frameworks for different steps

---

## 8. Shadow Testing

**Path:** AWS Console → SageMaker → Deployments & inference → Shadow tests

1. Select production endpoint (existing)
2. Select shadow variant (new model to test)
3. Configure traffic split (e.g., 10% to shadow)
4. Shadow variant gets real traffic but responses not sent to client
5. Compare metrics between production and shadow
6. If shadow performs better → promote to production

---

## 9. Inference Recommender

**Path:** AWS Console → SageMaker → Deployments & inference → Inference Recommender

1. Select model
2. Run recommendation job
3. SageMaker tests multiple instance types automatically
4. Returns: Best instance type + expected latency + cost
5. Use this to pick optimal deployment config

---

## 10. Model Monitor

**Path:** AWS Console → SageMaker → Deployments & inference → Model monitoring

1. Create monitoring schedule on your endpoint
2. Types:
   - Data quality monitoring (input drift)
   - Model quality monitoring (accuracy drift)
   - Bias drift monitoring
   - Feature attribution drift
3. Baseline: Set from training data
4. Schedule: Hourly/Daily
5. Alerts: CloudWatch alarms when drift detected

---

## 11. Kubernetes & Kubeflow Integration

- SageMaker Operators for Kubernetes
- Deploy SageMaker models from K8s cluster
- Use kubectl to manage SageMaker resources
- Use when: Team already on Kubernetes, want SageMaker managed training/inference

---

## 12. Edge Manager

**Path:** AWS Console → SageMaker → Edge Manager

- Package models for edge devices (IoT, mobile)
- Deploy to edge fleet
- Monitor edge model performance from cloud
- Use when: Need inference at edge (no internet)

---

## Console Quick Reference

| Task | Console Path |
|------|-------------|
| Create Real-Time Endpoint | SageMaker → Endpoints → Create |
| Create Serverless Endpoint | SageMaker → Endpoints → Create (serverless config) |
| Batch Transform | SageMaker → Batch transform jobs → Create |
| Shadow Test | SageMaker → Shadow tests → Create |
| Inference Recommender | SageMaker → Inference Recommender |
| Model Monitor | SageMaker → Model monitoring → Create schedule |
| Check Endpoint Status | SageMaker → Endpoints → Click name |
| View Invocation Logs | CloudWatch → Log groups → /aws/sagemaker/Endpoints |
