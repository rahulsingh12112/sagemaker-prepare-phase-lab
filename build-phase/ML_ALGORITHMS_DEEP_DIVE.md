# ML Algorithms Deep Dive — Theory + Technical Examples

## Sample Data Used in All Examples:

```
Customer A: credit=750, income=80k → Approved (1)
Customer B: credit=550, income=30k → Rejected (0)
Customer C: credit=680, income=60k → ??? (predict this)
```

---

# 1. XGBoost (Extreme Gradient Boosting)

## Theory:

Builds multiple decision trees **sequentially** — each next tree focuses on the **residual error (mistakes)** of the previous tree. Final answer = sum of all tree scores.

**Key concepts:**
- **Boosting** = one after another, keep correcting mistakes
- **Gradient** = measure the direction of error (where it went most wrong)
- **Regularization** = add penalty to prevent overfitting

## Example:

```
Tree 1 (rough prediction):
    credit_score > 650?
    YES → +0.6 (towards approve)
    NO  → -0.4 (towards reject)
    
Customer C: 680 > 650 → YES → score = +0.6
Actual answer = 1, Predicted = 0.6 → Error = 0.4 (still has error)

Tree 2 (trains only on error=0.4):
    income > 50000?
    YES → +0.2
    NO  → -0.1
    
Customer C: 60000 > 50000 → YES → score = +0.2

Final: 0.6 + 0.2 = 0.8 → Sigmoid(0.8) = 0.69 → > 0.5 → APPROVED ✅
```

## When to Use:
- Tabular data + medium/large size
- Need high accuracy
- **Default choice** — when confused, use XGBoost

## Key Hyperparameters:

| Parameter | What it does | Default |
|-----------|-------------|---------|
| `n_estimators` | How many trees to build | 100 |
| `max_depth` | How deep each tree can go | 6 |
| `learning_rate` | How much each tree contributes | 0.3 |
| `subsample` | What % of data each tree gets | 1.0 |

---

# 2. Linear Learner (Logistic/Linear Regression)

## Theory:

Learns a **straight line/plane** that maps features to output. Assigns a **weight** to each feature — important features get higher weights, useless features get lower.

**Key concepts:**
- **Weights** = importance of each feature (model learns these)
- **Bias** = a constant number (intercept)
- **Sigmoid** = converts output to 0-1 range (for classification)
- **Gradient Descent** = slowly adjust weights to minimize error

## Example:

```
After training, model learned this formula:

y = 0.001×income + 0.005×credit_score - 0.3×late_payments - 2.5
    (weight=0.001)  (weight=0.005)      (weight=-0.3)     (bias)

Customer C: income=60000, credit=680, late_payments=2

y = 0.001×60000 + 0.005×680 - 0.3×2 - 2.5
y = 60 + 3.4 - 0.6 - 2.5 = 60.3

Sigmoid(60.3) = 0.99 → 99% confidence → APPROVED ✅

Interpretation:
- income weight is positive → more income = approve
- late_payments weight is negative → more late payments = reject
- credit_score has highest weight → most important feature
```

## When to Use:
- Data is linearly separable (simple patterns)
- Need a fast baseline
- Need interpretability (understand which feature matters most)

## Key Hyperparameters:

| Parameter | What it does |
|-----------|-------------|
| `predictor_type` | 'binary_classifier' or 'regressor' |
| `learning_rate` | How fast it learns |
| `l1/l2 regularization` | Prevent overfitting |
| `epochs` | How many training rounds |

---

# 3. LightGBM (Light Gradient Boosting Machine)

## Theory:

Same gradient boosting as XGBoost — but **2 optimizations** make it faster:

1. **Leaf-wise splitting** — only split the best leaf, not the entire level
2. **Histogram-based** — bucket continuous values into bins for faster split finding

## Example:

```
XGBoost (Level-wise):
Level 0:       [Root]
Level 1:    [Left]  [Right]        ← both split
Level 2: [LL][LR]  [RL][RR]       ← all 4 split (8 computations)

LightGBM (Leaf-wise):
Level 0:       [Root]
Level 1:    [Left]  [Right]        ← only Left split (best gain)
Level 2: [LL][LR]  [Right]        ← only LR split (best gain)
                                    (4 computations — half the work!)

Same depth achieved — but with fewer splits.
Result: FASTER with similar accuracy.
```

## When to Use:
- Large data (100K+ rows)
- Need speed
- XGBoost feels too slow

## XGBoost vs LightGBM:

| | XGBoost | LightGBM |
|--|---------|----------|
| Speed | Slower | **Faster** |
| Small data | **Better** | May overfit |
| Large data | Good | **Better** |
| Default choice | ✅ Safe | ✅ When speed matters |

---

# 4. CatBoost (Categorical Boosting)

## Theory:

Gradient boosting — but **automatically handles categorical features** without one-hot encoding. Uses **Ordered Target Encoding** — prevents data leakage.

## Example:

```
Other models (manual encoding required):
home_ownership = "OWN"  → one_hot → [1, 0, 0]
home_ownership = "RENT" → one_hot → [0, 1, 0]

CatBoost (automatic):
Step 1: Look at average target for "OWN" customers:
        Customer 1 (OWN) → Approved
        Customer 4 (OWN) → Approved  
        Customer 7 (OWN) → Rejected
        Average = (1+1+0)/3 = 0.67

Step 2: Replace "OWN" with 0.67

BUT with ordered encoding — for each row, only uses PREVIOUS rows' average:
        Row 1: no previous OWN → default 0.5
        Row 4: 1 previous OWN (approved) → 1.0
        Row 7: 2 previous OWN (2 approved) → 1.0
        → No data leakage!
```

## When to Use:
- Many categorical columns (text values)
- Don't want to do one-hot encoding manually
- Data has mix of categorical + numerical features

---

# 5. K-NN (K-Nearest Neighbors)

## Theory:

Doesn't learn any formula — **memorizes the entire training data**. For a new point, finds K nearest neighbors and takes majority vote.

**Key concepts:**
- **K** = how many neighbors to check (usually 3, 5, 7)
- **Distance** = Euclidean (straight-line distance)
- **Scaling required** = otherwise large numbers dominate

## Example:

```
K=3, Customer C: income=60000, credit=680, late_payments=2

Calculate distance to all existing customers:
A (Approved): √((80k-60k)² + (750-680)² + (0-2)²) = 20,000.12
B (Rejected): √((30k-60k)² + (550-680)² + (5-2)²) = 30,002.82  
D (Approved): √((65k-60k)² + (700-680)² + (1-2)²) = 5,000.04
E (Rejected): √((25k-60k)² + (500-680)² + (6-2)²) = 35,232.10

3 nearest: D(5000), A(20000), B(30000)
D=Approved, A=Approved, B=Rejected
Majority: 2 Approved vs 1 Rejected → APPROVED ✅

⚠️ PROBLEM: income (60000) dominates the distance calculation!
   Solution: Apply StandardScaler first so all features are in same range.
```

## When to Use:
- Small dataset (<10K rows)
- Simple patterns
- Quick prototype

## Limitations:
- Very slow on large data (calculates distance to every point for each prediction)
- Performs poorly on high-dimensional data

---

# 6. AutoGluon-Tabular

## Theory:

**AutoML** framework — automatically trains multiple models, compares them, builds ensembles, and returns the best. You just provide data.

## Example:

```python
# You write:
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label='loan_approved').fit(train_data)
```

```
Internally this happens:
┌─────────────────────────────────────┐
│ AutoGluon internally:               │
│                                     │
│ 1. XGBoost      → 85.2% accuracy   │
│ 2. LightGBM     → 87.1% accuracy   │
│ 3. CatBoost     → 86.4% accuracy   │
│ 4. Random Forest → 84.0% accuracy  │
│ 5. Neural Net   → 82.3% accuracy   │
│ 6. K-NN         → 78.5% accuracy   │
│                                     │
│ Ensemble (weighted average):        │
│ 0.4×LightGBM + 0.3×CatBoost        │
│ + 0.3×XGBoost = 89.0% 🏆           │
└─────────────────────────────────────┘

Output: Best model or ensemble automatically selected.
```

## When to Use:
- Don't want to think about model selection — just want best result
- Have time for training (tries many models)
- Competitions / POC / prototyping

---

# 7. TabTransformer

## Theory:

Applies **Transformer architecture** (same as used in ChatGPT) to tabular data. Uses **self-attention** to learn relationships between categorical features.

## Example:

```
Step 1: Embedding (categorical → vector)
    "OWN"       → [0.2, 0.8, 0.1, 0.5]
    "education" → [0.9, 0.3, 0.7, 0.2]
    "male"      → [0.4, 0.6, 0.5, 0.8]

Step 2: Self-Attention (learn relationships)
    Attention("OWN", "education") = 0.8 (strong connection)
    Attention("OWN", "male") = 0.2 (weak connection)
    
    → Model learns: "OWN + education loan = safer combination"

Step 3: Numerical features added directly
    [income=60000, credit=680, late=2]

Step 4: Combined vector → MLP → Prediction
    [0.3, 0.7, 0.2, 0.6, 60000, 680, 2] → Neural Network → 0.82 → APPROVED
```

## When to Use:
- Very large data (100K+ rows)
- Complex categorical relationships
- On small data, XGBoost is better — TabTransformer is overkill

---

# 8. Factorization Machines

## Theory:

Linear model + learns **feature interactions**. Each feature has a **latent vector** — dot product of two feature vectors = their interaction strength.

## Example:

```
Linear Model:
y = 0.001×income + 0.005×credit - 0.3×late

Factorization Machines:
y = 0.001×income + 0.005×credit - 0.3×late
    + interaction(income, credit)          ← NEW
    + interaction(income, late)            ← NEW
    + interaction(credit, late)            ← NEW

How interaction is calculated:
    income's vector  = [0.1, 0.3]
    credit's vector  = [0.4, 0.2]
    
    interaction = dot_product = 0.1×0.4 + 0.3×0.2 = 0.10
    
    Meaning: "high income + high credit COMBINATION is extra positive"

Use case: Recommendation systems where data is sparse
    User×Movie matrix has 99% empty cells
    FM can predict those empty cells using learned interactions
```

## When to Use:
- Sparse data (recommendation, CTR prediction)
- Feature combinations are important
- **For loan approval: NOT recommended** — XGBoost is better

---

# Final Summary

| Model | One Line | For Loan Approval Lab |
|-------|---------|----------------------|
| XGBoost | Trees one after another, correcting mistakes | ✅ Primary choice |
| Linear Learner | Divide with a straight line | ✅ Baseline |
| LightGBM | Faster version of XGBoost | ✅ Alternative |
| CatBoost | Auto-handles categorical data | ✅ Good |
| K-NN | Look at nearest neighbors, take majority vote | ⚠️ Only for small data |
| AutoGluon | Try all models, return the best | ✅ Lazy but best |
| TabTransformer | Transformer for tables | ❌ Overkill |
| Factorization Machines | Learns sparse interactions | ❌ Not for this use case |
