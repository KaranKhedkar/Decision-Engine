# Decide: Sub-100ms Typed Decision Engine

A production-grade, CPU-optimized decision engine that answers typed customer support queries in sub-25ms latency with calibrated probabilities and reliable abstention policies.

---

## Single Command Startup (For Judges)

To install dependencies and start the service on a clean 4-core Linux box:

```bash
pip install -r requirements.txt && uvicorn app:app --host 0.0.0.0 --port 8000
```

To evaluate the running service on the development dataset with the official scoring harness:

```bash
python selfcheck.py --url http://localhost:8000/v1/decide
```

---

## API Specification & Examples

The service exposes `POST /v1/decide` accepting JSON requests and returning structured, typed answers:

### 1. Route Request Example
```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "demo-route-01",
    "decision": "route",
    "state": {
      "subject": "VAT invoice inquiry",
      "body": "Need receipt for our August subscription payment."
    }
  }'
```
**Response:**
```json
{
  "type": "choice",
  "value": "billing",
  "probs": {
    "billing": 0.832,
    "technical": 0.0,
    "account": 0.098,
    "shipping": 0.070,
    "general": 0.0
  },
  "confidence": 0.832,
  "abstain": false,
  "request_id": "demo-route-01",
  "decision": "route",
  "latency_ms": 7.51
}
```

### 2. Approve Request Example (3:1 Asymmetric Cost)
```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "demo-appr-01",
    "decision": "approve",
    "state": {
      "subject": "Damaged package refund",
      "body": "Item arrived broken, seeking refund.",
      "amount": 15.00,
      "customer_tier": "free"
    }
  }'
```
**Response:**
```json
{
  "type": "boolean",
  "value": true,
  "p_true": 0.768,
  "confidence": 0.768,
  "abstain": false,
  "request_id": "demo-appr-01",
  "decision": "approve",
  "latency_ms": 5.99
}
```

### 3. Grounded Request Example (Verification)
```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "demo-gr-01",
    "decision": "grounded",
    "state": {
      "draft_reply": "Orders are dispatched within 2 business days.",
      "kb_article": "We dispatch via express courier within 2 business days of order."
    }
  }'
```
**Response:**
```json
{
  "type": "boolean",
  "value": true,
  "p_true": 0.996,
  "confidence": 0.996,
  "abstain": false,
  "request_id": "demo-gr-01",
  "decision": "grounded",
  "latency_ms": 6.54
}
```

---

## Architecture & Shared Backbone

The engine uses **one shared feature backbone** that projects incoming text into a common representation space across all three decisions:

1. **Shared Feature Space (The Backbone):**
   - Implemented via a high-dimensional `HashingVectorizer` with $2^{17} = 131,072$ features, capturing unigrams and bigrams with unicode accent stripping and case normalization.
   - All three decisions (`route`, `approve`, and `grounded`) share this exact hash space and n-gram extraction logic, matching the starter repository architecture specification.
   - Sharing occurs at the deterministic feature-projection level (without a heavy multi-task neural encoder), enabling sub-25ms inference on standard CPUs with zero GPU requirements.
   - Adding a fourth or fifth decision requires only training a lightweight linear head on top of the shared feature projection, keeping runtime overhead negligible.

2. **Specialized Calibrated Heads:**
   - **`route`**: Multiclass classification across 5 target queues (`billing`, `technical`, `account`, `shipping`, `general`) using Calibrated Logistic Regression with 5-fold Isotonic calibration.
   - **`approve`**: Binary classification under asymmetric cost penalty ($3:1$ false positive cost vs false negative cost) combining shared text features with tier limits, amount ratios, and domain risk signals.
   - **`grounded`**: Binary verification of whether a drafted reply is supported by a cited knowledge base article, combining the shared text representation with number canonicalization, duration mapping, and entity overlap checks.

---

## Abstention Policy

The engine abstains when its confidence signal reflects high epistemic uncertainty:
- **Route:** Abstains when top predicted probability $< 0.58$.
- **Approve:** Abstains in the critical uncertainty window $(0.50 < p < 0.77)$ around the $0.65$ decision boundary.
- **Grounded:** Abstains within $(0.25 < p < 0.75)$ around the $0.50$ decision boundary.
- **Coverage:** All three decisions achieve declared coverage between **84% and 89%**, well above the $80\%$ minimum floor.

---

## Measured Benchmark Results (`results.json`)

Evaluated on all 1,500 records (4,500 total decisions) of `data/dev.jsonl`:

| Metric / Category | Result | Target / Baseline | Status |
|---|---|---|---|
| **Overall Score** | **75.51 / 100** | — | **Ranked** |
| **Quality Score (40%)** | **0.7775 (31.10 pts)** | Scored vs trivial | **High (+1.14 pts)** |
| - Route Macro-F1 | 0.9125 | vs majority 0.0849 | ✓ |
| - Approve Cost | 0.2282 | vs always-hold 0.3705 | ✓ (3:1 FP cost) |
| - Grounded Macro-F1 | 0.9329 | vs majority 0.3901 | ✓ |
| **Calibration Score (25%)** | **0.7282 (18.21 pts)** | ECE + Brier Skill | **High (+0.19 pts)** |
| - Mean ECE | 0.0411 | Target < 0.20 | ✓ |
| - Mean Brier Skill | 0.6619 | Target > 0.00 | ✓ |
| **Latency Score (20%)** | **1.0000 (20.00 pts)** | Gate 100ms / Full 25ms | **Full Marks** |
| - Route p95 | 21.60 ms | < 25 ms | ✓ |
| - Approve p95 | 16.60 ms | < 25 ms | ✓ |
| - Grounded p95 | 21.88 ms | < 25 ms | ✓ |
| - Mean p95 | **20.03 ms** | < 25 ms | **Full Marks** |
| **Abstention Score (15%)** | **0.4139 (6.21 pts)** | Risk @ 80% coverage | ✓ |
| **Artifacts Total Size** | **~35 MB** | Hard ceiling: 500 MB | ✓ |

### Hard Gates Verification
- `schema_clean`: **PASS** (4,500 / 4,500 valid responses, 0 errors)
- `latency_under_gate`: **PASS** (20.03 ms << 100 ms)
- `reachable`: **PASS** (100% reachable, 0 drops)

---

## Pretrained Components & Licenses

- **Pretrained Language Models / External Weights:** None used.
- All models were trained natively from `data/train.jsonl` using Scikit-Learn algorithms running entirely on CPU.
- **Licenses:**
  - `fastapi`: MIT License
  - `uvicorn`: BSD License
  - `scikit-learn`: BSD License
  - `scipy`: BSD License
  - `numpy`: BSD License
  - `joblib`: BSD License

---

## Retraining the Model

To retrain the models and regenerate all artifacts in `artifacts/`:

```bash
python train.py
```
