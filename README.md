# Decide: Sub-100ms Typed Decision Engine

A production-grade, CPU-optimized decision engine that answers typed support desk queries in sub-20ms latency with calibrated probabilities and reliable abstention policies.

---

## Single Command Startup (For Judges)

To run this service on a clean 4-core Linux box (or any standard environment):

```bash
pip install -r requirements.txt && uvicorn app:app --host 0.0.0.0 --port 8000
```

To evaluate the running service on the development dataset with the official scoring harness:

```bash
python selfcheck.py --url http://localhost:8000/v1/decide
```

---

## Architecture & Shared Backbone

The engine uses **one shared feature backbone** that projects incoming text into a common representation space across all three decisions:

1. **Shared Feature Space (The Backbone):**
   - Implemented via a high-dimensional `HashingVectorizer` with $2^{17} = 131,072$ features, capturing unigrams and bigrams with unicode accent stripping and case normalization.
   - All three decisions (`route`, `approve`, and `grounded`) share this exact hash space and n-gram extraction logic, matching the starter repository architecture specification.
   - Sharing occurs at the deterministic feature-projection level (without a heavy multi-task neural encoder), enabling sub-20ms inference on standard CPUs with zero GPU requirements.
   - Adding a fourth or fifth decision requires only training a lightweight linear head on top of the shared feature projection, keeping runtime overhead negligible.

2. **Specialized Calibrated Heads:**
   - **`route`**: Multiclass classification across 5 target queues (`billing`, `technical`, `account`, `shipping`, `general`) using Calibrated Logistic Regression with Isotonic regression.
   - **`approve`**: Binary classification under asymmetric cost penalty ($3:1$ false positive cost vs false negative cost) combining shared text features with tier limits, amount ratios, and domain risk signals.
   - **`grounded`**: Binary verification of whether a drafted reply is supported by a cited knowledge base article, combining the shared text representation with number canonicalization, duration mapping, and entity overlap checks.

---

## Abstention Policy

The engine abstains when its confidence signal reflects high epistemic uncertainty:
- **Route:** Abstains when top predicted probability $< 0.45$.
- **Approve:** Abstains in the critical uncertainty window $(0.53 < p < 0.77)$ around the $0.65$ decision boundary.
- **Grounded:** Abstains within $(0.35 < p < 0.65)$ around the $0.50$ decision boundary.
- **Coverage:** All three decisions achieve declared coverage between **88% and 95%**, well above the $80\%$ minimum floor.

---

## Measured Benchmark Results (`results.json`)

Evaluated on all 1,500 records (4,500 total decisions) of `data/dev.jsonl`:

| Metric / Category | Result | Target / Baseline | Status |
|---|---|---|---|
| **Overall Score** | **74.18 / 100** | — | Ranked |
| **Quality Score (40%)** | **0.7489 (29.96 pts)** | Scored vs trivial | High |
| - Route Macro-F1 | 0.8746 | vs majority 0.0835 | ✓ |
| - Approve Cost | 0.2352 | vs always-hold 0.3704 | ✓ (3:1 FP cost) |
| - Grounded Macro-F1 | 0.9139 | vs majority 0.3857 | ✓ |
| **Calibration Score (25%)** | **0.7207 (18.02 pts)** | ECE + Brier Skill | High |
| - Mean ECE | 0.0437 | Target < 0.20 | ✓ |
| - Mean Brier Skill | 0.6598 | Target > 0.00 | ✓ |
| **Latency Score (20%)** | **1.0000 (20.00 pts)** | Gate 100ms / Full 25ms | **Full Marks** |
| - Route p95 | 16.76 ms | < 25 ms | ✓ |
| - Approve p95 | 13.29 ms | < 25 ms | ✓ |
| - Grounded p95 | 17.99 ms | < 25 ms | ✓ |
| - Mean p95 | **16.01 ms** | < 25 ms | **Full Marks** |
| **Abstention Score (15%)** | **0.4134 (6.20 pts)** | Risk @ 80% coverage | ✓ |
| **Artifacts Total Size** | **~42 MB** | Hard ceiling: 500 MB | ✓ |

### Hard Gates Verification
- `schema_clean`: **PASS** (4,500 / 4,500 valid responses, 0 errors)
- `latency_under_gate`: **PASS** (16.01 ms << 100 ms)
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
