# Model Card: Decide Decision Engine

## Model Overview
- **Name:** Decide System One Multi-Task Decision Engine
- **Task:** Typed sub-100ms classification and decision making for inbound customer support operations:
  1. **Route (`route`)**: Classify support requests into one of 5 target queues (`billing`, `technical`, `account`, `shipping`, `general`).
  2. **Approve (`approve`)**: Decide whether a customer request can be auto-approved under asymmetric cost penalty (3:1 FP cost vs FN cost).
  3. **Grounded (`grounded`)**: Verify whether an agent's drafted reply is strictly grounded in and supported by a cited knowledge base article.
- **Architecture:** Single shared backbone with task-specific calibrated heads:
  - **Shared Backbone:** `HashingVectorizer` ($2^{17} = 131,072$ features, unigrams and bigrams, unicode accent stripping, lowercased) that maps raw text across all three tasks into a single consistent sparse feature space. Adding new decision tasks requires only training a lightweight linear head on top of the shared feature projection.
  - **Route Head:** Multiclass Calibrated Logistic Regression with Isotonic regression calibration.
  - **Approve Head:** Binary Calibrated Logistic Regression combining the shared text representation with engineered features (tier limits, monetary amounts, interaction ratios, and 18 high-risk security/compliance keyword indicators).
  - **Grounded Head:** Binary Calibrated Logistic Regression combining normalized text representations with structured alignment features (canonical number parsing, duration/time expression extraction, 28 critical entity/role checks, and n-gram overlap metrics).

## Training & Evaluation Data
- **Training Set:** `data/train.jsonl` (6,500 labeled customer support interactions).
- **Development Set:** `data/dev.jsonl` (1,500 labeled customer support interactions; 4,500 evaluated decisions).
- **Pretrained Weights:** None. Fully trained on domain data with zero external API dependencies, running 100% locally on CPU.

## Measured Performance (on `data/dev.jsonl` via `selfcheck.py`)

### Ranked Summary Score: **74.18 / 100**
| Axis | Weight | Score | Contribution |
|---|---|---|---|
| **Quality** | 40% | 0.7489 | **29.96 / 40.00** |
| **Calibration** | 25% | 0.7207 | **18.02 / 25.00** |
| **Latency** | 20% | 1.0000 | **20.00 / 20.00** |
| **Abstention** | 15% | 0.4134 | **6.20 / 15.00** |
| **Total** | 100% | — | **74.18 / 100.00** |

### Per-Decision Metrics
- **Routing:**
  - Macro-F1: **0.8746** (vs majority baseline 0.0835)
  - Accuracy: **0.8453**
  - ECE: **0.0563**
  - Brier Skill: **0.7040**
  - Declared Coverage: **94.3%**
- **Approval (3:1 Asymmetric Cost):**
  - Cost: **0.2352** (vs always-hold 0.3704, always-approve 1.8888)
  - Macro-F1: **0.8913**
  - ECE: **0.0346**
  - Brier Score: **0.0948** (Brier Skill: 0.6090)
  - Declared Coverage: **88.7%**
- **Grounding:**
  - Macro-F1: **0.9139** (vs majority baseline 0.3857)
  - ECE: **0.0400**
  - Brier Score: **0.0784** (Brier Skill: 0.6660)
  - Declared Coverage: **93.9%**

### Latency (Single-threaded, warm, persistent HTTP connection)
- **Route p95:** 16.76 ms (p50: 11.05 ms)
- **Approve p95:** 13.29 ms (p50: 8.57 ms)
- **Grounded p95:** 17.99 ms (p50: 11.68 ms)
- **Mean p95:** **16.01 ms** (passes 100ms hard gate and beats the 25ms full-marks threshold).

### Gates Verification
- `schema_clean`: **PASS** (4,500 / 4,500 valid responses, 0 violations).
- `latency_under_gate`: **PASS** (16.01 ms << 100 ms).
- `reachable`: **PASS** (0 connection drops, 0 transport errors).

## Abstention Policy & Selective Prediction
The engine abstains when its confidence signal reflects high epistemic uncertainty:
- **Route:** Abstains when top predicted class probability $< 0.45$.
- **Approve:** Abstains within an uncertainty band $(0.53 < p < 0.77)$ centered near the asymmetric threshold ($0.65$).
- **Grounded:** Abstains within the boundary band $(0.35 < p < 0.65)$ around the $0.50$ threshold.
- **Coverage Guarantees:** All three decision types maintain declared coverage between **88% and 95%**, reliably exceeding the required 80% coverage floor.

## Failure Modes & Edge Cases
1. **Noisy Ground Truth in Routing:** Real support tickets often involve multi-topic requests (e.g., billing questions about account cancellation). In such cases, the model distributes probability mass across related queues (e.g. `billing` and `account`), properly lowering confidence rather than hallucinating certainty.
2. **Ambiguous Grounding Expressions:** When numerical quantities are expressed in colloquial or relative terms not captured by standard dictionaries (e.g. "a couple of business days"), the model abstains or penalizes confidence.
3. **Hard Ceiling Approvals:** High dollar amounts or requests containing explicit GDPR deletion, 2FA recovery, or account closure keywords are constrained to zero probability of automatic approval to mitigate financial and security risk.
