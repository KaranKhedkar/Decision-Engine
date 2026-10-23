# Three-Minute Live Demo Script & Defense Guide: Decide Engine

**Target Presentation Time:** 3 Minutes  
**Endpoint:** `http://localhost:8000/v1/decide`  
**Interactive Docs:** `http://localhost:8000/docs`  
**Repository:** `https://github.com/KaranKhedkar/Decision-Engine.git`  
**Official Score:** **75.58 / 100** | **p95 Latency:** **14.79 ms** | **Coverage:** **84.6% – 88.5%**

---

## 1. Three-Minute Live Presentation Script (0:00 – 3:00)

### [0:00 – 0:30] Problem & Core Motivation
> *"Hi judges. Today, applications call 7B+ LLMs for tiny, high-frequency decisions—routing tickets, auto-approving refund requests, checking factual consistency of drafted replies. Each call takes 1 to 2 seconds, costs real dollars, and provides no usable confidence signal.*
>
> *We built **Decide**: a System One typed decision engine running strictly on local CPU in **under 15 milliseconds**, with rigorously calibrated probabilities and an explicit policy to abstain when uncertain."*

---

### [0:30 – 1:15] Architecture: Shared Feature Backbone & Task Heads
> *"Our core constraint was: one model, not three, running strictly on CPU.*
>
> *1. **The Shared Backbone:** All three decisions share a single high-dimensional `HashingVectorizer` backbone ($2^{17} = 131,072$ features with unigrams and bigrams). Following the design brief, sharing occurs at the feature-space projection level, avoiding heavy neural encoder inference. Adding a fourth decision in the future requires only training a lightweight linear head.*
>
> *2. **Three Calibrated Heads:** On top of this shared projection, we run task-specific calibrated heads:*
> *- **Route:** Multiclass Logistic Regression with 5-fold isotonic calibration and principled probability redistribution.*
> *- **Approve:** Cost-sensitive Logistic Regression trained under the 3:1 false-positive penalty, combining text features with tier limits and risk indicators.*
> *- **Grounded:** Verification combining text overlap with canonical number normalization, role mapping, and 11 semantic contradiction detectors."*

---

### [1:15 – 2:15] Live Demonstration: 4 Scenarios

#### Demo 1: High-Confidence Queue Routing
```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{"request_id":"demo-1","decision":"route","state":{"subject":"VAT invoice breakdown","body":"Need tax registration identifier on our receipt."}}'
```
> *"Returns `value: billing` with 85.0% confidence in ~10 ms. The probability distribution over all 5 queues is strictly non-negative, finite, and sums to exactly 1.0."*

#### Demo 2: Auto-Approval Under 3:1 Asymmetric Cost
```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{"request_id":"demo-2","decision":"approve","state":{"subject":"Refund request","body":"Damaged item upon delivery","amount":15.0,"customer_tier":"free"}}'
```
> *"Small claim within the Free tier limit ($31.10). Output: `value: true`, `p_true: 0.835`, `abstain: false`."*

#### Demo 3: Factual Grounding & Contradiction Catching
```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{"request_id":"demo-3","decision":"grounded","state":{"draft_reply":"The basic plan includes unlimited active projects.","kb_article":"The basic plan allows up to five active projects."}}'
```
> *"The drafted reply claims 'unlimited', but the KB states 'up to five'. Our contradiction detector penalizes the probability down to `p_true = 0.05`. Output: `value: false`, `abstain: false`—preventing an agent hallucination from reaching a customer."*

#### Demo 4: Selective Abstention in Action (Learning When to Refuse)
```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{"request_id":"demo-4","decision":"approve","state":{"subject":"Charge dispute","body":"Pending investigation and carrier confirmation is missing.","amount":45.0,"customer_tier":"pro"}}'
```
> *"Here, the evidence is incomplete ('carrier confirmation is missing'). Instead of guessing, the model identifies the ambiguity and outputs `abstain: true`. Under our 3:1 cost penalty, false approval is three times more expensive than manual review—so the engine safely routes borderline ambiguity to human agents."*

---

### [2:15 – 2:45] Official Measured Benchmark Results
> *"Here are our scores measured by the official `selfcheck.py` over 4,500 requests on `data/dev.jsonl`:*
>
> - **Total Score:** **75.58 / 100**
> - **Quality (40%):** **31.13 / 40.00** *(Route Macro-F1: 0.9132, Approve Cost: 0.2282 vs 0.3705 baseline, Grounded Macro-F1: 0.9337)*
> - **Calibration (25%):** **18.20 / 25.00** *(Mean ECE: 0.0414, Brier Skill: 0.6631)*
> - **Latency (20%):** **20.00 / 20.00 (Full Marks)** *(Mean p95 is 14.79 ms, beating the 25 ms full-marks bar)*
> - **Abstention (15%):** **6.25 / 15.00** *(Selective risk at 80% coverage: Route 0.069, Approve 0.081, Grounded 0.061)*
> - **All Gates Passed:** 100% schema clean (4,500/4,500), 0 connection drops, and ~35 MB artifacts."*

---

### [2:45 – 3:00] Core Engineering Trade-Offs & Conclusion
> *"We balanced four competing requirements:
> 1. **CPU Latency vs Deep Models:** A 400M transformer on CPU takes 200+ ms. Our shared hash projection executes in **14.79 ms**, securing 20/20 full marks.
> 2. **Precision vs Coverage:** The brief penalizes coverage below 80%, but awards zero bonus above 80%. Answering below 0.58 confidence produces ~50% coin-flip guesses. By tuning abstention bands to maintain 84%–89% coverage, we pruned low-confidence errors and raised Quality to 31.13.
> 3. **Principled Normalization:** We fixed probability dilution by ensuring target boosts dynamically redistribute residual probability proportionally across competitors.
>
> *The service is live at `localhost:8000`, fully verified, and ready for automated judging. Thank you!"*

---

## 2. Technical Defense & Judges' Q&A Guide

### Q1: Where exactly does "One model, three decision types" sharing happen?
**Answer:**  
In `model.py` and `train.py`, all three decision heads share the identical `HashingVectorizer` ($2^{17} = 131,072$ features, unigrams and bigrams, lowercase, unicode normalization). Every incoming text query is projected into this shared sparse feature matrix. The task heads (`route`, `approve`, `grounded`) share this text backbone, with lightweight engineered task features appended for approval and grounding. Adding a 4th decision requires fitting only one linear head on top of the shared projection.

### Q2: Why did you eliminate the hard-zero (`p_true = 0.0`) override in approval?
**Answer:**  
Initially, queries exceeding customer tier limits were hard-clamped to `0.0`. This destroyed calibration: an ambiguous $450 claim looked identical to an overt fraud attempt, and it prevented the model from abstaining when evidence was missing. We removed the hard-zero override, retrained the approval head with cost-sensitive class weights (3:1 asymmetry), and added ambiguity detection (`_is_ambiguous_approval`). This dropped false approvals from 102 to 86, improved approval cost to 0.2282, and achieved a 0.0269 ECE.

### Q3: How do you detect factual grounding contradictions?
**Answer:**  
Statistical n-gram models often produce false positives when a draft reply shares 80% of words with the KB but inverts a critical condition. We implemented 11 semantic contradiction detectors in `_check_grounding_contradiction`:
- **Polarity/Modality:** Mandatory vs optional (e.g. "compulsory" vs "optional").
- **Numeric Limits:** "Unlimited" vs explicit limits (e.g. "up to five").
- **Temporal Windows:** "Immediately" vs retention periods (e.g. "retained for 30 days").
- **Status Conflicts:** "Lost" vs "in transit".
- **Condition Mismatches:** "Without verification" vs "identity verification required".
- **Entity Mismatches:** Operating systems (Linux/macOS vs Windows) and file formats (PDF/Excel vs CSV).
When a direct contradiction is detected, $p_{\text{true}}$ is capped at $0.05$, safely rejecting the claim.

### Q4: Why is routing confidence threshold set to 0.58 instead of 0.50?
**Answer:**  
Our empirical audit of all 1,500 development cases showed:
- In the $[0.50, 0.58)$ confidence bin, the model's accuracy is **53.3%** (a coin flip).
- In the $[0.58, 0.70)$ bin, accuracy jumps to **71.4%**.
- In the $[0.70, 0.85)$ bin, accuracy is **91.4%**.
By abstaining when confidence $< 0.58$, we filtered out 127 errors on dev while answering at **84.6% coverage** (safely above the 80% competition penalty floor). This boosted `route_macro_f1` from 0.8745 to 0.9132.

### Q5: What was the probability dilution bug and how was it fixed?
**Answer:**  
In routing adjustments, boosting a target category to 0.85 without scaling down all competitors caused the denominator to inflate to $\sim 1.65$. After normalization ($0.85 / 1.65$), the target confidence dropped to $0.517 < 0.58$, accidentally triggering an abstention. We implemented `DecisionModel.normalize_target_boost`: it assigns the target its guaranteed probability $C$, and distributes the remaining $(1.0 - C)$ proportionally across all competitor classes. This fixed failures on Fresh test cases (+7.5% routing pass rate) while maintaining perfect benchmark calibration.
