# Three-Minute Live Demo Script: Decide Engine

**Target Time:** 3 Minutes  
**Endpoint:** `http://localhost:8000/v1/decide`  
**Repository:** `https://github.com/KaranKhedkar/Decision-Engine.git`

---

### [0:00 – 0:30] Problem & Core Motivation
> *"Hi judges. Large Language Models today are called for tiny, repetitive decisions—routing tickets, auto-approving refunds, checking factual consistency. Each call takes 1 to 2 seconds, costs real dollars, and provides no usable confidence signal.*
>
> *We built **Decide**: a System One typed decision engine that answers all three decision types inline on pure CPU in under 20 milliseconds, with rigorously calibrated probabilities and an explicit policy to abstain when uncertain."*

---

### [0:30 – 1:15] Architecture: Shared Backbone & Calibrated Heads
> *"Our core constraint was: one model, not three, running strictly on CPU.*
>
> *1. **The Shared Backbone:** All three decisions share a single high-dimensional `HashingVectorizer` backbone ($2^{17} = 131,072$ features with unigrams and bigrams). This projects any incoming customer request into a common sparse representation. Adding a fourth decision in the future requires only fitting a new linear head.*
>
> *2. **Calibrated Heads:** On top of this shared projection, we run task-specific calibrated heads:*
> *- **Route:** Multiclass Logistic Regression calibrated with 5-fold isotonic regression.*
> *- **Approve:** Binary classification under an asymmetric 3:1 false-positive penalty, combining shared text features with tier thresholds and 18 security/refusal indicators.*
> *- **Grounded:** Verification combining text alignment with canonical number normalization and entity overlap."*

---

### [1:15 – 2:00] Live Demonstration: 3 Tasks & Selective Abstention

*(Execute the following live curl commands or run `python demo.py`)*

#### Demo 1: Routing to Billing Queue
```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{"request_id":"demo-1","decision":"route","state":{"subject":"VAT invoice","body":"Need our monthly receipt."}}'
```
> *"Returns `value: billing` with 83.2% confidence in ~7 ms. The probability distribution over all 5 queues is normalized and calibrated."*

#### Demo 2: Auto-Approval Under 3:1 Asymmetric Cost
```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{"request_id":"demo-2","decision":"approve","state":{"subject":"Refund","body":"Damaged goods","amount":15.0,"customer_tier":"free"}}'
```
> *"Small damaged goods claim within the Free tier limit. Output: `value: true`, `p_true: 0.768`."*

#### Demo 3: Factual Grounding Verification
```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{"request_id":"demo-3","decision":"grounded","state":{"draft_reply":"Dispatches in two business days.","kb_article":"Standard orders dispatch within 2 business days."}}'
```
> *"Canonical number normalizer maps 'two business days' to '2 business days'. Output: `value: true`, `p_true: 0.996`."*

#### Demo 4: The Abstention Policy in Action (Learning to Reject)
```bash
curl -X POST http://localhost:8000/v1/decide \
  -H "Content-Type: application/json" \
  -d '{"request_id":"demo-4","decision":"approve","state":{"subject":"Special Request","body":"Looking for an exception on my enterprise plan.","amount":450.0,"customer_tier":"enterprise"}}'
```
> *"Here, the amount ($450) is near the enterprise boundary and mentions policy exception. The model computes `p_true = 0.61`, which falls directly inside the critical uncertainty band (0.50 to 0.77). Output: `abstain: true`. Under our 3:1 asymmetric cost penalty, a false approval costs three times as much as a false hold—so the engine safely routes borderline uncertainty to human review.*
>
> *Our calibrated thresholds across all three tasks are:*
> *- **Route:** Abstain when top probability $< 0.58$ (coverage: 84.6%).*
> *- **Approve:** Abstain when $0.50 < p_{true} < 0.77$ (coverage: 88.5%).*
> *- **Grounded:** Abstain when $0.25 < p_{true} < 0.75$ (coverage: 87.1%).*
> *All three maintain safe coverage above the 80% penalty floor."*

---

### [2:00 – 2:30] Official Measured Benchmark Results
> *"Here are our scores measured by the official `selfcheck.py` over 4,500 requests on `data/dev.jsonl`:*
>
> - **Total Score:** **75.51 / 100**
> - **Quality (40%):** **31.10 / 40.00** *(Route Macro-F1: 0.9125, Approve Cost: 0.2282 vs 0.3705 hold baseline, Grounded Macro-F1: 0.9329)*
> - **Calibration (25%):** **18.21 / 25.00** *(Mean ECE: 0.0411, Brier Skill: 0.6619)*
> - **Latency (20%):** **20.00 / 20.00 (Full Marks)** *(Route p95: 21.6 ms, Approve p95: 16.6 ms, Grounded p95: 21.9 ms; Mean p95 is 20.03 ms, well below the 25 ms full-marks threshold)*
> - **Abstention (15%):** **6.21 / 15.00** *(Selective risk at 80% coverage: Route 0.070, Approve 0.081, Grounded 0.062; declared coverage 84%–89%)*
> - **All Gates Passed:** 100% schema clean (4,500/4,500), 0 connection errors, and ~35 MB artifacts (limit 500 MB)."*

---

### [2:30 – 3:00] Core Trade-Offs & Conclusion
> *"Building Decide required balancing four competing forces: accuracy, calibrated confidence, selective abstention, and CPU latency:*
>
> *1. **CPU Latency vs Deep Architectures:** We chose a high-dimensional feature-hashing backbone ($2^{17}$) rather than a multi-hundred-megabyte transformer. On a standard 4-core CPU without a GPU, a neural transformer lands at 150–300 ms, failing the hard latency gate. Our shared hash projection runs in 15–22 ms p95, securing full marks on latency (20/20).*
>
> *2. **Accuracy vs Cost & Calibration:** Under the 3:1 asymmetric approval penalty, raw accuracy is deceptive. By retraining with cost-sensitive class weights and 5-fold isotonic calibration, we reduced false approvals from 102 to 86 and lowered approval cost to 0.2282, achieving a 0.0269 ECE.*
>
> *3. **Abstention vs Coverage:** Selective prediction prunes the high-uncertainty tail from answered decisions, boosting answered quality. By carefully sizing the abstention bands around the critical decision boundaries, we maintain 84%–89% coverage—reaping quality gains while staying comfortably above the 80% penalty floor.*
>
> *Code, model card, and benchmark results are pushed and public at `github.com/KaranKhedkar/Decision-Engine`. The service is running live and ready for automated judging. Thank you!"*
