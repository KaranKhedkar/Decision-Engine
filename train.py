"""
Training script for the Decide engine.

Shared architecture:
  - HashingVectorizer on text -> shared sparse feature matrix (shared backbone)
  - Task-specific engineered features appended
  - Calibrated LogisticRegression per task head (all share the hashed text backbone)

Calibration:
  - 5-fold cross-validation with isotonic regression calibrator

Usage:
  python train.py
"""

import json
import math
import os
import re
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from scipy.sparse import hstack, csr_matrix
import joblib

QUEUES = ["billing", "technical", "account", "shipping", "general"]
QUEUE_TO_IDX = {q: i for i, q in enumerate(QUEUES)}
TIERS = ["free", "pro", "enterprise"]
TIER_TO_IDX = {t: i for i, t in enumerate(TIERS)}
FP_COST = 3.0
ARTIFACTS_DIR = "artifacts"

NUM_WORDS = {
    'one thousand': '1000', 'half an hour': '30 minutes', 'a quarter of an hour': '15 minutes',
    'an hour': '60 minutes', 'one hour': '60 minutes', 'two hours': '120 minutes',
    'a week': '7 days', 'one week': '7 days', 'two weeks': '14 days',
    'a month': '30 days', 'one month': '30 days', 'two months': '60 days',
    'six months': '180 days', 'a year': '365 days', 'one year': '365 days',
    'fourteen days': '14 days', 'thirty days': '30 days', 'sixty days': '60 days',
    'ninety days': '90 days', 'forty-eight hours': '48 hours', 'twenty-four hours': '24 hours',
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
    'fifteen': '15', 'twenty': '20', 'thirty': '30', 'forty': '40',
    'fifty': '50', 'sixty': '60', 'seventy': '70', 'eighty': '80',
    'ninety': '90', 'hundred': '100', 'thousand': '1000'
}

TIME_PATTERN = re.compile(
    r'\b(\d+)\s*(day|days|hour|hours|minute|minutes|week|weeks|month|months|year|years|business day|business days)\b'
)

ROLES = [
    'all users', 'administrators', 'workspace owners', 'all subscribers', 'annual subscribers',
    'monthly subscribers', 'pro and enterprise', 'starter keys', 'sandbox keys', 'production keys',
    'all plans', 'free plan', 'pro plan', 'enterprise plan', 'domestic orders', 'all orders',
    'international orders', 'csv and json', 'csv only', 'json only', 'signed download link',
    'card on file', 'original payment method', 'store credit', 'sms code', 'recovery codes',
    'authenticator app', 'email link', 'standard courier', 'express partner'
]

SECURITY_KEYWORDS = [
    'authenticator', 'codes', 'two-factor', 'device', 'lost', 'backup', 'permanently',
    'disable', 'partnership', 'close', 'reseller', 'gdpr', 'erasure', 'personal',
    'deleted', 'dispute', 'exception', 'outside', 'usual policy'
]

TIER_LIMITS = {"free": 31.10, "pro": 121.29, "enterprise": 499.45}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_text(text):
    text = text.lower()
    for k, v in NUM_WORDS.items():
        text = re.sub(r'\b' + k + r'\b', v, text)
    return text


def extract_numbers(text):
    text = normalize_text(text)
    return [float(x) for x in re.findall(r'\b\d+(?:\.\d+)?\b', text)]


def extract_times(text):
    text = normalize_text(text)
    return set(TIME_PATTERN.findall(text))


def build_approve_features(record):
    s = record["state"]
    amt = float(s.get("amount", 0.0))
    tier = s.get("customer_tier", "free")
    text = (s.get("subject", "") + " " + s.get("body", "")).lower()

    features = []
    limit = TIER_LIMITS.get(tier, 50.0)

    features.append(amt)
    features.append(math.log1p(amt))
    features.append(1.0 if amt == 0.0 else 0.0)
    features.append(1.0 if amt > limit else 0.0)
    features.append(amt / max(limit, 1.0))
    features.append(max(0.0, limit - amt))

    for t in TIERS:
        features.append(1.0 if tier == t else 0.0)

    sec_count = 0
    for w in SECURITY_KEYWORDS:
        if w in text:
            sec_count += 1
            features.append(1.0)
        else:
            features.append(0.0)
    features.append(sec_count)
    features.append(1.0 if sec_count > 0 else 0.0)

    body = s.get("body", "")
    features.append(len(body))
    features.append(len(body.split()))

    return features


def build_grounded_features(record):
    s = record["state"]
    raw_draft = s.get("draft_reply", "").lower()
    raw_kb = s.get("kb_article", "").lower()
    draft = normalize_text(raw_draft)
    kb = normalize_text(raw_kb)

    features = []
    draft_nums = set(extract_numbers(draft))
    kb_nums = set(extract_numbers(kb))
    novel_nums = draft_nums - kb_nums
    features.append(len(novel_nums))
    features.append(1.0 if novel_nums else 0.0)
    features.append(len(kb_nums - draft_nums))
    features.append(len(draft_nums))
    features.append(len(kb_nums))
    features.append(len(draft_nums & kb_nums) / len(draft_nums) if draft_nums else 1.0)

    d_words = set(draft.split())
    k_words = set(kb.split())
    features.append(len(d_words & k_words) / len(d_words) if d_words else 1.0)
    features.append(len(d_words & k_words) / len(k_words) if k_words else 1.0)
    features.append(len(d_words - k_words))

    d_bigrams = set(zip(draft.split()[:-1], draft.split()[1:]))
    k_bigrams = set(zip(kb.split()[:-1], kb.split()[1:]))
    features.append(len(d_bigrams & k_bigrams) / len(d_bigrams) if d_bigrams else 1.0)
    features.append(len(d_bigrams - k_bigrams))

    d_times = extract_times(draft)
    k_times = extract_times(kb)
    novel_times = d_times - k_times
    features.append(len(novel_times))
    features.append(1.0 if novel_times else 0.0)

    phrase_mismatches = 0
    for p in ROLES:
        if p in raw_draft and p not in raw_kb:
            phrase_mismatches += 1
            features.append(1.0)
        else:
            features.append(0.0)
    features.append(phrase_mismatches)
    features.append(1.0 if phrase_mismatches > 0 else 0.0)

    features.append(len(draft) / max(len(kb), 1.0))
    features.append(abs(len(draft) - len(kb)))

    return features


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("Loading data...")
    train_data = load_jsonl("data/train.jsonl")
    dev_data = load_jsonl("data/dev.jsonl")
    print(f"  Train: {len(train_data)}, Dev: {len(dev_data)}")

    # Shared text representation
    print("\nBuilding shared text features...")
    shared_vectorizer = HashingVectorizer(
        n_features=2**17,
        ngram_range=(1, 2),
        alternate_sign=False,
        lowercase=True,
        strip_accents="unicode",
    )

    train_route_text = [(r['state'].get('subject', '') + ' ' + r['state'].get('body', '')).strip() for r in train_data]
    dev_route_text = [(r['state'].get('subject', '') + ' ' + r['state'].get('body', '')).strip() for r in dev_data]

    train_grounded_text = [normalize_text(r['state'].get('draft_reply', '') + ' [SEP] ' + r['state'].get('kb_article', '')) for r in train_data]
    dev_grounded_text = [normalize_text(r['state'].get('draft_reply', '') + ' [SEP] ' + r['state'].get('kb_article', '')) for r in dev_data]

    X_train_route_text = shared_vectorizer.transform(train_route_text)
    X_dev_route_text = shared_vectorizer.transform(dev_route_text)

    X_train_gr_text = shared_vectorizer.transform(train_grounded_text)
    X_dev_gr_text = shared_vectorizer.transform(dev_grounded_text)

    # Task engineered features
    print("Building task-specific features...")
    train_appr_f = np.array([build_approve_features(r) for r in train_data])
    dev_appr_f = np.array([build_approve_features(r) for r in dev_data])

    train_gr_f = np.array([build_grounded_features(r) for r in train_data])
    dev_gr_f = np.array([build_grounded_features(r) for r in dev_data])

    X_train_route = X_train_route_text
    X_dev_route = X_dev_route_text

    X_train_approve = hstack([X_train_route_text, csr_matrix(train_appr_f)])
    X_dev_approve = hstack([X_dev_route_text, csr_matrix(dev_appr_f)])

    X_train_grounded = hstack([X_train_gr_text, csr_matrix(train_gr_f)])
    X_dev_grounded = hstack([X_dev_gr_text, csr_matrix(dev_gr_f)])

    # Labels
    y_route_train = np.array([QUEUE_TO_IDX[r["labels"]["route"]] for r in train_data])
    y_route_dev = np.array([QUEUE_TO_IDX[r["labels"]["route"]] for r in dev_data])

    y_approve_train = np.array([1 if r["labels"]["approve"] else 0 for r in train_data])
    y_approve_dev = np.array([1 if r["labels"]["approve"] else 0 for r in dev_data])

    y_grounded_train = np.array([1 if r["labels"]["grounded"] else 0 for r in train_data])
    y_grounded_dev = np.array([1 if r["labels"]["grounded"] else 0 for r in dev_data])

    # 1. Route Head
    print("\n" + "="*60)
    print("TRAINING: Route Head")
    print("="*60)
    route_lr = LogisticRegression(C=3.0, max_iter=300, solver='lbfgs')
    route_cal = CalibratedClassifierCV(route_lr, method='isotonic', cv=5)
    route_cal.fit(X_train_route, y_route_train)
    route_probs = route_cal.predict_proba(X_dev_route)
    route_preds = np.argmax(route_probs, axis=1)
    print(f"  Dev macro-F1: {f1_score(y_route_dev, route_preds, average='macro'):.4f}")

    # 2. Approve Head
    print("\n" + "="*60)
    print("TRAINING: Approve Head")
    print("="*60)
    appr_lr = LogisticRegression(C=2.0, max_iter=300, solver='lbfgs')
    appr_cal = CalibratedClassifierCV(appr_lr, method='isotonic', cv=5)
    appr_cal.fit(X_train_approve, y_approve_train)
    appr_probs = appr_cal.predict_proba(X_dev_approve)[:, 1]

    best_cost = float('inf')
    best_threshold = 0.65
    for t in np.linspace(0.5, 0.85, 36):
        preds = (appr_probs > t).astype(int)
        cost = sum(FP_COST if p == 1 and y == 0 else (1.0 if p == 0 and y == 1 else 0.0) for p, y in zip(preds, y_approve_dev)) / len(y_approve_dev)
        if cost < best_cost:
            best_cost = cost
            best_threshold = t
    print(f"  Best cost: {best_cost:.4f} at threshold {best_threshold:.3f}")

    # 3. Grounded Head
    print("\n" + "="*60)
    print("TRAINING: Grounded Head")
    print("="*60)
    gr_lr = LogisticRegression(C=2.0, max_iter=300, solver='lbfgs')
    gr_cal = CalibratedClassifierCV(gr_lr, method='isotonic', cv=5)
    gr_cal.fit(X_train_grounded, y_grounded_train)
    gr_probs = gr_cal.predict_proba(X_dev_grounded)[:, 1]

    for i, r in enumerate(dev_data):
        d_nums = set(extract_numbers(r['state'].get('draft_reply', '')))
        k_nums = set(extract_numbers(r['state'].get('kb_article', '')))
        if len(d_nums - k_nums) > 0:
            gr_probs[i] = min(gr_probs[i], 0.10)

    gr_preds = (gr_probs > 0.5).astype(int)
    print(f"  Dev macro-F1: {f1_score(y_grounded_dev, gr_preds, average='macro'):.4f}")

    # Save artifacts
    print("\n" + "="*60)
    print("SAVING ARTIFACTS")
    print("="*60)
    config = {
        "vectorizer_params": {
            "n_features": 2**17,
            "ngram_range": (1, 2),
            "alternate_sign": False,
            "lowercase": True,
            "strip_accents": "unicode",
        },
        "approve_threshold": float(best_threshold),
        "queues": QUEUES,
        "tiers": TIERS,
        "tier_limits": TIER_LIMITS,
    }
    with open(os.path.join(ARTIFACTS_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    joblib.dump(route_cal, os.path.join(ARTIFACTS_DIR, "route_model.joblib"))
    joblib.dump(appr_cal, os.path.join(ARTIFACTS_DIR, "approve_model.joblib"))
    joblib.dump(gr_cal, os.path.join(ARTIFACTS_DIR, "grounded_model.joblib"))

    print("Saved all artifacts successfully!")


if __name__ == "__main__":
    main()
