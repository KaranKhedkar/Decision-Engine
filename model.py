"""
THIS IS THE DECIDE ENGINE MODEL IMPLEMENTATION.

Architecture:
  - Shared HashingVectorizer transforms text -> sparse features (shared backbone)
  - Task-specific engineered features are appended
  - Calibrated LogisticRegression per task head
  - All three heads share the exact same HashingVectorizer backbone
"""

import json
import math
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import joblib
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import hstack, csr_matrix

QUEUES: List[str] = ["billing", "technical", "account", "shipping", "general"]
TIERS = ["free", "pro", "enterprise"]

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


def normalize_text(text: str) -> str:
    text = text.lower()
    for k, v in NUM_WORDS.items():
        text = re.sub(r'\b' + k + r'\b', v, text)
    return text


def extract_numbers(text: str) -> List[float]:
    text = normalize_text(text)
    return [float(x) for x in re.findall(r'\b\d+(?:\.\d+)?\b', text)]


def extract_times(text: str):
    text = normalize_text(text)
    return set(TIME_PATTERN.findall(text))


class DecisionModel:
    """
    One model, three decision types sharing a single feature backbone.
    """

    def __init__(self, artifacts_dir: str = "artifacts"):
        config_path = os.path.join(artifacts_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = {}

        vec_params = self.config.get("vectorizer_params", {
            "n_features": 2**17,
            "ngram_range": (1, 2),
            "alternate_sign": False,
            "lowercase": True,
            "strip_accents": "unicode",
        })
        self.vectorizer = HashingVectorizer(
            n_features=vec_params.get("n_features", 2**17),
            ngram_range=tuple(vec_params.get("ngram_range", [1, 2])),
            alternate_sign=vec_params.get("alternate_sign", False),
            lowercase=vec_params.get("lowercase", True),
            strip_accents=vec_params.get("strip_accents", "unicode"),
        )

        route_path = os.path.join(artifacts_dir, "route_model.joblib")
        if os.path.exists(route_path):
            self.route_model = joblib.load(route_path)
        else:
            self.route_model = None

        appr_path = os.path.join(artifacts_dir, "approve_model.joblib")
        if os.path.exists(appr_path):
            self.approve_model = joblib.load(appr_path)
        else:
            self.approve_model = None

        grounded_path = os.path.join(artifacts_dir, "grounded_model.joblib")
        if os.path.exists(grounded_path):
            self.grounded_model = joblib.load(grounded_path)
        else:
            self.grounded_model = None

        self.approve_threshold = self.config.get("approve_threshold", 0.65)
        self.tier_limits = self.config.get("tier_limits", TIER_LIMITS)

    # -- shared text feature extraction ------------------------------------
    def _text_features(self, text: str):
        return self.vectorizer.transform([text])

    # -- routing ------------------------------------------------------------
    def route(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, float], float, bool]:
        if self.route_model is None:
            probs = {q: 1.0 / len(QUEUES) for q in QUEUES}
            return QUEUES[0], probs, probs[QUEUES[0]], False

        text = (state.get("subject", "") + " " + state.get("body", "")).strip()
        X = self._text_features(text)

        proba = self.route_model.predict_proba(X)[0]
        classes = self.route_model.classes_

        probs = {}
        for i, cls_idx in enumerate(classes):
            probs[QUEUES[cls_idx]] = float(proba[i])

        for q in QUEUES:
            if q not in probs:
                probs[q] = 0.0

        total = sum(probs.values())
        if total > 0 and abs(total - 1.0) > 0.001:
            probs = {q: p / total for q, p in probs.items()}

        value = max(probs, key=probs.get)
        confidence = probs[value]

        # Abstain on low confidence to maintain ~85% coverage while filtering uncertain predictions
        abstain = confidence < 0.58

        return value, probs, confidence, bool(abstain)

    # -- approval -----------------------------------------------------------
    def approve(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        if self.approve_model is None:
            return False, 0.5, False

        text = (state.get("subject", "") + " " + state.get("body", "")).strip()
        X_text = self._text_features(text)

        feats = self._approve_features(state)
        X = hstack([X_text, csr_matrix([feats])])

        p_true = float(self.approve_model.predict_proba(X)[0, 1])

        value = p_true > self.approve_threshold

        # Abstain in uncertainty band (0.50 - 0.77) around threshold 0.65 to maintain ~88% coverage
        abstain = 0.50 < p_true < 0.77

        return bool(value), float(p_true), bool(abstain)

    # -- grounding ----------------------------------------------------------
    def grounded(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        if self.grounded_model is None:
            return False, 0.5, False

        draft_raw = state.get("draft_reply", "")
        kb_raw = state.get("kb_article", "")
        text = normalize_text(draft_raw) + " [SEP] " + normalize_text(kb_raw)
        X_text = self._text_features(text)

        feats = self._grounded_features(state)
        X = hstack([X_text, csr_matrix([feats])])

        p_true = float(self.grounded_model.predict_proba(X)[0, 1])

        # Hallucination penalty: novel numbers in draft reply strongly indicate ungrounded
        draft_nums = set(extract_numbers(draft_raw))
        kb_nums = set(extract_numbers(kb_raw))
        if len(draft_nums - kb_nums) > 0:
            p_true = min(p_true, 0.10)

        value = p_true > 0.50

        # Abstain in uncertainty band (0.25 - 0.75) to maintain ~87% coverage while filtering hard cases
        abstain = 0.25 < p_true < 0.75

        return bool(value), float(p_true), bool(abstain)

    # -- feature helpers ---------------------------------------------------
    def _approve_features(self, state: Dict[str, Any]) -> List[float]:
        amt = float(state.get("amount", 0.0))
        tier = state.get("customer_tier", "free")
        text = (state.get("subject", "") + " " + state.get("body", "")).lower()

        features = []
        limit = self.tier_limits.get(tier, 50.0)

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

        body = state.get("body", "")
        features.append(len(body))
        features.append(len(body.split()))

        return features

    def _grounded_features(self, state: Dict[str, Any]) -> List[float]:
        raw_draft = state.get("draft_reply", "").lower()
        raw_kb = state.get("kb_article", "").lower()
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
