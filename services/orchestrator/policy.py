# services/orchestrator/policy.py
from typing import Any, Dict, List, Optional
import re, time, json, os

DEFAULT_POLICY: Dict[str, Any] = {
  "defaults": {
    "latency_budget_ms": 80,
    "ml_models": ["if","ocsvm","ae","lstm_ae"],  # IF + OCSVM + AE + LSTM-AE
    "nlp_models": ["rules","tfidf_lr"],   # regex guardrail + TF-IDF+LR
    "fusion": {"mode":"weighted", "alpha":0.6, "beta":0.4},
    "thresholds": {"ml":0.7, "nlp":0.6, "final":0.7},
    "shadow": {"nlp": [], "ml": []}
  },
  "routes": [
    # Web access logs
    { "match": {"log_kind":"access_log", "endpoint_regex": r"^/login|/account"},
      "latency_budget_ms": 60,
      "ml_models": ["if","ocsvm","ae","lstm_ae"],
      "nlp_models": ["rules","cnn_char"],
      "thresholds": {"final": 0.65},
      "fusion": {"mode":"weighted", "alpha":0.5, "beta":0.5}
    },
    { "match": {"log_kind":"access_log", "suspect_regex": r"select|union|<script"},
      "latency_budget_ms": 90,
      "nlp_models": ["rules","bert_cls"],    # важча модель при підозрі
      "shadow": {"nlp":["tfidf_lr"]}         # збираємо офлайн метрики
    },

    # AWS CloudTrail
    { "match": {"log_kind":"cloudtrail", "endpoint_regex": r"^/ec2.amazonaws.com:StartInstances"},
      "latency_budget_ms": 120,
      "ml_models": ["if","ocsvm","ae","lstm_ae"],      # кілька класиків + LSTM-AE
      "nlp_models": ["rules","tfidf_lr"],    # userAgent + params
      "thresholds": {"final": 0.6}
    },
    { "match": {"log_kind":"cloudtrail", "user_no_mfa": True},
      "latency_budget_ms": 120,
      "ml_models": ["if","ocsvm","ae","lstm_ae"],
      "nlp_models": ["rules","bert_cls"],
      "shadow": {"ml": []}                   # тимчасово без ML у тіні (всі моделі в основному тракті)"
    }
  ]
}

def _match(route:Dict[str,Any], tf:Dict[str,str], num:Dict[str,float]) -> bool:
    m = route.get("match",{})
    # log_kind
    if "log_kind" in m:
        if tf.get("log_kind") != m["log_kind"]:
            return False
    # endpoint pattern
    if "endpoint_regex" in m and not re.search(m["endpoint_regex"], tf.get("endpoint","")):
        return False
    # швидка підозра за regex у тексті
    if "suspect_regex" in m:
        blob = " ".join([tf.get("payload",""), tf.get("headers",""), tf.get("endpoint","")])
        if not re.search(m["suspect_regex"], blob, flags=re.I):
            return False
    # CloudTrail спец-ознаки:
    if "user_no_mfa" in m and m["user_no_mfa"] is True:
        # мапимо це в preprocess: tf["mfa"]="true|false"
        if tf.get("mfa","").lower() != "false":
            return False
    return True

def pick_route(policy:Dict[str,Any], tf:Dict[str,str], num:Dict[str,float]) -> Dict[str,Any]:
    for r in policy.get("routes",[]):
        if _match(r, tf, num):
            return r
    return policy["defaults"]

# (опційно) гарячий reload із S3/FS — пропускаю деталізацію; можемо зберігати останній успішний.