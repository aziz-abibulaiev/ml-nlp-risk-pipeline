"""Fixed, declared hyperparameters for the content detector (D10: no autotune).

Nothing in this project may grid-search, random-search, or Optuna-tune these. They are
stated here once and used identically everywhere. Changing a value is a deliberate,
reviewable edit — not something a training run discovers.
"""

SEED = 42

# ---- shared normalization ----
DECODE_MAX_PASSES = 3  # unquote_plus to fixed point, capped (fixes the double-encoding blind spot)

# ---- p_text: char n-gram TF-IDF -> linear ----
# Two SEPARATE text fields, each its own vectorizer, so their weights differ (D16.3):
#   field A = decode_fixed_point(url_path_raw + url_query + request_body)
#   field B = decode_fixed_point(user_agent + referer)
TFIDF = dict(analyzer="char", ngram_range=(3, 5), min_df=2, lowercase=False, sublinear_tf=True)
# max_features caps memory; generous and fixed, not tuned.
TFIDF_MAX_FEATURES_A = 200_000
TFIDF_MAX_FEATURES_B = 50_000
LOGREG = dict(class_weight="balanced", max_iter=2000, C=1.0, solver="liblinear", random_state=SEED)

# ---- p_numeric: shape statistics -> gradient boosting ----
HGB = dict(
    learning_rate=0.1, max_iter=300, max_depth=None, max_leaf_nodes=31,
    l2_regularization=1.0, early_stopping=False, random_state=SEED,
)

# ---- fusion ----
# Declared rule: mean of logits of p_text and p_numeric, then a 1-D Platt calibration fit
# ONLY on the calibration slice. Interpretable; each channel's contribution stays measurable.
FUSION = "mean_logit"
FUSION_WEIGHTS = (0.5, 0.5)  # (text, numeric)

# ---- operating point ----
# FPR-budget threshold chosen on the CALIBRATION slice only; never on test.
FPR_BUDGETS = (0.01, 0.05)
PRIMARY_FPR = 0.01

# ---- family head (testbed-only) ----
HGB_FAMILY = dict(
    learning_rate=0.1, max_iter=300, max_depth=None, max_leaf_nodes=31,
    l2_regularization=1.0, early_stopping=False, random_state=SEED,
)

# clean cross-tool families (0 payload overlap between custom and ffuf) — D15 / 6-quater
CLEAN_XTOOL_FAMILIES = ["sqli", "xss", "cmd_injection", "ssti"]
# families whose ffuf slice is off-family content (relabeled) -> binary only, no per-family holdout
BINARY_ONLY_FFUF_OFFCONTENT = ["nosql_injection", "ssrf"]
# families with a small shared-payload intersection to drop before per-family holdout
DROP_SHARED_BEFORE_XTOOL = ["path_traversal", "scan"]

CONTENT_FAMILIES = ["sqli", "xss", "cmd_injection", "nosql_injection", "ldap_injection",
                    "ssti", "path_traversal", "ssrf", "scan"]
BEHAVIOURAL_FAMILIES = ["brute_force", "cred_stuffing", "resource_exhaustion"]
CONTEXT_FAMILIES = ["bola_idor", "mass_assignment", "business_logic_abuse"]

MODEL_DIR = "models/content"
