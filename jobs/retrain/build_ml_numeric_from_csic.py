import logging
import os
import re
import math
import logging
import collections
import pandas as pd
import kagglehub

from urllib.parse import urlparse, parse_qs
from kagglehub import KaggleDatasetAdapter

log = logging.getLogger(__name__)

# Path to save the output ML numeric features CSV
OUT_PATH = os.getenv("ML_NUM_FILE", "./data/ml_numeric.csv")

# By default, pull datasets from Kaggle:
USE_KAGGLE = os.getenv("USE_KAGGLE_CSIC", "1") in ("1", "true", "True", "yes", "on")
KAGGLE_FILE_PATH = os.getenv(
    "CSIC_KAGGLE_FILE",
    "csic_database.csv",
)

# Local CSV is used only if USE_KAGGLE_CSIC=0 is explicitly set
LOCAL_CSIC_PATH = os.getenv("CSIC_LOCAL_PATH", "./data/csic_database.csv")
HEADER_COLS = [
    "User-Agent", "Pragma", "Cache-Control", "Accept",
    "Accept-encoding", "Accept-charset", "language",
    "host", "cookie", "content-type", "connection"
]

# HTTP method mapping
METHOD_MAP = {
    "GET": 1,
    "POST": 2,
    "HEAD": 3,
    "PUT": 4,
    "DELETE": 5,
    "OPTIONS": 6,
    "TRACE": 7,
    "CONNECT": 8,
}

# Suspicious characters set
SUSPICIOUS_CHARS = set("<>\"'`;$(){}[]\\|&")


def load_csic_df() -> pd.DataFrame:
    """
    Loads the CSIC dataset from Kaggle or local disk.
    
    Returns:
        pd.DataFrame: The loaded CSIC dataset.
    """
    if USE_KAGGLE:
        if kagglehub is None:
            raise RuntimeError("USE_KAGGLE_CSIC=1, but kagglehub is not installed")
        if not KAGGLE_FILE_PATH:
            raise RuntimeError("CSIC_KAGGLE_FILE must be set (relative path in dataset)")
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "ispangler/csic-2010-web-application-attacks",
            KAGGLE_FILE_PATH,
        )
        return df

    # Load from local path
    if not os.path.exists(LOCAL_CSIC_PATH):
        raise FileNotFoundError(f"CSIC_LOCAL_PATH does not exist: {LOCAL_CSIC_PATH}")
    return pd.read_csv(LOCAL_CSIC_PATH)

def char_entropy(s: str) -> float:
    """
    Calculates the character-level entropy of a string.
    
    Args:
        s (str): Input string.
    Returns:
        float: Character-level entropy.
    """
    if not s:
        return 0.0
    cnt = collections.Counter(s)
    n = len(s)
    return float(-sum((c/n) * math.log2(c/n) for c in cnt.values()))

def suspicious_char_ratio_from_string(s: str) -> float:
    """
    Calculates the ratio of suspicious characters in a string.
    
    Args:
        s (str): Input string.
    Returns:
        float: Ratio of suspicious characters.
    """
    if not s:
        return 0.0
    sus = sum(ch in SUSPICIOUS_CHARS for ch in s)
    return float(sus / max(1, len(s)))

def parse_payload_size(lenght_value) -> int:
    """
    Parses the payload size from a string value.
    
    Args:
        lenght_value: The length value to parse.
    Returns:
        int: Parsed payload size.
    """
    if not isinstance(lenght_value, str):
        return 0
    m = re.search(r"(\d+)", lenght_value)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except ValueError:
        return 0

def http_method_code(method_value) -> int:
    """
    Maps HTTP method string to an integer code.
    
    Args:
        method_value: The HTTP method string.
    Returns:
        int: Corresponding method code.
    """
    if not isinstance(method_value, str):
        return 0
    m = method_value.strip().upper()
    return METHOD_MAP.get(m, 0)

def is_json_payload(content_type_value) -> int:
    """
    Checks if the content type indicates a JSON payload.
    
    Args:
        content_type_value: The content type string.
    Returns:
        int: 1 if JSON payload, else 0.
    """
    if not isinstance(content_type_value, str):
        return 0
    return 1 if "json" in content_type_value.lower() else 0

def featurize_row(row) -> dict:
    """
    Featurizes a single row from the CSIC dataset.
    Args:
        row: A row from the CSIC dataset.
    Returns:
        dict: A dictionary of extracted features.
    """
    url = row.get("URL", "") or ""
    content = row.get("content", "") or ""

    # URL parts
    parsed = urlparse(str(url))
    path = parsed.path or "/"
    endpoint_len = len(path)
    path_depth = sum(1 for seg in path.split("/") if seg)

    query = parse_qs(parsed.query, keep_blank_values=True)
    query_param_count = len(query)

    # headers
    headers_values = []
    header_count = 0
    for col in HEADER_COLS:
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            headers_values.append(v.strip())
            header_count += 1

    headers_blob = " ".join(headers_values)
    headers_entropy = char_entropy(headers_blob)

    # payload size
    payload_size = parse_payload_size(row.get("lenght"))

    # suspicious chars
    combined = (str(url) or "") + " " + (str(content) or "")
    suspicious_ratio = suspicious_char_ratio_from_string(combined)

    # method
    method_code = http_method_code(row.get("Method"))

    # is_json
    is_json = is_json_payload(row.get("content-type"))

    # label
    cls = row.get("classification", None)
    try:
        label = int(cls)
    except Exception:
        label = None

    return {
        "endpoint_len": endpoint_len,
        "header_count": header_count,
        "payload_size": payload_size,
        "headers_entropy": headers_entropy,
        "query_param_count": query_param_count,
        "path_depth": path_depth,
        "suspicious_char_ratio": suspicious_ratio,
        "is_json_payload": is_json,
        "http_method_code": method_code,
        "label": label,
    }

def main():
    """
    Main function to build ML numeric features from the CSIC dataset.

    Returns:
        None
    """
    df = load_csic_df()
    rows = []
    for _, r in df.iterrows():
        feats = featurize_row(r)
        rows.append(feats)

    out_df = pd.DataFrame(rows)

    feature_cols = [
        "endpoint_len",
        "header_count",
        "payload_size",
        "headers_entropy",
        "query_param_count",
        "path_depth",
        "suspicious_char_ratio",
        "is_json_payload",
        "http_method_code",
    ]
    cols = feature_cols + (["label"] if "label" in out_df.columns else [])
    out_df = out_df[cols]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    if os.path.exists(OUT_PATH):
        # add to existing file
        base_df = pd.read_csv(OUT_PATH)

        missing = [c for c in cols if c not in base_df.columns]
        if missing:
            raise ValueError(
                f"[csic→ml_numeric] existing {OUT_PATH} has no columns {missing}; "
                f"have={base_df.columns.tolist()}"
            )

        base_df = base_df[cols]
        merged = pd.concat([base_df, out_df], ignore_index=True)
    else:
        # new file
        merged = out_df

    merged.to_csv(OUT_PATH, index=False)
    log.info(f"[csic→ml_numeric] saved {out_df.shape[0]} new rows; total={merged.shape[0]} -> {OUT_PATH}")
    log.info(f"[csic→ml_numeric] columns: {merged.columns.tolist()}")

if __name__ == "__main__":
    main()