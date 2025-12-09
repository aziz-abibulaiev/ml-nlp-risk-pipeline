import os
import re
import math
import logging
import collections
import pandas as pd
import kagglehub

from urllib.parse import urlparse, parse_qs

log = logging.getLogger(__name__)

# Path to save the output ML numeric features CSV
OUT_PATH = os.getenv("ML_NUM_FILE", "./data/ml_numeric.csv")

# By default, pull datasets from Kaggle:
USE_KAGGLE_ATTACKLAB = os.getenv("USE_KAGGLE_ATTACKLAB", "1") in ("1", "true", "True", "yes", "on")
ATTACKLAB_KAGGLE_FILE = os.getenv("ATTACKLAB_KAGGLE_FILE", "attack-simulation-alert.csv")

# Local CSV is used only if you explicitly set USE_KAGGLE_ATTACKLAB=0
LOCAL_ATTACKLAB_PATH = os.getenv("ATTACKLAB_LOCAL_PATH", "./data/attacklab.csv")

# HTTP headers present in the attack-lab CSV
HEADER_COLS = [
    "http.user_agent",
    "http.cookie",
    "http.host",
    "http.referer",
    "http.content_type",
    "http.content_length_header",
    "http.authorization",
    "http.connection",
    "http.location",
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

SUSPICIOUS_CHARS = set("<>\"'`;$(){}[]\\|&")


def get_attacklab_csv_path() -> str:
    """
    Gets the path to the attack-lab CSV file, either from Kaggle or local disk.
    
    Returns:
        str: Path to the attack-lab CSV file.
    """
    if USE_KAGGLE_ATTACKLAB:
        if kagglehub is None:
            raise RuntimeError("USE_KAGGLE_ATTACKLAB=1, but kagglehub is not installed")
        if not ATTACKLAB_KAGGLE_FILE:
            raise RuntimeError("ATTACKLAB_KAGGLE_FILE must be set (relative path in dataset)")

        # Download the dataset to disk, without using pandas
        root = kagglehub.dataset_download("moradrawashdeh/attack-simulation-lab")
        csv_path = os.path.join(root, ATTACKLAB_KAGGLE_FILE)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Не знайшов {ATTACKLAB_KAGGLE_FILE} у {root}")
        return csv_path

    # fallback – local path
    if not os.path.exists(LOCAL_ATTACKLAB_PATH):
        raise FileNotFoundError(f"ATTACKLAB_LOCAL_PATH does not exist: {LOCAL_ATTACKLAB_PATH}")
    return LOCAL_ATTACKLAB_PATH

def char_entropy(s: str) -> float:
    """
    Calculates the character-level entropy of a string.
    
    Args:
        s (str): Input string.
    Returns:
        float: Character entropy.
    """
    if not s:
        return 0.0
    cnt = collections.Counter(s)
    n = len(s)
    return float(-sum((c / n) * math.log2(c / n) for c in cnt.values()))

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

def parse_payload_size(content_len_value) -> int:
    """
    Parses the payload size from the Content-Length header value.
    
    Args:
        content_len_value: The Content-Length header value.
    Returns:
        int: Parsed payload size, or 0 if invalid.
    """
    if not isinstance(content_len_value, str):
        return 0
    m = re.search(r"(\d+)", content_len_value)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except ValueError:
        return 0

def http_method_code(method_value) -> int:
    """
    Maps HTTP method string to its corresponding code.
    
    Args:
        method_value: The HTTP method string.
    Returns:
        int: Corresponding method code, or 0 if unknown.
    """
    if not isinstance(method_value, str):
        return 0
    m = method_value.strip().upper()
    return METHOD_MAP.get(m, 0)

def is_json_payload(content_type_value) -> int:
    """
    Determines if the Content-Type indicates a JSON payload.
    
    Args:
        content_type_value: The Content-Type header value.
    Returns:
        int: 1 if JSON payload, 0 otherwise.
    """
    if not isinstance(content_type_value, str):
        return 0
    return 1 if "json" in content_type_value.lower() else 0

def label_from_alert(alert_value):
    """
    Converts alert string to label: "benign" -> 0, others -> 1.
    
    Args:
        alert_value: The alert string.
    Returns:
        int or None: 0 for "benign", 1 for others, None if invalid.
    """
    if not isinstance(alert_value, str):
        return None
    s = alert_value.strip().lower()
    if not s:
        return None
    return 0 if s == "benign" else 1

def build_url_from_row(row) -> str:
    """
    Constructs the full URL from the row data.
    
    Args:
        row: The input data row.
    Returns:
        str: Constructed full URL.
    """
    host_val = row.get("http.host")
    host = host_val.strip() if isinstance(host_val, str) else ""

    uri_val = row.get("http.request.uri")
    if not isinstance(uri_val, str) or not uri_val.strip():
        uri_val = row.get("http.request.full_uri")

    uri = uri_val.strip() if isinstance(uri_val, str) else ""

    if uri.startswith("http://") or uri.startswith("https://"):
        return uri or "/"

    if host:
        if not uri:
            return f"http://{host}/"
        if uri.startswith("/"):
            return f"http://{host}{uri}"
        return f"http://{host}/{uri}"

    # fallback – just URI or "/" if empty
    return uri or "/"

def featurize_row(row) -> dict:
    """
    Extracts numeric features from a data row.
    
    Args:
        row: The input data row.
    Returns:
        dict: Extracted features.
    """
    url = build_url_from_row(row)

    # URL parts
    parsed = urlparse(url)
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
    payload_size = parse_payload_size(row.get("http.content_length_header"))

    # suspicious chars
    combined = " ".join([
        str(url) or "",
        str(row.get("http.cookie") or ""),
        str(row.get("ftp.request.arg") or ""),
        str(row.get("http.request.uri") or ""),
    ]).strip()
    suspicious_ratio = suspicious_char_ratio_from_string(combined)

    # method
    method_code = http_method_code(row.get("http.request.method"))

    # is_json
    is_json = is_json_payload(row.get("http.content_type"))

    # label – from alert (benign / suspicious)
    lbl = label_from_alert(row.get("alert"))

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
        "label": lbl,
    }

def main():
    """
    Main function to build ML numeric features from the attack-lab dataset.
    Returns:
        None
    """
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    csv_path = get_attacklab_csv_path()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Remove existing output file
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

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
    cols = feature_cols + ["label"]

    chunk_size = int(os.getenv("ATTACKLAB_CHUNKSIZE", "50000"))
    header_written = False
    total_rows = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=True):
        rows = []
        for _, r in chunk.iterrows():
            rows.append(featurize_row(r))

        out_df = pd.DataFrame(rows)
        # Reorder columns
        out_df = out_df[cols]

        mode = "w" if not header_written else "a"
        out_df.to_csv(OUT_PATH, mode=mode, index=False, header=not header_written)
        header_written = True
        total_rows += out_df.shape[0]

        log.info("[attacklab→ml_numeric] processed chunk, total_rows=%s", total_rows)

    log.info("[attacklab→ml_numeric] saved %s rows -> %s", total_rows, OUT_PATH)
    log.info("[attacklab→ml_numeric] columns: %s", cols)

if __name__ == "__main__":
    main()
