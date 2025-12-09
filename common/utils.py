import hashlib, json
import os
import re
import logging
import math
import ipaddress
from urllib.parse import urlparse, parse_qs
from typing import Any, Mapping

log = logging.getLogger(__name__)

# Environment variable regex patterns
ENV_INT_RE   = re.compile(r"^\s*['\"]?(\d+)['\"]?\s*$")
ENV_FLOAT_RE = re.compile(r"^\s*['\"]?([0-9]+(?:\.[0-9]+)?)['\"]?\s*$")
ENV_BOOL_RE  = re.compile(r"^\s*['\"]?(true|false|1|0|yes|no|on|off)['\"]?\s*$", re.I)

# PII Redaction regex patterns
PII_IP_RE = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')  # IPv4
PII_EMAIL_RE = re.compile(r'([a-zA-Z0-9_.+-]+)@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
PII_PHONE_RE = re.compile(r'\+?\d[\d\s\-()]{7,}\d')

# PII Masking regex patterns
EMAIL_RE  = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
IPV4_RE   = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
PHONE_RE  = re.compile(
    r'\b(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3,4}[\s\-]?\d{3,4}\b'
)
ISO_TS_RE  = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')
SUS_CHARS = set('\'"%;<>(){}[]\\|&$`')


def make_hash_id(ts: str, endpoint: str, user_id: str, payload: dict) -> str:
    """
    Creates a SHA256 hash ID from timestamp, endpoint, user ID, and payload.
    
    Args:
        ts: Timestamp string.
        endpoint: API endpoint string.
        user_id: User identifier string.
        payload: Request payload dictionary.
        
    Returns:
        str: SHA256 hash ID as a hexadecimal string.
    """
    h = hashlib.sha256()
    ts_s  = str(ts or "")
    ep_s  = str(endpoint or "")
    uid_s = str(user_id or "")
    try:
        pay_s = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        pay_s = str(payload)
    blob = (ts_s + ep_s + uid_s + pay_s).encode("utf-8", "ignore")
    h.update(blob)
    return h.hexdigest()

def sigmoid(x: float) -> float:
    """
    Computes the sigmoid function.
    
    Args:
        x: Input value.
    Returns:
        The sigmoid of x.
    """
    return 1.0/(1.0+math.exp(-x))

def strip_wrapping_quotes(s: str) -> str:
    """
    Strips wrapping single or double quotes from a string.
    
    Args:
        s: The input string.
    Returns:
        The string without wrapping quotes.
    """
    s = s.strip()
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ("'", '"')):
        return s[1:-1].strip()
    return s

def getenv_str(name: str, default: str = "") -> str:
    """
    Gets an environment variable as a string, with optional default.
    
    Args:
        name: The name of the environment variable.
        default: The default value if the environment variable is not set.
    Returns:
        The value of the environment variable or the default.
    """
    val = os.getenv(name)
    if val is None:
        return default
    return strip_wrapping_quotes(val)

def getenv_int(name: str, default: int) -> int:
    """
    Gets an environment variable as an integer, with optional default.
    
    Args:
        name: The name of the environment variable.
        default: The default value if the environment variable is not set or invalid.
    Returns:
        The integer value of the environment variable or the default.
    """
    raw = os.getenv(name)
    if not raw:
        return default
    m = ENV_INT_RE.match(raw)
    if m:
        return int(m.group(1))
    log.warning("Invalid int for %s='%s', fallback to %s", name, raw, default)
    return default

def getenv_float(name: str, default: float) -> float:
    """
    Gets an environment variable as a float, with optional default.
    
    Args:
        name: The name of the environment variable.
        default: The default value if the environment variable is not set or invalid.
    Returns:
        The float value of the environment variable or the default.
    """
    raw = os.getenv(name)
    if not raw:
        return default
    m = ENV_FLOAT_RE.match(raw)
    if m:
        return float(m.group(1))
    log.warning("Invalid float for %s='%s', fallback to %s", name, raw, default)
    return default

def getenv_bool(name: str, default: bool) -> bool:
    """
    Gets an environment variable as a boolean, with optional default.
    
    Args:
        name: The name of the environment variable.
        default: The default value if the environment variable is not set or invalid.
    Returns:
        The boolean value of the environment variable or the default.
    """
    raw = os.getenv(name)
    if not raw:
        return default
    m = ENV_BOOL_RE.match(raw)
    if not m:
        log.warning("Invalid bool for %s='%s', fallback to %s", name, raw, default)
        return default
    token = m.group(1).lower()
    return token in ("true", "1", "yes", "on")

def require_env(name: str) -> str:
    """
    Checks for required environments presense.
    
    Args:
        name: The name of the environment variable.
    Returns:
        The value of the environment variable.
    """
    val = os.getenv(name)
    if not val:
        raise ValueError(f"Required environment variable {name} is not set")
    return val
  
def redact_scalar(val: Any) -> Any:
    """
    Redacts PII from scalar values (strings).
    
    Args:
        val: The scalar value to redact.
    Returns:
        The redacted scalar value.
    """
    if not isinstance(val, str):
        return val
    s = val
    s = PII_IP_RE.sub('[IP]', s)
    s = PII_EMAIL_RE.sub(r'\1@[DOMAIN]', s)
    s = PII_PHONE_RE.sub('[PHONE]', s)
    return s

def redact_pii(obj: Any) -> Any:
    """
    Deep-redact only for logging (keeps structure).
    
    Args:
        obj: The object to redact.
    Returns:
        The redacted object.
    """
    if isinstance(obj, Mapping):
        return {k: redact_pii(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_pii(v) for v in obj]
    return redact_scalar(obj)

def safe_log_json(obj: Any) -> str:
    """
    Dump JSON for logs (with PII redaction).
    
    Args:
        obj: The object to serialize.
    Returns:
        The JSON string with PII redacted.
    """
    try:
        return json.dumps(redact_pii(obj), ensure_ascii=False)
    except Exception:
        return "<unserializable>"
      
def safe_len_bytes(obj: Any) -> int:
    """
    Calculate the length in bytes of the JSON representation of an object.
    
    Args:
        obj: The object to measure.
    Returns:
        The length in bytes of the JSON string or string representation.
    """
    try:
        return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    except Exception:
        try:
            return len(str(obj).encode("utf-8"))
        except Exception:
            return 0

def shannon_entropy(s: str) -> float:
    """
      Calculate the Shannon entropy of a string.
      
      Args:
          s: The input string.
      Returns:
          The Shannon entropy value.
    """
    if not s:
        return 0.0
    from collections import Counter
    cnt = Counter(s)
    n = len(s)
    return -sum((c/n) * math.log2(c/n) for c in cnt.values())

def clip_text(s: str, n: int) -> str:
    """
    Clip a string to a maximum length, appending a truncation notice if necessary.
    
    Args:
        s: The input string.
        n: The maximum length.
    Returns:
        The clipped string.
    """
    if s is None:
        return ""
    if len(s) <= n:
        return s
    return s[:n] + "...<truncated>"

def is_private_ip(ip: str) -> bool:
    """
    Check if an IP address is private.
    
    Args:
        ip: The IP address as a string.
    Returns:
        True if the IP address is private, False otherwise.
    """
    try:
        return ipaddress.ip_address(ip).is_private
    except Exception:
        return False

def suspicious_ratio(s: str) -> float:
    """
    Calculate the ratio of suspicious characters in a string.
    
    Args:
        s: The input string.
    Returns:
        The ratio of suspicious characters to the total length of the string.
    """
    if not s:
        return 0.0
    total = len(s)
    bad = sum(1 for ch in s if ch in SUS_CHARS)
    return bad / total if total else 0.0


def mask_pii(s: str) -> str:
    """
    Masks PII in a string by replacing emails, IP addresses, and phone numbers with placeholders.
    
    Args:
        s: The input string.
    Returns:
        The string with PII masked.
    """
    if not s:
        return s
    s = EMAIL_RE.sub("[EMAIL]", s)
    s = IPV4_RE.sub("[IP]", s)
    s = PHONE_RE.sub("[PHONE]", s)
    return s
  
def mask_pii_in_obj(obj):
    """
    Masks PII in an object by recursively replacing emails, IP addresses, and phone numbers with placeholders.
    
    Args:
        obj: The input object.
    Returns:
        The object with PII masked.
    """
    if obj is None:
        return obj
    if isinstance(obj, str):
        return mask_pii(obj)
    if isinstance(obj, dict):
        return {k: mask_pii_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [mask_pii_in_obj(v) for v in obj]
    return obj
  
def ensure_compact_json(obj: Any) -> str:
    """
    Ensures a compact JSON representation of an object.

    Args:
        obj: The input object.
    Returns:
        A compact JSON string representation of the object.
    """
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(obj)

def is_json_like(obj: Any) -> bool:
    """
    Checks if an object is JSON-like.
    
    Args:
        obj: The input object.
    Returns:
        True if the object is JSON-like, False otherwise.
    """
    if isinstance(obj, (dict, list)):
        return True
    if isinstance(obj, str):
        t = obj.strip()
        if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
            try:
                json.loads(t)
                return True
            except Exception:
                return False
    return False

def path_depth(endpoint: str) -> int:
    """
    Calculate the depth of a URL path.
    
    Args:
        endpoint: The URL path string.
    Returns:
        The depth of the path.
    """
    return len([p for p in endpoint.split("/") if p])

def query_param_count(url_or_endpoint: str) -> int:
    """
    Calculate the number of query parameters in a URL or endpoint.
    
    Args:
        url_or_endpoint: The full URL or URL path string.
    Returns:
        The count of query parameters.
    """
    if "?" in url_or_endpoint:
        q = url_or_endpoint.split("?", 1)[1]
        return sum(len(v) for v in parse_qs(q, keep_blank_values=True).values())
    try:
        pr = urlparse(url_or_endpoint)
        return sum(len(v) for v in parse_qs(pr.query, keep_blank_values=True).values())
    except Exception:
        return 0
      

def sanitize_value(v: str) -> str:
    """
    Sanitize a string value by masking PII unless it matches an ISO timestamp.
    
    Args:
        v: The input string.
    Returns:
        The sanitized string.
    """
    if not isinstance(v, str):
        return v
    if ISO_TS_RE.fullmatch(v):
        return v
    s = v
    s = EMAIL_RE.sub('[EMAIL]', s)
    s = IPV4_RE.sub('[IP]', s)
    s = PHONE_RE.sub('[PHONE]', s)
    return s

def mask_pii_in_obj(obj):
    """
    Recursively mask PII in an object.
    
    Args:
        obj: The input object.
    Returns:
        The object with PII masked.
    """
    if isinstance(obj, dict):
        return {k: mask_pii_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [mask_pii_in_obj(x) for x in obj]
    if isinstance(obj, str):
        return sanitize_value(obj)
    return obj