from __future__ import annotations
import html, urllib.parse, unicodedata, re

_ZW_RE = re.compile(r"[\u200B-\u200D\uFEFF]")
_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def norm_text(s: str) -> str:
    """
    Normalize text by decoding, unescaping, 
    removing control/zero-width chars, 
    and standardizing whitespace.
    
    Args:
        s (str): Input string to normalize.
    Returns:
        str: Normalized string.
    """
    if not s:
        return ""
    try:
        s = urllib.parse.unquote_plus(s)
    except Exception:
        pass
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    s = (
        s.replace("\\u0027", "'")
         .replace("\\u003d", "=")
         .replace("\\'", "'")
         .replace('\\"', '"')
         .replace("`", "'")
    )
    s = _ZW_RE.sub("", s)
    s = _CTRL_RE.sub("", s)
    return " ".join(s.split()).lower()