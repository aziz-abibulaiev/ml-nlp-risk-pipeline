"""Shared realistic User-Agent pool.

Used by the benign driver AND the attack generators so that the User-Agent distribution
overlaps across benign and attack traffic and across tools. This keeps UA from becoming a
label tell — a detector must not be able to separate attack from benign, or one tool from
another, by the UA string alone. sqlmap uses --random-agent (its own real-browser DB,
which overlaps these); ffuf is given a UA drawn from this pool per request where possible.
"""

REALISTIC_UAS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36 Edg/118.0",
    "PostmanRuntime/7.36.0",
    "python-requests/2.32.3",
    "curl/8.4.0",
]
