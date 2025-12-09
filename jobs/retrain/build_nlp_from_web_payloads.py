import argparse
import csv
from pathlib import Path
from typing import Literal

import pandas as pd


def map_web_payload_attack_type(raw_type: str, raw_id: str) -> str:
    """
    Maps raw_type and raw_id to standardized attack_type.
    
    Args:
        raw_type (str): The raw type string.
        raw_id (str): The raw id string.
    Returns:
        str: Standardized attack type or empty string if unknown.
    """
    t = (raw_type or "").strip().lower()
    i = (raw_id or "").strip().lower()

    if i.startswith("sqli-"):
        return "sql_injection"
    if i.startswith("xss-"):
        return "xss"
    if i.startswith("ssrf-"):
        return "ssrf"
    if i.startswith("cmdi-") or i.startswith("cmdinj-"):
        return "cmd_injection"
    if i.startswith("csrf-"):
        return "business_logic"

    if "sqli" in t or "sql" in t:
        return "sql_injection"
    if "xss" in t:
        return "xss"
    if "ssrf" in t:
        return "ssrf"
    if "cmd" in t or "command" in t:
        return "cmd_injection"
    if "csrf" in t:
        return "business_logic"

    return ""


def build_nlp_df_from_web_payloads(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Builds NLP training DataFrame from raw web payloads DataFrame.
    
    Args:
        df_raw (pd.DataFrame): Raw DataFrame with web payloads.
    Returns:
        pd.DataFrame: Processed DataFrame with columns: text, label, attack_type.
    """
    rows = []
    for _, r in df_raw.iterrows():
        payload = str(r.get("payload", "") or "").strip()
        if not payload:
            continue

        ctx = str(r.get("context", "") or "").strip()
        raw_type = str(r.get("type", "") or "").strip()
        raw_id = str(r.get("id", "") or "").strip()
        attack_type = map_web_payload_attack_type(raw_type, raw_id)

        # Combine context and payload for text field
        if ctx:
            text = f"{ctx}: {payload}"
        else:
            text = payload

        rows.append(
            {
                "text": text,
                "label": 1,
                "attack_type": attack_type,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["text", "label", "attack_type"])

    return pd.DataFrame(rows, columns=["text", "label", "attack_type"])


def _load_json(file_path: Path) -> pd.DataFrame:
    """
    Loads JSON or JSONL file and parse into DataFrame.
    
    Args:
        file_path (Path): Path to the JSON/JSONL file.
    Returns:
        pd.DataFrame: Parsed DataFrame from the file.
    """
    
    import json
    import sys

    txt = file_path.read_text(encoding="utf-8")

    txt = txt.replace("\ufeff", "").replace("\xa0", " ")
    stripped = txt.lstrip()

    rows = []

    if stripped.startswith("["):
        in_string = False
        escape = False
        brace = 0
        cur = []
        started = False

        for ch in stripped:
            if not started:
                if ch == "{":
                    started = True
                    brace = 1
                    cur = ["{"]
                continue

            cur.append(ch)

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    brace += 1
                elif ch == "}":
                    brace -= 1
                    if brace == 0:
                        obj_str = "".join(cur)
                        try:
                            obj = json.loads(obj_str)
                            rows.append(obj)
                        except json.JSONDecodeError as e:
                            print(
                                f"[web-payloads] skip bad JSON object: {e}",
                                file=sys.stderr,
                            )
                        started = False
                        cur = []

        if not rows:
            raise SystemExit(
                "[web-payloads] no valid JSON objects parsed from array"
            )
        return pd.DataFrame(rows)

    # if JSONL
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(
                f"[web-payloads] skip bad JSONL line: {e}",
                file=sys.stderr,
            )

    if not rows:
        raise SystemExit(
            "[web-payloads] no valid rows parsed from file (neither JSON array nor JSONL)"
        )

    return pd.DataFrame(rows)


def load_from_kaggle() -> pd.DataFrame:
    """
    Loads Web Application Payloads dataset from Kaggle.
    
    Returns:
        pd.DataFrame: DataFrame containing the dataset.
    """
    try:
        import kagglehub
    except ImportError:
        raise SystemExit(
            "[web-payloads] kagglehub is not installed. "
            "Install via: pip install kagglehub[pandas-datasets]"
        )

    dataset_dir = kagglehub.dataset_download("cyberprince/web-application-payloads-dataset")
    dataset_dir = Path(dataset_dir)
    file_path = dataset_dir / "WEB_APPLICATION_PAYLOADS.jsonl"

    if not file_path.exists():
        raise SystemExit(f"[web-payloads] file not found in dataset: {file_path}")

    return _load_json(file_path)


def write_csv_with_quotes(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save df to CSV with ALL fields in double quotes, like:

    "text","label","attack_type"
    "User viewed product page",0,""
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = ["text", "label", "attack_type"]
    for c in cols:
        if c not in df.columns:
            if c == "attack_type":
                df[c] = ""
            else:
                raise SystemExit(f"[web-payloads] missing required column: {c}")
    df = df[cols]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(cols)
        for _, row in df.iterrows():
            writer.writerow([row["text"], int(row["label"]), row["attack_type"]])


def main() -> None:
    """
    Main function to download dataset and build/extend nlp_train.csv.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download Web Application Payloads dataset from Kaggle and "
            "build/extend ./data/nlp_train.csv (text,label,attack_type)."
        )
    )
    parser.add_argument(
        "--out-csv",
        default="./data/nlp_train.csv",
        help="Output CSV path (default: ./data/nlp_train.csv)",
    )
    parser.add_argument(
        "--mode",
        choices=["append", "override"],
        default="append",
        help="append (default): extend existing CSV if present; "
             "override: write only web payloads",
    )

    args = parser.parse_args()
    mode: Literal["append", "override"] = args.mode  # type: ignore

    out_path = Path(args.out_csv)

    print("[web-payloads] loading dataset from Kaggle ...")
    df_raw = load_from_kaggle()
    print(f"[web-payloads] raw rows from Kaggle: {df_raw.shape[0]}")

    df_web = build_nlp_df_from_web_payloads(df_raw)
    print(f"[web-payloads] built {df_web.shape[0]} rows (label=1) from web payloads")

    if mode == "override":
        final_df = df_web
        print("[web-payloads] mode=override -> ignoring existing nlp_train.csv if present")
    else:
        if out_path.exists():
            base = pd.read_csv(out_path)
            if "attack_type" not in base.columns:
                base["attack_type"] = ""
            base["label"] = base["label"].astype(int)
            final_df = pd.concat([base, df_web], ignore_index=True)
            print(
                f"[web-payloads] mode=append -> base rows={base.shape[0]}, "
                f"final rows={final_df.shape[0]}"
            )
        else:
            final_df = df_web
            print(
                "[web-payloads] mode=append but base csv not found -> using only web payloads"
            )

    write_csv_with_quotes(final_df, out_path)
    print(f"[web-payloads] written -> {out_path}")


if __name__ == "__main__":
    main()