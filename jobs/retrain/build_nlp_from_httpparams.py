import os
import argparse
import pandas as pd
import logging
import csv
import kagglehub

from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

# Default output path
DEFAULT_OUT = os.getenv("NLP_TRAIN_FILE", "./data/nlp_train.csv")

# Kaggle dataset pull setting
USE_KAGGLE_HTTPPARAMS = os.getenv("USE_KAGGLE_HTTPPARAMS", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# Kaggle dataset handle and file path
HTTPPARAMS_KAGGLE_HANDLE = os.getenv(
    "HTTPPARAMS_KAGGLE_HANDLE",
    "evg3n1j/httpparamsdataset",
)

# Kaggle file path within the dataset
HTTPPARAMS_KAGGLE_FILE = os.getenv(
    "HTTPPARAMS_KAGGLE_FILE",
    "payload_full.csv",
)

# Local CSV path if not using Kaggle
LOCAL_HTTPPARAMS_PATH = os.getenv(
    "HTTPPARAMS_LOCAL_PATH",
    "./data/HttpParamsDataset_payload_train.csv",
)

# Ability to override column names if the dataset does not have payload / label / attack_type
HTTPPARAMS_TEXT_COL = os.getenv("HTTPPARAMS_TEXT_COL", "payload")
HTTPPARAMS_LABEL_COL = os.getenv("HTTPPARAMS_LABEL_COL", "label")
HTTPPARAMS_ATTACK_COL = os.getenv("HTTPPARAMS_ATTACK_COL", "attack_type")

# Deduplication by text+label+attack_type (enabled by default)
NLP_DEDUP = os.getenv("NLP_DEDUP", "1").lower() in ("1", "true", "yes", "on")


def _find_col(df: pd.DataFrame, primary: str, fallbacks: List[str]) -> Optional[str]:
    """
    Finds a column in the DataFrame by checking primary and fallback names.
    
    Args:
        df (pd.DataFrame): The DataFrame to search.
        primary (str): The primary column name to look for.
        fallbacks (List[str]): A list of fallback column names to check.
    Returns:
        Optional[str]: The found column name, or None if not found.
    """
    candidates = [primary] + fallbacks
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_httpparams_df() -> pd.DataFrame:
    """
    Loads the HttpParamsDataset from Kaggle or local disk.
    
    Returns:
        pd.DataFrame: The loaded HttpParamsDataset.
    """
    if USE_KAGGLE_HTTPPARAMS:
        if kagglehub is None:
            raise RuntimeError(
                "USE_KAGGLE_HTTPPARAMS=1, але kagglehub не встановлено. "
                "Додай kagglehub в requirements або вистав USE_KAGGLE_HTTPPARAMS=0."
            )
        logger.info(
            f"[httpparams] downloading Kaggle dataset: "
            f"{HTTPPARAMS_KAGGLE_HANDLE} (file={HTTPPARAMS_KAGGLE_FILE})"
        )
        root = kagglehub.dataset_download(HTTPPARAMS_KAGGLE_HANDLE)
        csv_path = Path(root) / HTTPPARAMS_KAGGLE_FILE
    else:
        csv_path = Path(LOCAL_HTTPPARAMS_PATH)

    if not csv_path.exists():
        raise FileNotFoundError(f"[httpparams] CSV not found at: {csv_path}")

    logger.info(f"[httpparams] loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"[httpparams] loaded shape={df.shape}")
    return df

def normalize_httpparams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводимо raw HttpParamsDataset до схеми:
        text,label,attack_type
    """

    # --- знайдемо колонки з текстом/лейблом/типом ---
    text_col = _find_col(
        df,
        HTTPPARAMS_TEXT_COL,
        ["payload", "value", "param_value", "param", "text"],
    )
    if not text_col:
        raise ValueError(
            f"[httpparams] не знайшов текстову колонку "
            f"(пробував: payload/value/param_value/param/text). "
            f"df.columns={list(df.columns)}"
        )

    label_col = _find_col(
        df,
        HTTPPARAMS_LABEL_COL,
        ["label", "class", "y", "target"],
    )
    if not label_col:
        raise ValueError(
            f"[httpparams] не знайшов колонку з лейблом "
            f"(пробував: label/class/y/target). df.columns={list(df.columns)}"
        )

    attack_col = _find_col(
        df,
        HTTPPARAMS_ATTACK_COL,
        ["attack_type", "attack", "type", "category"],
    )
    # attack_col може бути None — тоді зробимо простий fallback

    logger.info(
        f"[httpparams] using columns: text={text_col}, "
        f"label={label_col}, attack_type={attack_col}"
    )

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)

    # --- label 0/1 ---

    def map_label(v) -> int:
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("norm", "benign", "normal", "0"):
                return 0
            if s in ("anom", "anomaly", "attack", "malicious", "1"):
                return 1
        try:
            iv = int(v)
            return 0 if iv == 0 else 1
        except Exception:
            # дефолтно вважатимемо нормою, щоб не псувати датасет
            return 0

    out["label"] = df[label_col].apply(map_label).astype(int)

    # --- attack_type (нормалізуємо під наші ключі) ---
    if attack_col:
        raw_atk = df[attack_col].astype(str).str.strip().str.lower()

        def map_attack(a: str) -> str:
            if not a:
                return "generic"
            if a in ("norm", "benign", "normal", "0"):
                return "normal"
            if a in ("sqli", "sql_injection", "sql", "sql-i"):
                return "sql_injection"
            if a in ("xss", "cross-site scripting", "cross_site_scripting"):
                return "xss"
            if a in ("cmdi", "cmd_injection", "command_injection"):
                return "cmd_injection"
            if "path" in a and "travers" in a:
                return "path_traversal"
            # решта: залишаємо як є, але без пробілів
            return a.replace(" ", "_")

        out["attack_type"] = raw_atk.apply(map_attack)
    else:
        # якщо немає явної attack_type колонки:
        out["attack_type"] = out["label"].apply(
            lambda y: "normal" if y == 0 else "generic"
        )

    # трішки почистимо
    out["text"] = out["text"].str.strip()
    # викидаємо порожні рядки
    before = out.shape[0]
    out = out[out["text"] != ""]
    after = out.shape[0]
    if before != after:
        logging.log(f"[httpparams] dropped {before - after} empty-text rows")

    # дедуп, якщо увімкнено
    if NLP_DEDUP:
        before = out.shape[0]
        out = out.drop_duplicates(subset=["text", "label", "attack_type"])
        after = out.shape[0]
        if before != after:
            logger.info(
                f"[httpparams] dedup: removed {before - after} duplicates "
                f"(by text+label+attack_type)"
            )

    logger.info(f"[httpparams] normalized shape={out.shape}")
    return out


def merge_with_existing(new_df: pd.DataFrame, out_path: Path, mode: str) -> pd.DataFrame:
    """
    mode = 'override' / 'append'
    - override: повертаємо тільки new_df
    - append:   якщо файл існує — читаємо, конкатимо, вирівнюємо колонки
    """
    if mode == "override":
        logger.info(f"[httpparams] mode=override → перезаписуємо {out_path}")
        return new_df

    # append
    if not out_path.exists():
        logger.info(
            f"[httpparams] mode=append, але {out_path} не існує → "
            f"просто створюємо з нового датасету"
        )
        return new_df

    logger.info(f"[httpparams] mode=append → читаємо існуючий {out_path}")
    try:
        old = pd.read_csv(out_path)
    except Exception as e:
        logger.info(f"[httpparams] failed to read existing {out_path}: {e}")
        return new_df

    # гарантуємо наявність core-полів з обох боків
    for col in ("text", "label", "attack_type"):
        if col not in old.columns:
            old[col] = ""
        if col not in new_df.columns:
            new_df[col] = ""

    cols = ["text", "label", "attack_type"]
    merged = pd.concat([old[cols], new_df[cols]], ignore_index=True)

    if NLP_DEDUP:
        before = merged.shape[0]
        merged = merged.drop_duplicates(subset=["text", "label", "attack_type"])
        after = merged.shape[0]
        if before != after:
            logger.info(f"[httpparams] append dedup: removed {before - after} duplicates")

    logger.info(f"[httpparams] merged shape={merged.shape}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build / update NLP train CSV from HttpParamsDataset."
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"Вихідний CSV (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--mode",
        choices=["override", "append"],
        default="override",
        help="override: перезаписати файл; append: додати до існуючого",
    )

    args = parser.parse_args()
    out_path = Path(args.out)
    mode = args.mode

    logger.info(f"[httpparams] starting… out={out_path}, mode={mode}")

    df_raw = load_httpparams_df()
    df_norm = normalize_httpparams(df_raw)
    df_final = merge_with_existing(df_norm, out_path, mode)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(
        out_path,
        index=False,
        quoting=csv.QUOTE_NONNUMERIC
    )

    logger.info(
        f"[httpparams] written {df_final.shape[0]} rows, "
        f"columns={list(df_final.columns)} -> {out_path}"
    )


if __name__ == "__main__":
    main()