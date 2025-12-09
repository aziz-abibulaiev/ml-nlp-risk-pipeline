import base64
import json
import os
from pathlib import Path
from typing import Dict


def load_training_set():
    """
    Loads training set feature vectors and labels.
    
    Returns:
        List[List[float]]: Feature vectors.
    """
    return [[0.1,  12, 3.3],[0.11, 15, 3.0],[0.09, 9, 3.6],[0.5,  80, 5.1]]

def upload_model(path: str):
    """
    Uploads the model path to configmap, secret, or env file based on environment variables.
    
    Args:
        path (str): Path to the model.
    """
    config_key = os.getenv("ML_INFER_MODEL_PATH_KEY", "MODEL_PATH_IF")
    model_location = os.getenv("ML_INFER_MODEL_URI", path).strip() or path
    updates = {config_key: model_location}

    updated = False
    cfg_map_file = os.getenv("ML_INFER_CONFIGMAP_FILE", "").strip()
    if cfg_map_file:
        _update_configmap(Path(cfg_map_file), updates)
        updated = True

    secret_file = os.getenv("ML_INFER_SECRET_FILE", "").strip()
    if secret_file:
        _update_secret(Path(secret_file), updates)
        updated = True

    env_file = os.getenv("ML_INFER_ENV_FILE", "").strip()
    if env_file:
        _update_env_file(Path(env_file), updates)
        updated = True

    if not updated:
        os.environ.update(updates)


def _update_configmap(path: Path, updates: Dict[str, str]):
    """
    Updates a configmap JSON file with the given updates.
    
    Args:
        path (Path): Path to the configmap file.
        updates (Dict[str, str]): Key-value pairs to update.
    """
    _update_json_store(path, updates, nested_key="data")


def _update_secret(path: Path, updates: Dict[str, str]):
    """
    Updates a secret JSON file with base64-encoded values.
    
    Args:
        path (Path): Path to the secret file.
        updates (Dict[str, str]): Key-value pairs to update.
    """
    encoded_updates = {
        key: base64.b64encode(value.encode("utf-8")).decode("ascii")
        for key, value in updates.items()
    }
    _update_json_store(path, encoded_updates, nested_key="data")


def _update_json_store(path: Path, updates: Dict[str, str], nested_key: str | None):
    """
    Updates a JSON file with the given updates, optionally under a nested key.
    
    Args:
        path (Path): Path to the JSON file.
        updates (Dict[str, str]): Key-value pairs to update.
        nested_key (str | None): Optional nested key under which to place updates.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Dict[str, str]] | Dict[str, str] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text() or "{}")
        except json.JSONDecodeError:
            payload = {}

    if nested_key:
        container = payload.get(nested_key)
        if not isinstance(container, dict):
            container = {}
        container.update(updates)
        payload[nested_key] = container
        path.write_text(json.dumps(payload, indent=2))
    else:
        base_doc = payload if isinstance(payload, dict) else {}
        base_doc.update(updates)
        path.write_text(json.dumps(base_doc, indent=2))


def _update_env_file(path: Path, updates: Dict[str, str]):
    """
    Updates an env file with the given key-value pairs.
    
    Args:
        path (Path): Path to the env file.
        updates (Dict[str, str]): Key-value pairs to update.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    kv: Dict[str, str] = {}
    if path.exists():
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            kv[key.strip()] = value.strip()
    kv.update(updates)
    lines = [f"{k}={v}" for k, v in sorted(kv.items())]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
