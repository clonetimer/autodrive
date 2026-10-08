"""Machine-readable evidence, not hand-written experimental numbers."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from typing import Any
import numpy as np


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def write_json(path: str | Path, data: Any) -> None:
    save_json(data, path)


def append_csv(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def runtime_manifest() -> dict:
    versions = {}
    for package in ['numpy', 'torch', 'gymnasium', 'stable-baselines3', 'panda3d', 'metadrive-simulator', 'PyYAML', 'opencv-python', 'pytest']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
            'platform': platform.platform(), 'versions': versions, 'device': 'cpu'}
