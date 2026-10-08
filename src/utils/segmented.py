"""Utilities for bounded subprocess training segments.

Segmenting is an engineering reliability mechanism for heavyweight simulators:
weights/optimizer state continue from checkpoints while the physical simulator
starts a fresh episode in a fresh process. It is not bitwise trajectory resume.
"""
from __future__ import annotations
from pathlib import Path
import json


def segment_targets(start: int, total: int, segment_steps: int) -> list[int]:
    if start < 0 or total < start or segment_steps <= 0:
        raise ValueError("Require 0 <= start <= total and positive segment_steps")
    if start == total:
        return []
    values = []
    cursor = start
    while cursor < total:
        cursor = min(total, cursor + segment_steps)
        values.append(cursor)
    return values


def completed_training_artifact(directory: str | Path, expected_steps: int) -> dict | None:
    directory = Path(directory)
    result_path, model_path = directory / "result.json", directory / "model.pt"
    if not result_path.is_file() or not model_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if result.get("status") != "training_completed_not_a_driving_safety_claim":
        return None
    if int(result.get("world_steps_total", -1)) != int(expected_steps):
        return None
    if model_path.stat().st_size <= 0:
        return None
    return result
