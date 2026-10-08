"""Portable, non-executable trajectory schema and CSV conversion.

This is NOT a claim to implement full raw Waymo/nuScenes conversion. Already
converted trusted ScenarioNet datasets can be used with real ScenarioEnv.
"""
from __future__ import annotations
from pathlib import Path
import csv
import json
import numpy as np
from src.utils.logger import save_json


def validate_scenario(data: dict) -> dict:
    if data.get('schema_version') != 1 or not isinstance(data.get('tracks'), list):
        raise ValueError('Expected portable scenario schema_version=1 and tracks list')
    seen = set()
    for track in data['tracks']:
        name = str(track['id'])
        if name in seen or name.startswith('agent'):
            raise ValueError('Track IDs must be unique and not use the reserved agent prefix')
        seen.add(name)
        samples = np.asarray(track['samples'], dtype=float)
        if samples.ndim != 2 or samples.shape[1] != 5 or samples.shape[0] < 2:
            raise ValueError('Each track needs >=2 rows [time_s, x_m, y_m, yaw_rad, speed_mps]')
        if not np.isfinite(samples).all() or (np.diff(samples[:, 0]) <= 0).any() or (samples[:, 4] < 0).any():
            raise ValueError('Track time must increase; states must be finite and speeds nonnegative')
        if float(track.get('length', 4.5)) <= 0 or float(track.get('width', 1.8)) <= 0:
            raise ValueError('Track dimensions must be positive')
    if 'provenance' not in data:
        raise ValueError('Scenario provenance is mandatory (synthetic / source + license)')
    return data


def load_scenario(path: str | Path) -> dict:
    return validate_scenario(json.loads(Path(path).read_text(encoding='utf-8')))


def csv_to_scenario(source: str | Path, destination: str | Path, provenance: str) -> dict:
    tracks = {}
    with Path(source).open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        expected = {'time_s', 'track_id', 'x_m', 'y_m', 'yaw_rad', 'speed_mps'}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing CSV columns: {expected - set(reader.fieldnames or [])}')
        for row in reader:
            tracks.setdefault(row['track_id'], []).append([float(row[key]) for key in ['time_s', 'x_m', 'y_m', 'yaw_rad', 'speed_mps']])
    data = {'schema_version': 1, 'provenance': provenance,
            'tracks': [{'id': name, 'samples': sorted(rows)} for name, rows in sorted(tracks.items())]}
    validate_scenario(data)
    save_json(data, destination)
    return data
