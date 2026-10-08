"""Inspect supplied ScenarioNet/MetaDrive numerical pickles without a simulator.

A strict reduction whitelist rejects executable foreign globals. This reader is
for known numerical Scenario Description files, not a general untrusted-pickle
sandbox. Files are size bounded and arrays with object dtype are rejected.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import io
from pathlib import Path
import pickle
import numpy as np
from src.utils.logger import sha256


class NumericalScenarioUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        permitted = {
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
            ('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('numpy._core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('numpy.core.multiarray', 'scalar'): np._core.multiarray.scalar,
            ('numpy._core.multiarray', 'scalar'): np._core.multiarray.scalar,
            ('numpy.core.numeric', '_frombuffer'): np._core.numeric._frombuffer,
            ('numpy._core.numeric', '_frombuffer'): np._core.numeric._frombuffer,
            ('collections', 'defaultdict'): defaultdict,
            ('builtins', 'int'): int,
            ('builtins', 'set'): set,
        }
        if (module, name) not in permitted:
            raise pickle.UnpicklingError(f'Forbidden pickle global: {module}.{name}')
        return permitted[(module, name)]

    def persistent_load(self, pid):
        raise pickle.UnpicklingError('Persistent pickle references are not allowed')


def numerical_load(path: str | Path, max_bytes=64 * 1024 ** 2):
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError('Scenario exceeds configured byte limit')
    with path.open('rb') as stream:
        result = NumericalScenarioUnpickler(stream).load()
        if stream.read(1):
            raise ValueError('Trailing bytes after scenario pickle')
    queue, seen, nodes = [result], set(), 0
    while queue:
        value = queue.pop()
        nodes += 1
        if nodes > 2_000_000:
            raise ValueError('Scenario object graph too large')
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError('Object-dtype arrays are forbidden')
        elif isinstance(value, dict):
            queue.extend(value.keys()); queue.extend(value.values())
        elif isinstance(value, (list, tuple, set)):
            queue.extend(value)
        elif not isinstance(value, (str, bytes, bool, int, float, complex, np.generic, type(None))):
            raise ValueError(f'Unsupported data type: {type(value).__name__}')
    return result


def describe_scenario(data: dict) -> dict:
    if not isinstance(data, dict) or not {'id', 'length', 'tracks', 'map_features', 'metadata'}.issubset(data):
        raise ValueError('Not a Scenario Description dictionary')
    length = int(data['length'])
    metadata = data['metadata']
    ts = np.asarray(metadata.get('ts'), dtype=float)
    if ts.shape != (length,) or length < 2 or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise ValueError('Invalid scenario timestamps')
    sdc_id = str(metadata.get('sdc_id'))
    tracks = data['tracks']
    if sdc_id not in tracks:
        raise ValueError('Recorded ego ID missing from tracks')
    types, valid_samples, segments = Counter(), 0, 0
    for key, track in tracks.items():
        state = track['state']
        valid = np.asarray(state['valid']).reshape(-1)
        position = np.asarray(state['position'])
        velocity = np.asarray(state['velocity'])
        heading = np.asarray(state['heading']).reshape(-1)
        if (valid.shape != (length,) or position.shape[0] != length or position.ndim != 2
                or position.shape[1] < 2 or velocity.shape != (length, 2) or heading.shape != (length,)):
            raise ValueError(f'Track state shape mismatch: {key}')
        if not np.isin(valid, [0, 1]).all():
            raise ValueError(f'Invalid track validity mask: {key}')
        mask = valid.astype(bool)
        for arr in (position, velocity, heading):
            if not np.isfinite(arr[mask]).all():
                raise ValueError(f'Nonfinite valid track state: {key}')
        valid_samples += int(mask.sum())
        segments += int(np.count_nonzero(np.diff(np.r_[False, mask].astype(int)) == 1))
        types[str(track['type'])] += 1
    map_types = Counter(str(feature['type']) for feature in data['map_features'].values())
    for key, feature in data['map_features'].items():
        for name in ('polyline', 'polygon'):
            if name in feature:
                xy = np.asarray(feature[name])
                if xy.ndim != 2 or xy.shape[1] < 2 or not np.isfinite(xy).all():
                    raise ValueError(f'Invalid map geometry {key}.{name}')
    ego = tracks[sdc_id]['state']
    mask = np.asarray(ego['valid']).reshape(-1).astype(bool)
    speed = np.linalg.norm(np.asarray(ego['velocity'])[mask], axis=1)
    return {'scenario_id': str(data['id']), 'dataset': str(metadata.get('dataset', 'unspecified')),
            'coordinate_system': str(metadata.get('coordinate', metadata.get('coordinate_system', 'unspecified'))),
            'length': length, 'duration_seconds': float(ts[-1] - ts[0]),
            'dt_min': float(np.diff(ts).min()), 'dt_max': float(np.diff(ts).max()),
            'recorded_ego_id': sdc_id, 'ego_valid_frames': int(mask.sum()),
            'ego_mean_speed_mps': float(speed.mean()) if len(speed) else 0,
            'tracks': len(tracks), 'track_types': dict(types), 'valid_track_samples': valid_samples,
            'contiguous_track_segments': segments, 'map_features': len(data['map_features']),
            'map_types': dict(map_types), 'dynamic_map_states': len(data.get('dynamic_map_states', {})),
            'status': 'numerical_schema_validated_not_simulated'}


def inspect_directory(directory: str | Path) -> dict:
    directory = Path(directory)
    rows = []
    for path in sorted(directory.glob('sd_*.pkl')):
        row = describe_scenario(numerical_load(path))
        rows.append(dict(row, file=path.name, sha256=sha256(path), bytes=path.stat().st_size))
    if not rows:
        raise ValueError('No sd_*.pkl scenarios found')
    ids = [r['scenario_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate scenario IDs')
    summary_path = directory / 'dataset_summary.pkl'
    summary_ids = None
    if summary_path.is_file():
        summary = numerical_load(summary_path)
        if not isinstance(summary, dict):
            raise ValueError('Dataset summary must be a mapping')
        summary_ids = list(summary)
        if set(summary_ids) != {r['file'] for r in rows}:
            raise ValueError('Dataset summary and scenario files disagree')
    return {'status': 'supplied_scenarios_audited', 'directory': str(directory.resolve()),
            'scenarios': rows, 'scenario_count': len(rows), 'summary_index_verified': summary_ids is not None,
            'total_track_samples': sum(r['valid_track_samples'] for r in rows),
            'real_metadrive_steps': 0, 'closed_loop_evaluated': False,
            'note': 'Supplied MetaDrive assets include converted Waymo examples. This is data validation, '
                    'not a new Waymo download, model training, or real MetaDrive execution.'}
