import copy
import pickle
import numpy as np
import pytest
from src.datasets.scenarionet_inspect import numerical_load, describe_scenario, inspect_directory


def scenario():
    return {'id': 'fixture', 'length': 3, 'metadata': {'ts': np.array([0., .1, .2]), 'sdc_id': 'ego'},
            'tracks': {'ego': {'type': 'VEHICLE', 'state': {
                'position': np.zeros((3, 3)), 'heading': np.zeros(3),
                'velocity': np.ones((3, 2)), 'valid': np.ones(3, dtype=bool)}}},
            'map_features': {'lane': {'type': 'LANE_SURFACE_STREET', 'polyline': np.zeros((3, 2))}}}


def test_numeric_pickle_roundtrip_without_engine_import(tmp_path):
    p = tmp_path / 'data.pkl'; p.write_bytes(pickle.dumps(scenario(), protocol=4))
    result = describe_scenario(numerical_load(p))
    assert result['tracks'] == 1 and result['length'] == 3
    assert result['status'] == 'numerical_schema_validated_not_simulated'


def test_foreign_pickle_reducer_cannot_execute(tmp_path):
    marker = tmp_path / 'executed'
    class Attack:
        def __reduce__(self):
            import os
            return os.system, (f'touch {marker}',)
    path = tmp_path / 'attack.pkl'; path.write_bytes(pickle.dumps(Attack()))
    with pytest.raises(pickle.UnpicklingError, match='Forbidden'):
        numerical_load(path)
    assert not marker.exists()


def test_pickle_object_arrays_rejected(tmp_path):
    path = tmp_path / 'objects.pkl'; path.write_bytes(pickle.dumps(np.array(['x'], dtype=object)))
    with pytest.raises(ValueError, match='Object-dtype'):
        numerical_load(path)


def test_file_size_bound_and_trailing_bytes(tmp_path):
    path = tmp_path / 'data.pkl'; path.write_bytes(pickle.dumps(scenario()))
    with pytest.raises(ValueError, match='byte limit'):
        numerical_load(path, max_bytes=1)
    path.write_bytes(path.read_bytes() + b'extra')
    with pytest.raises(ValueError, match='Trailing'):
        numerical_load(path)


@pytest.mark.parametrize('bad', ['time', 'shape', 'nan', 'ego', 'valid', 'map'])
def test_scenario_schema_rejects_invalid_samples(bad):
    d = scenario()
    if bad == 'time': d['metadata']['ts'][1] = 0
    if bad == 'shape': d['tracks']['ego']['state']['velocity'] = np.zeros((2, 2))
    if bad == 'nan': d['tracks']['ego']['state']['position'][1, 0] = np.nan
    if bad == 'ego': d['metadata']['sdc_id'] = 'missing'
    if bad == 'valid': d['tracks']['ego']['state']['valid'] = [0, 2, 1]
    if bad == 'map': d['map_features']['lane']['polyline'] = [1, 2]
    with pytest.raises(ValueError): describe_scenario(d)


def test_dataset_summary_crosscheck(tmp_path):
    (tmp_path / 'sd_fixture.pkl').write_bytes(pickle.dumps(scenario()))
    (tmp_path / 'dataset_summary.pkl').write_bytes(pickle.dumps({'sd_fixture.pkl': {}}))
    report = inspect_directory(tmp_path)
    assert report['summary_index_verified'] and report['scenario_count'] == 1
    assert report['real_metadrive_steps'] == 0 and not report['closed_loop_evaluated']
    (tmp_path / 'dataset_summary.pkl').write_bytes(pickle.dumps({'missing.pkl': {}}))
    with pytest.raises(ValueError, match='disagree'):
        inspect_directory(tmp_path)
