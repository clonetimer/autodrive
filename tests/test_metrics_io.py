from io import BytesIO
from pathlib import Path
import hashlib
import json
import struct
import zipfile
import numpy as np
import pytest
from src.datasets.download import download_file, safe_extract_zip
from src.datasets.scenario import csv_to_scenario, validate_scenario
from src.datasets.replay import ReplayWorld
from src.evaluation.metrics import EpisodeMetrics, summarize
from src.evaluation.evaluator import evaluate
from src.baselines.pid_controller import PIDController
from src.utils.config import EnvConfig, TrainConfig, load_config
from src.utils.events import ScalarWriter, crc32c
from src.utils.logger import save_json


def test_metrics_time_weighting_and_success():
    metric = EpisodeMetrics(3, 'agent0')
    info = dict(dt=0.2, speed=10, lane_offset=-0.3, jerk=-2, lane_width=3.6,
                collision=False, out_of_road=False, success=False, route_completion=0.7)
    metric.update(1, info, False, False, 2)
    metric.update(2, dict(info, dt=0.1, speed=4, success=True, route_completion=1), True, False, 3)
    row = metric.result()
    assert row['avg_speed_mps'] == pytest.approx(8)
    assert row['jerk_mean_abs_mps3'] == 2
    assert row['jerk_rms_mps3'] == 2
    assert row['success'] and row['return'] == 3
    assert summarize([row])['success_rate'] == 1


def test_empty_metrics_rejected():
    with pytest.raises(ValueError):
        summarize([])
    with pytest.raises(ValueError):
        EpisodeMetrics(0, 'agent0').result()


def test_exported_evaluation_reports(tmp_path):
    cfg = EnvConfig(traffic_vehicles=0, max_steps=120, road_length=60)
    report = evaluate(lambda name, seed: PIDController(), cfg, [2000, 2001], tmp_path)
    marker = __import__('json').loads((tmp_path / 'evaluation.complete.json').read_text())
    assert marker['status'] == 'complete' and marker['episodes'] == 2 and marker['world_steps'] == report['world_steps']
    assert report['metrics']['success_rate'] == 1
    for filename in ['evaluation.json', 'evaluation.md', 'evaluation.html', 'episodes.csv']:
        assert (tmp_path / filename).stat().st_size > 0
    with pytest.raises(ValueError):
        evaluate(lambda name, seed: PIDController(), cfg, [2000, 2000])


def test_crc32c_and_event_records(tmp_path):
    assert crc32c(b'123456789') == 0xE3069283
    writer = ScalarWriter(tmp_path)
    writer.add_scalar('train/reward', 1.25, 42)
    writer.close()
    data = writer.path.read_bytes()
    records = []
    offset = 0
    while offset < len(data):
        length = struct.unpack('<Q', data[offset:offset+8])[0]
        payload = data[offset+12:offset+12+length]
        assert len(payload) == length
        from src.utils.events import _masked_crc
        assert data[offset+8:offset+12] == _masked_crc(data[offset:offset+8])
        assert data[offset+12+length:offset+16+length] == _masked_crc(payload)
        records.append(payload)
        offset += 16 + length
    assert len(records) == 2
    assert b'brain.Event:2' in records[0] and b'train/reward' in records[1]


def test_atomic_json_refuses_nan(tmp_path):
    with pytest.raises(ValueError):
        save_json({'bad': float('nan')}, tmp_path / 'bad.json')
    assert not (tmp_path / 'bad.json').exists()


@pytest.mark.parametrize('values', [{'backend': 'pretend'}, {'dt': 0}, {'num_agents': 0}, {'observation': 'unknown'}, {'lanes': 0}])
def test_invalid_env_config(values):
    with pytest.raises(ValueError):
        EnvConfig(**values)


def test_yaml_strict_unknown_fields(tmp_path):
    (tmp_path / 'c.yaml').write_text('nonsense: 1')
    with pytest.raises(TypeError):
        load_config(tmp_path / 'c.yaml')


def test_safe_zip_rejects_traversal(tmp_path):
    archive = tmp_path / 'evil.zip'
    with zipfile.ZipFile(archive, 'w') as zip:
        zip.writestr('../outside.py', 'bad')
    with pytest.raises(ValueError):
        safe_extract_zip(archive, tmp_path / 'out')
    assert not (tmp_path / 'outside.py').exists()


def test_safe_zip_extract_and_size_limit(tmp_path):
    archive = tmp_path / 'valid.zip'
    with zipfile.ZipFile(archive, 'w') as zip:
        zip.writestr('data/file.txt', 'test')
    safe_extract_zip(archive, tmp_path / 'out')
    assert (tmp_path / 'out/data/file.txt').read_text() == 'test'
    with pytest.raises(ValueError):
        safe_extract_zip(archive, tmp_path / 'limited', max_bytes=1)


def test_download_rejects_license_and_http(tmp_path):
    with pytest.raises(PermissionError):
        download_file('https://example.com/data', tmp_path / 'data', '0' * 64)
    with pytest.raises(ValueError):
        download_file('http://example.com/data', tmp_path / 'data', '0' * 64, True)


def test_checksum_download_with_mock_transport(tmp_path, monkeypatch):
    payload = b'trajectory-data'
    class Response(BytesIO):
        headers = {'Content-Length': str(len(payload))}
        def geturl(self):
            return 'https://example.com/data'
    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: Response(payload))
    digest = hashlib.sha256(payload).hexdigest()
    result = download_file('https://example.com/data', tmp_path / 'data', digest, True)
    assert result.read_bytes() == payload
    with pytest.raises(ValueError):
        download_file('https://example.com/data', tmp_path / 'bad', '0' * 64, True)
    assert not (tmp_path / 'bad.part').exists()


def test_csv_conversion_and_nonreactive_replay(tmp_path):
    csv = tmp_path / 'tracks.csv'
    csv.write_text('time_s,track_id,x_m,y_m,yaw_rad,speed_mps\n0,lead,37,0,0,5\n10,lead,87,0,0,5\n')
    output = tmp_path / 'scenario.json'
    csv_to_scenario(csv, output, 'synthetic unit test')
    env = ReplayWorld(EnvConfig(backend='replay', dataset=str(output), traffic_vehicles=0))
    env.reset(seed=0)
    env.step({'agent0': [0, 1]})
    assert env.vehicles['lead'].x == pytest.approx(37.5)
    assert env.vehicles['agent0'].x > 5


def test_scenario_rejects_nonmonotonic_tracks():
    data = {'schema_version': 1, 'provenance': 'synthetic',
            'tracks': [{'id': 'lead', 'samples': [[1, 0, 0, 0, 1], [1, 1, 0, 0, 1]]}]}
    with pytest.raises(ValueError):
        validate_scenario(data)


def test_wilson_intervals_are_not_degenerate_at_boundaries():
    from src.evaluation.metrics import wilson_interval
    low = wilson_interval(0, 20)
    high = wilson_interval(20, 20)
    assert low[0] == pytest.approx(0) and 0.15 < low[1] < 0.17
    assert 0.83 < high[0] < 0.85 and high[1] == pytest.approx(1)
    assert wilson_interval(10, 20)[0] < 0.5 < wilson_interval(10, 20)[1]
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_composable_config_and_cycle_detection(tmp_path):
    (tmp_path / 'env.yaml').write_text('env:\n  traffic_vehicles: 2\n  observation: hybrid\n')
    (tmp_path / 'main.yaml').write_text('include: [env.yaml]\nseed: 5\nenv:\n  traffic_vehicles: 4\n')
    result = load_config(tmp_path / 'main.yaml')
    assert result.env.traffic_vehicles == 4 and result.env.observation == 'hybrid'
    (tmp_path / 'env.yaml').write_text('include: [main.yaml]')
    with pytest.raises(ValueError, match='Cyclic'):
        load_config(tmp_path / 'main.yaml')


def test_replay_track_delayed_spawn_and_lifetime(tmp_path):
    data = {'schema_version': 1, 'provenance': 'synthetic',
            'tracks': [{'id': 'late', 'samples': [[0.15, 70, 0, 0, 5], [0.3, 70.75, 0, 0, 5]]}]}
    path = tmp_path / 'tracks.json'
    save_json(data, path)
    world = ReplayWorld(EnvConfig(backend='replay', dataset=str(path), traffic_vehicles=0))
    world.reset(seed=1)
    assert not world.vehicles['late'].active
    world.step({'agent0': [0, 0]})
    assert not world.vehicles['late'].active
    world.step({'agent0': [0, 0]})
    assert world.vehicles['late'].active
    assert world.vehicles['late'].x == pytest.approx(70.25)
    world.step({'agent0': [0, 0]})
    world.step({'agent0': [0, 0]})
    assert not world.vehicles['late'].active
