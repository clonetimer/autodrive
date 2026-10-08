from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _json(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cat(m):
    return float(m['collision_rate']) + float(m['out_of_road_rate'])


def test_v31_validation_gate_uses_validation10_only_and_passes_before_release14():
    selection = _json('reports/v3_1/selection_validation10.json')
    assert selection['selection_split'] == 'validation10'
    assert selection['release14_opened'] is False
    assert selection['promotion_gate']['passed'] is True
    assert selection['promoted'] is True
    assert selection['adapter_raw_metrics']['success_rate'] - selection['base_metrics']['success_rate'] >= .10
    assert _cat(selection['base_metrics']) - _cat(selection['adapter_raw_metrics']) >= .10


def test_v31_release14_raw_learned_adapter_improves_directionally_but_is_not_overclaimed():
    report = _json('reports/v3_1/comparison.json')
    assert report['version'] == '3.1.0'
    assert report['release14']['consumed'] is True
    base = report['release14']['raw_base']
    adapter = report['release14']['learned_adapter_raw']
    assert adapter['success_rate'] == pytest.approx(.60)
    assert adapter['collision_rate'] == pytest.approx(1/15)
    assert adapter['out_of_road_rate'] == pytest.approx(5/15)
    assert adapter['success_rate'] > base['success_rate']
    assert _cat(adapter) < _cat(base)
    paired = report['release14']['base_to_adapter_paired_success']
    assert paired == {'improved': 4, 'regressed': 0, 'both_good': 5, 'both_bad': 6,
                      'exact_mcnemar_p_two_sided': 0.125}
    assert 'not a real-road safety claim' in report['claim_boundary']


def test_v31_curve_failure_is_preserved():
    release = _json('reports/v3_1/comparison.json')['release14']
    curve = release['learned_adapter_stages']['curve_tracking']
    assert curve['success_rate'] == 0.0
    assert curve['out_of_road_rate'] == 1.0


def test_v31_adapter_evaluation_has_no_geometric_route_assist_runtime():
    report = _json('reports/v3_1/release14_adapter_raw/suite_report.json')
    m = report['metrics']
    assert m['route_assist_mean_blend'] == 0.0
    assert m['route_assist_intervention_time_fraction'] == 0.0
    assert m['learned_adapter_mean_blend'] > 0.0
    assert m['learned_adapter_mean_steering_correction'] > 0.0
    assert m['learned_adapter_active_time_fraction'] > .9
    for episode in report['episodes']:
        assert episode['route_assist_mean_blend'] == 0.0


def test_v31_adapter_registry_hash_and_base_binding():
    registry = _json('models/route_adapter_registry.json')['metadrive_learned_route_adapter_v31']
    assert registry['version'] == '3.1.0'
    assert registry['runtime_geometric_route_assist'] is False
    assert _sha(ROOT / registry['path']) == registry['sha256']
    assert _sha(ROOT / registry['base_model']) == registry['base_model_sha256']


def test_v31_supervised_system_is_kept_separate_from_learned_adapter_registry():
    systems = _json('models/system_registry.json')
    row = systems['metadrive_learned_route_adapter_supervised_v31']
    assert row['geometric_route_assist_runtime'] is False
    assert 'SafetySupervisor' in row['longitudinal_component']
    assert row['release14_collision_rate'] == 0.0
    adapters = json.dumps(_json('models/route_adapter_registry.json'))
    assert 'SafetySupervisor' not in adapters


def test_release14_freeze_hashes_and_chronology_hold():
    freeze = _json('reports/v3_1/release14_freeze.json')
    assert freeze['release14_opened'] is False
    for relative, expected in freeze['files'].items():
        current = ROOT / relative
        archived = ROOT / 'reports/v3_1/frozen_sources' / relative
        candidate = current if current.is_file() and _sha(current) == expected else archived
        assert candidate.is_file(), relative
        assert _sha(candidate) == expected, relative
    freeze_time = datetime.fromisoformat(freeze['created_utc']).astimezone(timezone.utc).timestamp()
    requests = list((ROOT / 'reports/v3_1/release14_raw').rglob('stage_request.json'))
    requests += list((ROOT / 'reports/v3_1/release14_adapter_raw').rglob('stage_request.json'))
    requests += list((ROOT / 'reports/v3_1/release14_adapter_supervised').rglob('stage_request.json'))
    requests += list((ROOT / 'reports/v3_1/release14_adaptive_reference').rglob('stage_request.json'))
    assert requests
    assert min(path.stat().st_mtime for path in requests) > freeze_time


def test_v31_release_status_is_explicitly_not_safety_certified():
    status = _json('reports/v3_1/RELEASE_STATUS.json')
    assert status['version'] == '3.1.0'
    assert status['adapter_promoted'] is True
    assert status['release14_consumed'] is True
    assert status['geometric_route_assist_used_in_adapter_runtime'] is False
    assert status['curve_tracking_adapter_success_rate'] == 0.0
    assert 'not_safety_certified' in status['status']
