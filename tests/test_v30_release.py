from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _catastrophic(metrics: dict) -> float:
    return float(metrics['collision_rate']) + float(metrics['out_of_road_rate'])


def test_v30_release13_headline_and_claim_boundary():
    report = _json('reports/v3_0/comparison.json')
    assert report['version'] == '3.0.0'
    assert report['release13']['consumed'] is True
    raw = report['release13']['raw']
    fixed = report['release13']['fixed_v29_semantics']
    adaptive = report['release13']['adaptive_v30']

    assert adaptive['success_rate'] == fixed['success_rate'] == 0.8
    assert _catastrophic(adaptive) == _catastrophic(fixed) == 0.2
    assert adaptive['collision_rate'] == 0.0
    assert raw['success_rate'] < adaptive['success_rate']
    assert adaptive['route_assist_mean_blend'] < 0.2
    assert adaptive['learned_steering_authority_fraction'] > 0.8
    assert adaptive['route_assist_intervention_time_fraction'] > 0.5
    assert 'does not change the neural checkpoint' in report['claim_boundary']


def test_v30_paired_result_does_not_claim_fixed_superiority():
    paired = _json('reports/v3_0/comparison.json')['release13']['fixed_to_adaptive_paired_success']
    assert paired == {
        'improved': 1,
        'regressed': 1,
        'both_good': 11,
        'both_bad': 2,
        'exact_mcnemar_p_two_sided': 1.0,
    }


def test_v30_curve_tradeoff_is_preserved():
    release = _json('reports/v3_0/comparison.json')['release13']
    fixed_curve = release['fixed_stages']['curve_tracking']
    adaptive_curve = release['adaptive_stages']['curve_tracking']
    assert adaptive_curve['success_rate'] < fixed_curve['success_rate']
    assert adaptive_curve['route_assist_mean_blend'] < 0.5


def test_v30_system_registry_is_separate_from_learned_registry():
    systems = _json('models/system_registry.json')
    system = systems['metadrive_adaptive_shared_control_v30']
    assert system['version'] == '3.0.0'
    assert 'shared_control' in system['type']
    assert system['release13_consumed'] is True
    assert system['release13_mean_route_blend'] < 0.2
    assert system['release13_learned_steering_authority_fraction'] > 0.8

    learned = _json('models/registry.json')
    serialized = json.dumps(learned)
    assert 'adaptive_shared_control_v30' not in serialized
    assert '3.0.0' not in serialized


def test_release13_freeze_hashes_and_chronology_still_hold():
    freeze = _json('reports/v3_0/release13_freeze.json')
    for relative, expected in freeze['frozen_files'].items():
        assert _sha(ROOT / relative) == expected, relative

    freeze_time = datetime.fromisoformat(freeze['created_at_utc']).astimezone(timezone.utc).timestamp()
    request_files = list((ROOT / 'reports/v3_0/release13_raw').rglob('stage_request.json'))
    request_files += list((ROOT / 'reports/v3_0/release13_fixed').rglob('stage_request.json'))
    request_files += list((ROOT / 'reports/v3_0/release13_adaptive').rglob('stage_request.json'))
    assert request_files
    assert min(path.stat().st_mtime for path in request_files) > freeze_time


def test_v30_release_status_is_explicitly_not_safety_certified():
    status = _json('reports/v3_0/RELEASE_STATUS.json')
    assert status['version'] == '3.0.0'
    assert status['release13_consumed'] is True
    assert status['learned_model_promoted'] is False
    assert status['system_promoted'] is True
    assert status['pre_release_freeze_verified'] is True
    assert 'not_safety_certified' in status['status']
