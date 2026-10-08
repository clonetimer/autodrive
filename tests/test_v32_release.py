from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v32_selection_is_validation_only_and_release15_was_unopened_at_gate():
    selection = _json('reports/v3_2/selection_validation11.json')
    assert selection['protocol']['selection_only'] == 'validation11'
    assert selection['protocol']['release15_opened_before_selection'] is False
    assert all(selection['checks'].values())
    assert selection['promoted'] is True
    assert selection['candidate']['success_rate'] == pytest.approx(1.0)
    assert selection['candidate']['collision_rate'] == pytest.approx(0.0)
    assert selection['candidate']['out_of_road_rate'] == pytest.approx(0.0)


def test_v32_fresh_release15_improves_directionally_without_runtime_rules():
    report = _json('reports/v3_2/comparison.json')
    assert report['evaluation_split'] == 'release15'
    assert report['release15_consumed'] is True
    assert report['primary_runtime_rules'] == {'route_assist': False, 'safety_supervisor': False}
    base = report['baseline']['metrics']
    cand = report['candidate']['metrics']
    assert base['success_rate'] == pytest.approx(9 / 15)
    assert cand['success_rate'] == pytest.approx(14 / 15)
    assert cand['collision_rate'] == 0.0
    assert cand['out_of_road_rate'] == pytest.approx(1 / 15)
    assert cand['catastrophic_rate'] == pytest.approx(1 / 15)
    assert cand['mean_route_completion'] > base['mean_route_completion']
    assert cand['lane_abs_error_m'] < base['lane_abs_error_m']
    assert cand['route_assist_intervention_time_fraction'] == 0.0
    assert cand['safety_intervention_time_fraction'] == 0.0


def test_v32_paired_release15_result_and_curve_failure_are_preserved():
    report = _json('reports/v3_2/comparison.json')
    paired = report['paired_success']
    assert paired['failure_to_success'] == 5
    assert paired['success_to_failure'] == 0
    assert paired['both_success'] == 9
    assert paired['both_failure'] == 1
    assert paired['exact_mcnemar_p'] == pytest.approx(0.0625)
    curve = report['candidate']['stages']['curve_tracking']
    assert curve['success_rate'] == pytest.approx(2 / 3)
    assert curve['out_of_road_rate'] == pytest.approx(1 / 3)
    # Do not silently convert the remaining curve failure into a success claim.
    assert any(p['stage'] == 'curve_tracking' and not p['candidate_success'] for p in report['pairs'])


def test_v32_adapter_registry_hash_base_binding_and_release_metadata():
    row = _json('models/route_adapter_registry.json')['metadrive_curve_route_adapter_v32']
    assert row['version'] == '3.2.0'
    assert row['runtime_geometric_route_assist'] is False
    assert row['runtime_safety_supervisor_for_raw_evaluation'] is False
    assert row['release15_consumed'] is True
    assert row['release15_success_rate'] == pytest.approx(14 / 15)
    assert _sha(ROOT / row['path']) == row['sha256']
    assert _sha(ROOT / row['base_model']) == row['base_model_sha256']
    assert 'not_safety_certified' in row['status']


def test_v32_release15_freeze_hashes_snapshots_and_chronology_hold():
    freeze = _json('reports/v3_2/release15_freeze.json')
    assert freeze['status'] == 'frozen_before_release15_first_open'
    assert freeze['release15_consumed_at_freeze'] is False
    assert freeze['runtime_rules_for_primary_release15'] == 'none'
    # Registry changed only after release; preserve the exact pre-release bytes separately.
    frozen_registry = ROOT / 'reports/v3_2/frozen_route_adapter_registry.json'
    assert _sha(frozen_registry) == freeze['route_adapter_registry_sha256']
    for relative, meta in freeze['frozen_sources'].items():
        snapshot = ROOT / meta['snapshot']
        assert snapshot.is_file(), relative
        assert _sha(snapshot) == meta['snapshot_sha256'] == meta['current_sha256']
    freeze_time = datetime.fromisoformat(freeze['created_utc']).astimezone(timezone.utc).timestamp()
    requests = list((ROOT / 'reports/v3_2/release15_v31_raw_curve_ramp').rglob('stage_request.json'))
    requests += list((ROOT / 'reports/v3_2/release15_v31_raw_other').rglob('stage_request.json'))
    requests += list((ROOT / 'reports/v3_2/release15_v32_raw_curve_ramp').rglob('stage_request.json'))
    requests += list((ROOT / 'reports/v3_2/release15_v32_raw_other').rglob('stage_request.json'))
    assert requests
    assert min(p.stat().st_mtime for p in requests) > freeze_time


def test_v32_seed_domains_are_disjoint_and_release15_is_not_a_training_domain():
    from src.scenarios.generator import split_seeds
    domains = ['curve_train', 'validation11', 'release15']
    sets = {name: set(split_seeds(name, count=15 if name != 'curve_train' else 12)) for name in domains}
    assert sets['curve_train'].isdisjoint(sets['validation11'])
    assert sets['curve_train'].isdisjoint(sets['release15'])
    assert sets['validation11'].isdisjoint(sets['release15'])


def test_v32_release_status_is_explicitly_not_safety_certified():
    status = _json('reports/v3_2/RELEASE_STATUS.json')
    assert status['version'] == '3.2.0'
    assert status['adapter_promoted_on_validation11'] is True
    assert status['release15_consumed'] is True
    assert status['runtime_geometric_route_assist'] is False
    assert status['runtime_safety_supervisor_primary'] is False
    assert status['release15_curve_success_rate'] == pytest.approx(2 / 3)
    assert 'not_safety_certified' in status['status']
    assert 'not a safety certification' in status['claim_boundary']
