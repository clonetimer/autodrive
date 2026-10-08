import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_v29_release_report_preserves_hybrid_claim_boundary_and_fresh_freeze():
    report = json.loads((ROOT / 'reports/v2_9/comparison.json').read_text(encoding='utf-8'))
    assert report['version'] == '2.9.0'
    assert report['learned_model_promoted'] is False
    assert report['pre_release_freeze']['freeze_predates_first_release_request'] is True
    assert report['pre_release_freeze']['frozen_source_hashes_match'] is True
    assert report['release12']['consumed'] is True
    assert report['release12']['paired_success'] == {
        'improved': 10,
        'regressed': 1,
        'both_good': 4,
        'both_bad': 0,
        'exact_mcnemar_p_two_sided': 0.01171875,
    }
    # The hybrid result must remain visibly rule-assisted, not relabeled as raw RL.
    assert report['release12']['hybrid']['route_assist_intervention_time_fraction'] > 0.99
    assert report['release12']['hybrid']['safety_intervention_time_fraction'] > 0.0
    assert 'hybrid neural+rule system' in report['system']['claim_boundary']


def test_incomplete_v29_schema4_run_is_explicitly_nonrelease():
    status = json.loads((ROOT / 'experiments/v2_9/dev_partial_route_schema4_expert/STATUS.json').read_text())
    assert status['status'].startswith('incomplete')
    assert status['use_for_release_selection'] is False
    assert status['use_for_release12'] is False


def test_v29_system_registry_is_separate_from_learned_model_registry():
    systems = json.loads((ROOT / 'models/system_registry.json').read_text(encoding='utf-8'))
    learned = json.loads((ROOT / 'models/registry.json').read_text(encoding='utf-8'))
    row = systems['metadrive_hybrid_route_safety_v29']
    assert 'metadrive_hybrid_route_safety_v29' not in learned
    assert _sha_file(ROOT / row['learned_policy']) == row['learned_policy_sha256']
    assert _sha_file(ROOT / row['configuration']) == row['configuration_sha256']
    assert row['release12_consumed'] is True
    assert 'not_raw_policy' in row['status']


def _sha_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
