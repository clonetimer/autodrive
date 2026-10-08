import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def test_v28_artifacts_and_registry_remain_consistent_in_later_releases():
    # Historical v2.8 evidence must remain valid after the package version advances.
    from packaging.version import Version
    import src
    assert Version(src.__version__) >= Version('2.8.0')
    row = _load('models/registry.json')['metadrive_ppo_state_capability_distilled_v28']
    model = ROOT / row['path']
    assert model.is_file()
    assert _sha(model) == row['sha256']
    assert row['expert_distillation_samples'] == 9000
    assert row['release11_raw_success_rate'] == 0.32
    assert 'not_safety_certified' in row['status']


def test_v28_validation7_selects_before_release11_and_release_is_consumed():
    selection = _load('reports/v2_8/selection_validation7.json')
    report = _load('reports/v2_8/comparison.json')
    audit = _load('reports/v2_8/FREEZE_AUDIT.json')
    assert selection['selection_domain'] == 'validation7_seeds_22000_22199_only'
    assert selection['selected'] == 'pureexpert'
    assert selection['promotion']['promoted'] is True
    assert report['selection']['release11_used_for_selection'] is False
    assert report['release11']['slice'] == '23000-23024'
    assert report['release11']['consumed'] is True
    assert audit['model_and_selection_predate_release11'] is True
    assert audit['formal_freeze_manifest_created_before_release'] is False
    assert _sha(ROOT / report['training']['model']) == report['training']['model_sha256']


def test_v28_release11_improves_aggregate_but_keeps_stage_failures_visible():
    report = _load('reports/v2_8/comparison.json')['release11']
    assert report['candidate']['success_rate'] > report['base']['success_rate']
    assert report['candidate']['collision_rate'] < report['base']['collision_rate']
    assert report['candidate']['out_of_road_rate'] < report['base']['out_of_road_rate']
    assert report['catastrophic_change_pp'] < 0
    assert report['paired_success']['improved'] == 8
    assert report['paired_success']['regressed'] == 1
    # Aggregate improvement does not erase the per-stage regressions.
    assert report['candidate_by_stage']['curve_tracking']['out_of_road_rate'] == 1.0
    assert report['stage_deltas']['ramp_merge']['success_change_pp'] < 0


def test_v28_training_teacher_is_explicitly_not_deployed():
    training = _load('reports/v2_8/comparison.json')['training']
    result = _load('experiments/v2_8/capability_pureexpert_v4/result.json')
    assert training['deployment_uses_teacher'] is False
    assert result['deployment_uses_expert'] is False
    assert result['steering_blend'] == 1.0 and result['longitudinal_blend'] == 1.0
    assert result['aggregate_collection']['unique_seeds'] == 30


def test_v28_release_status_keeps_failed_capabilities_explicit():
    status = _load('reports/v2_8/RELEASE_STATUS.json')
    assert status['release11_consumed'] is True
    assert status['teacher_deployed'] is False
    assert status['acceptance']['aggregate_capability_improvement'].startswith('confirmed')
    assert status['acceptance']['curve_tracking_behavior_target'].startswith('failed')
    assert status['acceptance']['ramp_merge_behavior_target'].startswith('regressed')
