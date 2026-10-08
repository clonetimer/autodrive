import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def test_v27_version_and_curated_candidate_provenance_are_consistent():
    registry = json.loads((ROOT / 'models/registry.json').read_text(encoding='utf-8'))
    row = registry['metadrive_ppo_state_expert_distilled_v27']
    model = ROOT / row['path']
    assert model.is_file()
    assert _sha(model) == row['sha256']
    assert row['status'] == 'validation2_promoted_release6_not_confirmed_not_safety_certified'
    assert row['teacher'].startswith('MetaDrive IDMPolicy')


def test_v27_freeze_record_binds_model_and_selection_before_release6():
    freeze = json.loads((ROOT / 'reports/v2_7/FROZEN_MODEL.json').read_text(encoding='utf-8'))
    model = ROOT / freeze['model_path']
    selection = ROOT / freeze['selection_report']
    assert freeze['frozen_before_release6'] is True
    assert freeze['release6_used_for_selection'] is False
    assert freeze['promotion_passed'] is True
    assert _sha(model) == freeze['model_sha256']
    assert _sha(selection) == freeze['selection_sha256']


def test_v27_release6_report_records_nonconfirmation_and_consumes_split():
    status = json.loads((ROOT / 'reports/v2_7/RELEASE_STATUS.json').read_text(encoding='utf-8'))
    assert status['release6_consumed'] is True
    assert status['acceptance']['release6_robust_raw_safety_improvement'] == 'not_confirmed'
    assert status['acceptance']['path_following_behavior_target'] == 'failed_high_offroad'
    release = status['release6']
    assert release['selected_raw']['collision_rate'] >= release['base_raw']['collision_rate']
    assert release['selected_raw']['out_of_road_rate'] < release['base_raw']['out_of_road_rate']
    assert release['paired_success']['exact_mcnemar_p_two_sided'] == 1.0


def test_v27_readme_keeps_privileged_teacher_and_safety_boundaries_explicit():
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert 'training-only' in readme or '训练期' in readme
    assert '80.0%' in readme
    assert 'raw-RGB' in readme
    assert '禁止用于实车控制' in (ROOT / 'docs/KNOWN_LIMITATIONS.md').read_text(encoding='utf-8')
