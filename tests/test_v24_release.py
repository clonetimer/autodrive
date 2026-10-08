import hashlib
import json
from pathlib import Path

from src.agents.trainer import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def test_v24_curated_rgb_models_have_valid_hashes_and_load():
    registry = json.loads((ROOT / 'models/registry.json').read_text())
    for name in ('metadrive_ppo_rgb_ego_384', 'metadrive_rgb_distilled_48_multiseed'):
        entry = registry[name]
        model_path = ROOT / entry['path']
        assert model_path.is_file()
        assert _sha256(model_path) == entry['sha256']
        model, config, payload = load_checkpoint(model_path)
        assert config.env.backend == 'metadrive'
        assert config.env.observation == 'rgb_ego'
        assert payload['format_version'] >= 2
        assert model is not None


def test_distilled_rgb_checkpoint_records_privileged_teacher_provenance():
    registry = json.loads((ROOT / 'models/registry.json').read_text())
    entry = registry['metadrive_rgb_distilled_48_multiseed']
    _, config, payload = load_checkpoint(ROOT / entry['path'])
    evidence = payload['extra']['distillation']
    assert config.env.observation == 'rgb_ego'
    assert evidence['teacher_observation'] == 'state'
    assert evidence['student_observation'] == 'rgb_ego'
    assert evidence['teacher_actions_executed_during_collection'] is True
    assert evidence['critic_trained'] is False
    assert evidence['log_std_trained'] is False
    assert evidence['final_mse'] < evidence['initial_mse'] * 0.01


def test_release3_evidence_keeps_raw_and_supervised_systems_separate():
    raw = json.loads((ROOT / 'reports/v2_4/release3_source_raw/evaluation.json').read_text())
    supervised = json.loads((ROOT / 'reports/v2_4/release3_source_supervised/evaluation.json').read_text())
    assert raw['metrics']['safety_intervention_time_fraction'] == 0.0
    assert raw['metrics']['collision_rate'] > 0.0
    assert supervised['env_config']['safety_supervisor'] is True
    assert supervised['metrics']['collision_rate'] == 0.0
    assert supervised['metrics']['safety_intervention_time_fraction'] > 0.0
    assert raw['seeds'] == supervised['seeds']


def test_v24_curated_registry_never_points_to_dev_smoke_artifacts():
    registry = json.loads((ROOT / 'models/registry.json').read_text())
    for entry in registry.values():
        assert 'dev_smoke' not in entry.get('path', '')
        assert 'dev_smoke' not in entry.get('evaluation', '')
