import hashlib
import json
from pathlib import Path
from src.agents.trainer import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def test_v25_curated_safety_distilled_model_has_provenance_and_hash():
    registry=json.loads((ROOT/'models/registry.json').read_text())
    entry=registry['metadrive_ppo_state_safety_distilled_v25']
    path=ROOT/entry['path']
    assert digest(path)==entry['sha256']
    model,config,payload=load_checkpoint(path)
    evidence=payload['extra']['safety_distillation']
    assert config.env.backend=='metadrive' and config.env.observation=='state'
    assert config.env.safety_supervisor is False
    assert evidence['source_model']=='models/metadrive_ppo_state_32768.pt'
    assert evidence['critic_trained_during_distillation'] is False
    assert evidence['log_std_trained_during_distillation'] is False
    assert evidence['deployment_safety_supervisor_enabled'] is False
    assert evidence['selection_protocol']['split']=='validation'
    assert 'release4 was not evaluated until after candidate freeze' in evidence['selection_protocol']['criterion']
    assert model is not None


def test_v25_release4_evidence_is_paired_and_preserves_claim_boundary():
    base=json.loads((ROOT/'reports/v2_5/release4/base_raw/evaluation.json').read_text())
    distilled=json.loads((ROOT/'reports/v2_5/release4/curated_raw/evaluation.json').read_text())
    base_sup=json.loads((ROOT/'reports/v2_5/release4/base_supervised/evaluation.json').read_text())
    dist_sup=json.loads((ROOT/'reports/v2_5/release4/curated_supervised/evaluation.json').read_text())
    assert base['seeds']==distilled['seeds']==base_sup['seeds']==dist_sup['seeds']==list(range(7000,7040))
    assert distilled['metrics']['collision_rate'] < base['metrics']['collision_rate']
    assert distilled['metrics']['success_rate'] > base['metrics']['success_rate']
    # Raw learned policy is still not safe and must not be hidden by system metrics.
    assert distilled['metrics']['collision_rate'] > 0
    assert base_sup['metrics']['collision_rate']==dist_sup['metrics']['collision_rate']==0
    assert dist_sup['metrics']['safety_intervention_time_fraction'] < base_sup['metrics']['safety_intervention_time_fraction']
    assert dist_sup['metrics']['safety_mean_longitudinal_correction'] < base_sup['metrics']['safety_mean_longitudinal_correction']


def test_v25_machine_summary_matches_frozen_reports():
    summary=json.loads((ROOT/'reports/v2_5/comparison.json').read_text())
    assert summary['split']=='release4'
    assert summary['raw_policy']['paired_success_transitions']=={
        'improved':10,'regressed':3,'success_both':14,'failure_both':13}
    assert summary['supervised_system']['seeds_with_lower_intervention_fraction']==38
    assert summary['raw_policy']['distilled']['collision_rate']==0.4
    assert summary['supervised_system']['distilled']['collision_rate']==0.0
