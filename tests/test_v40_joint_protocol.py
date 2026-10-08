from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys

from src.scenarios.generator import SPLITS, split_seeds


def _load():
    root=Path(__file__).resolve().parents[1]
    scripts=root/'scripts'
    if str(scripts) not in sys.path: sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('v40_launcher',scripts/'run_v40_joint_ablation.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _args(tmp_path):
    return SimpleNamespace(
        variants=['P0','P1','M0','M1'], training_seeds=[6400,7400,8400],
        output=str(tmp_path/'out'), base_model=str(tmp_path/'base.pt'),
        iterations=3, seeds_per_stage=1, mix_betas='1.0,0.5,0.2',
        epochs=40, batch_size=512, learning_rate=3e-4, hidden_size=128,
        history_len=6, neighbor_heads=4, max_steering_correction=.18,
        max_longitudinal_correction=1.5, longitudinal_loss_weight=1.0,
        teacher_blend=.5, max_stage_steps=420, stage_timeout=150., torch_threads=1,
        resume=False)


def test_v40_plan_is_2x2_paired_and_training_only(tmp_path):
    mod=_load(); jobs=mod._command_plan(_args(tmp_path))
    assert len(jobs)==12
    assert {j['variant'] for j in jobs}=={'P0','P1','M0','M1'}
    assert {j['training_seed'] for j in jobs}=={6400,7400,8400}
    mapping={j['variant']:j for j in jobs if j['training_seed']==6400}
    assert mapping['P0']['collection_mode']=='protected' and not mapping['P0']['enable_longitudinal']
    assert mapping['P1']['collection_mode']=='protected' and mapping['P1']['enable_longitudinal']
    assert mapping['M0']['collection_mode']=='mixed' and not mapping['M0']['enable_longitudinal']
    assert mapping['M1']['collection_mode']=='mixed' and mapping['M1']['enable_longitudinal']
    for job in jobs:
        cmd=job['train_command']
        assert cmd[cmd.index('--split')+1]=='joint_train'
        assert all('validation14' not in token for token in cmd)
        assert all('release17' not in token for token in cmd)


def test_v40_new_domains_are_disjoint_and_release16_stays_historical():
    joint=set(range(*[SPLITS['joint_train'][0],SPLITS['joint_train'][1]+1]))
    val=set(range(SPLITS['validation14'][0],SPLITS['validation14'][1]+1))
    rel=set(range(SPLITS['release17'][0],SPLITS['release17'][1]+1))
    oldrel=set(range(SPLITS['release16'][0],SPLITS['release16'][1]+1))
    assert joint.isdisjoint(val) and joint.isdisjoint(rel) and val.isdisjoint(rel)
    assert oldrel.isdisjoint(joint|val|rel)
    assert split_seeds('joint_train',1)==[41000]
    assert split_seeds('validation14',1)==[42000]
    assert split_seeds('release17',1)==[43000]


def test_v40_matrix_resume_is_job_aware(tmp_path):
    mod=_load(); args=_args(tmp_path); args.resume=True
    job=mod._command_plan(args)[0]
    assert '--resume' not in job['train_command']
    assert '--resume' not in mod._execution_command(job, True)
    p=Path(job['train_output']); p.mkdir(parents=True); (p/'run_config.json').write_text('{}')
    assert mod._execution_command(job, True)[-1]=='--resume'
