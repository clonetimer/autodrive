from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys


def _load_script():
    root = Path(__file__).resolve().parents[1]
    scripts = root / 'scripts'
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    path = scripts / 'run_v33_risk_aux_ablation.py'
    spec = importlib.util.spec_from_file_location('v33_risk_ablation_script', path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _args(tmp_path):
    return SimpleNamespace(
        variants=['R0', 'R15'],
        training_seeds=[3300, 4300, 5300],
        output=str(tmp_path / 'out'),
        base_model=str(tmp_path / 'base.pt'),
        iterations=3,
        seeds_per_stage=2,
        mix_betas='1.0,0.5,0.2',
        epochs=80,
        batch_size=512,
        learning_rate=3e-4,
        hidden_size=128,
        neighbor_heads=4,
        max_correction=0.18,
        teacher_blend=0.5,
        collection_safety='protected',
        max_stage_steps=420,
        train_stage_timeout=120.0,
        validation_stage_timeout=120.0,
        validation_seeds_per_stage=3,
        focus_stages='',
        resume=False,
    )


def test_risk_aux_ablation_changes_only_auxiliary_weight(tmp_path):
    mod = _load_script()
    jobs = mod._command_plan(_args(tmp_path))
    assert len(jobs) == 6
    first = {j['variant']: j for j in jobs if j['training_seed'] == 3300}
    r0 = first['R0']['train_command']
    r15 = first['R15']['train_command']
    for command in (r0, r15):
        assert command[command.index('--history-len') + 1] == '6'
        assert '--use-neighbors' in command
        assert command[command.index('--split') + 1] == 'scene_train'
    assert r0[r0.index('--risk-loss-weight') + 1] == '0.0'
    assert r15[r15.index('--risk-loss-weight') + 1] == '0.15'


def test_risk_aux_ablation_never_opens_release16(tmp_path):
    mod = _load_script()
    jobs = mod._command_plan(_args(tmp_path))
    for job in jobs:
        validation = job['validation_command']
        assert validation[validation.index('--split') + 1] == 'validation13'
        assert all('release16' not in token for token in job['train_command'])
        assert all('release16' not in token for token in validation)


def test_risk_matrix_resume_is_job_aware(tmp_path):
    mod = _load_script()
    args = _args(tmp_path)
    args.resume = True
    job = mod._command_plan(args)[0]
    assert '--resume' not in job['train_command']
    assert '--resume' not in job['validation_command']
    assert '--resume' not in mod._train_command_for_execution(job, resume=True)
    train_dir = Path(job['train_output'])
    train_dir.mkdir(parents=True)
    (train_dir / 'run_config.json').write_text('{}')
    assert mod._train_command_for_execution(job, resume=True)[-1] == '--resume'
