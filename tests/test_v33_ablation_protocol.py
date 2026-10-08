from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys

from src.scenarios.generator import SPLITS, split_seeds


def _load_script():
    root = Path(__file__).resolve().parents[1]
    scripts = root / 'scripts'
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    path = scripts / 'run_v33_representation_ablation.py'
    spec = importlib.util.spec_from_file_location('v33_ablation_script', path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _args(tmp_path):
    return SimpleNamespace(
        variants=['A', 'B', 'C', 'D'],
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


def test_v33_ablation_plan_is_paired_and_never_opens_release16(tmp_path):
    mod = _load_script()
    jobs = mod._command_plan(_args(tmp_path))
    assert len(jobs) == 12
    assert {j['training_seed'] for j in jobs} == {3300, 4300, 5300}
    assert {j['variant'] for j in jobs} == {'A', 'B', 'C', 'D'}
    for job in jobs:
        assert '--split' in job['validation_command']
        split = job['validation_command'][job['validation_command'].index('--split') + 1]
        assert split == 'validation13'
        assert all('release16' not in token for token in job['train_command'])
        assert all('release16' not in token for token in job['validation_command'])
        assert '--training-seed' in job['train_command']


def test_v33_ablation_flags_match_registered_variants(tmp_path):
    mod = _load_script()
    jobs = mod._command_plan(_args(tmp_path))
    first = {j['variant']: j for j in jobs if j['training_seed'] == 3300}
    assert first['A']['train_command'][first['A']['train_command'].index('--history-len') + 1] == '1'
    assert '--no-use-neighbors' in first['A']['train_command']
    assert first['B']['train_command'][first['B']['train_command'].index('--history-len') + 1] == '6'
    assert '--no-use-neighbors' in first['B']['train_command']
    assert first['C']['train_command'][first['C']['train_command'].index('--history-len') + 1] == '1'
    assert '--use-neighbors' in first['C']['train_command']
    assert first['D']['train_command'][first['D']['train_command'].index('--history-len') + 1] == '6'
    assert '--use-neighbors' in first['D']['train_command']
    # Representation ablation must not be confounded by auxiliary risk labels.
    for job in first.values():
        command = job['train_command']
        assert command[command.index('--risk-loss-weight') + 1] == '0.0'


def test_v33_ablation_exposes_bounded_worker_controls(tmp_path):
    mod = _load_script()
    jobs = mod._command_plan(_args(tmp_path))
    command = jobs[0]['train_command']
    validation = jobs[0]['validation_command']
    assert command[command.index('--max-stage-steps') + 1] == '420'
    assert command[command.index('--stage-timeout') + 1] == '120.0'
    assert validation[validation.index('--stage-timeout') + 1] == '120.0'


def test_v33_clean_selection_domain_moved_after_validation12_smoke():
    assert split_seeds('validation12', 1) == [38000]
    assert split_seeds('validation13', 3) == [40000, 40001, 40002]
    validation13 = set(range(SPLITS['validation13'][0], SPLITS['validation13'][1] + 1))
    release16 = set(range(SPLITS['release16'][0], SPLITS['release16'][1] + 1))
    scene_train = set(range(SPLITS['scene_train'][0], SPLITS['scene_train'][1] + 1))
    assert validation13.isdisjoint(release16)
    assert validation13.isdisjoint(scene_train)


def test_scene_adapter_default_risk_auxiliary_is_opt_in():
    root = Path(__file__).resolve().parents[1]
    scripts = root / 'scripts'
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from scripts.train_scene_aware_adapter import DEFAULT_RISK_LOSS_WEIGHT
    assert DEFAULT_RISK_LOSS_WEIGHT == 0.0


def test_matrix_resume_is_job_aware_for_fresh_and_partial_trials(tmp_path):
    mod = _load_script()
    args = _args(tmp_path)
    args.resume = True
    job = mod._command_plan(args)[0]
    # The frozen plan is invariant; --resume is injected only at execution time.
    assert '--resume' not in job['train_command']
    assert '--resume' not in job['validation_command']
    fresh = mod._train_command_for_execution(job, resume=True)
    assert '--resume' not in fresh
    train_dir = Path(job['train_output'])
    train_dir.mkdir(parents=True)
    (train_dir / 'run_config.json').write_text('{}')
    partial = mod._train_command_for_execution(job, resume=True)
    assert partial[-1] == '--resume'


def test_matrix_plan_fingerprint_ignores_runtime_status_fields(tmp_path):
    mod = _load_script()
    manifest = {'schema': 'x', 'jobs': mod._command_plan(_args(tmp_path)), 'status': 'running'}
    a = mod._plan_fingerprint(manifest)
    manifest['status'] = 'training_complete'
    manifest['job_status'] = {'A_seed3300': 'train_complete'}
    manifest['plan_fingerprint_sha256'] = 'ignored'
    assert mod._plan_fingerprint(manifest) == a
