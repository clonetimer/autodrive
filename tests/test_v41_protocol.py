"""Protocol/governance tests for v4.1 factorized longitudinal learning."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys

from src.scenarios.generator import SPLITS, split_seeds


def _load_launcher():
    root = Path(__file__).resolve().parents[1]
    scripts = root / 'scripts'
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        'v41_launcher', scripts / 'run_v41_longitudinal_ablation.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_evaluator():
    root = Path(__file__).resolve().parents[1]
    scripts = root / 'scripts'
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        'v41_evaluator', scripts / 'evaluate_learned_route_suite.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(tmp_path):
    return SimpleNamespace(
        variants=['LP', 'LM'], training_seeds=[9400, 10400, 11400],
        output=str(tmp_path/'out'), base_model=str(tmp_path/'base.pt'),
        lateral_adapter=str(tmp_path/'lat.pt'), iterations=3, seeds_per_stage=1,
        mix_betas='1.0,0.5,0.2', epochs=40, batch_size=512,
        learning_rate=3e-4, hidden_size=128, history_len=6, neighbor_heads=4,
        use_neighbors=True, max_longitudinal_correction=1.5,
        max_stage_steps=420, stage_timeout=150., ttc_threshold=1.8,
        risk_threshold=.68, torch_threads=2, resume=False)


def test_v41_plan_is_paired_2x3_and_training_only(tmp_path):
    mod = _load_launcher(); jobs = mod._command_plan(_args(tmp_path))
    assert len(jobs) == 6
    assert {j['variant'] for j in jobs} == {'LP', 'LM'}
    assert {j['training_seed'] for j in jobs} == {9400, 10400, 11400}
    mapping = {j['variant']: j for j in jobs if j['training_seed'] == 9400}
    assert mapping['LP']['collection_mode'] == 'protected'
    assert mapping['LM']['collection_mode'] == 'mixed'
    for job in jobs:
        cmd = job['train_command']
        assert cmd[cmd.index('--split') + 1] == 'longitudinal_train'
        assert '--lateral-adapter' in cmd
        assert all('validation15' not in token for token in cmd)
        assert all('release18' not in token for token in cmd)
        # There is no steering/joint-action training switch in v4.1.
        assert '--max-steering-correction' not in cmd
        assert '--enable-longitudinal' not in cmd


def test_v41_domains_are_fresh_and_pairwise_disjoint():
    def domain(name):
        lo, hi = SPLITS[name]
        return set(range(lo, hi + 1))
    train = domain('longitudinal_train')
    val = domain('validation15')
    rel = domain('release18')
    old_val = domain('validation14')
    old_rel = domain('release17')
    assert train.isdisjoint(val) and train.isdisjoint(rel) and val.isdisjoint(rel)
    assert (train | val | rel).isdisjoint(old_val | old_rel)
    assert split_seeds('longitudinal_train', 1) == [44000]
    assert split_seeds('validation15', 1) == [45000]
    assert split_seeds('release18', 1) == [46000]


def test_v41_matrix_resume_is_job_aware(tmp_path):
    mod = _load_launcher(); args = _args(tmp_path); args.resume = True
    job = mod._command_plan(args)[0]
    assert '--resume' not in job['train_command']
    assert '--resume' not in mod._execution_command(job, True)
    p = Path(job['train_output']); p.mkdir(parents=True)
    (p/'run_config.json').write_text('{}')
    assert mod._execution_command(job, True)[-1] == '--resume'


def test_v41_evaluator_stage_request_binds_both_factorized_artifacts(tmp_path):
    mod = _load_evaluator()
    base = tmp_path/'base.pt'; base.write_bytes(b'base')
    lat = tmp_path/'lat.pt'; lat.write_bytes(b'lateral')
    long = tmp_path/'long.pt'; long.write_bytes(b'longitudinal')
    from src.evaluation.capability_suite import standard_capability_suite
    spec = standard_capability_suite('validation15', 1)[0]
    request = mod._stage_request(
        str(base), spec, False, 1.8, .68,
        lateral_adapter=str(lat), longitudinal_adapter=str(long))
    assert request['lateral_adapter'] == str(lat.resolve())
    assert request['longitudinal_adapter'] == str(long.resolve())
    assert request['lateral_adapter_sha256']
    assert request['longitudinal_adapter_sha256']
    assert request['steering_adapter'] is None


def test_v41_interaction_suite_uses_disjoint_validation_seed_block():
    from src.evaluation.capability_suite import standard_capability_suite
    from src.evaluation.interaction_suite import interaction_stress_suite
    standard = standard_capability_suite('validation15', 3)
    stress = interaction_stress_suite('validation15', 3)
    standard_seeds = {seed for spec in standard for seed in spec.seeds}
    stress_seeds = {seed for spec in stress for seed in spec.seeds}
    assert standard_seeds == set(range(45000, 45015))
    assert stress_seeds == set(range(45050, 45062))
    assert standard_seeds.isdisjoint(stress_seeds)
    assert [s.name for s in stress] == [
        'ramp_merge_stress', 'bottleneck_stress',
        'intersection_stress', 'dense_progress_stress']
    standard_dynamic = standard_capability_suite('validation15', 3)[1:]
    for stressed, baseline in zip(stress, standard_dynamic):
        assert stressed.traffic_density > baseline.traffic_density
