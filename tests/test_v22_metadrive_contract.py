"""Pure adapter-unit/config tests, not evidence that Panda3D has run."""
from pathlib import Path
import subprocess
import sys
import json
import numpy as np
import pytest
from src.envs.metadrive_contract import reference_plan
from src.envs.metadrive_wrapper import simulator_config, steering_radians
from src.utils.config import EnvConfig


def test_scenario_env_does_not_receive_procedural_only_keys(tmp_path):
    cfg = simulator_config(EnvConfig(backend='metadrive', dataset=str(tmp_path)))
    for key in ['map', 'traffic_density', 'start_seed', 'out_of_road_done']:
        assert key not in cfg
    assert cfg['crash_human_done'] and not cfg['reactive_traffic']
    assert cfg['num_scenarios'] == -1


def test_unsupported_multiagent_dataset_fails_before_engine_import():
    with pytest.raises(ValueError, match='single learning ego'):
        simulator_config(EnvConfig(backend='metadrive', dataset='/data', num_agents=2))


def test_joint_env_no_respawn_and_control_frequency():
    cfg = simulator_config(EnvConfig(backend='metadrive', num_agents=2, dt=.1, substeps=5))
    assert cfg['num_agents'] == 2 and not cfg['allow_respawn']
    assert cfg['physics_world_step_size'] * cfg['decision_repeat'] == pytest.approx(.1)


def test_steering_uses_actual_limit_in_degrees_not_old_magic_045():
    assert steering_radians(.5, 60.) == pytest.approx(np.pi / 6)
    assert steering_radians(-1., 45.) == pytest.approx(-np.pi / 4)
    with pytest.raises(ValueError): steering_radians(0., 0.)
    with pytest.raises(ValueError): steering_radians(float('nan'), 45.)


def test_reference_plan_disjoint_seeds_and_real_gym_api():
    plan = reference_plan()
    assert not set(range(100)) & set(plan['evaluation_seeds'])
    assert plan['observation'] == 'official_state' and plan['real_steps_executed'] == 0
    assert 'headless' not in plan['train_config'] and 'env_num' not in plan['train_config']
    with pytest.raises(ValueError, match='overlap'):
        reference_plan(train_start=0, train_count=100, eval_start=90)


def test_official_reference_plan_only_cli_outside_repo(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/train_metadrive_official.py'
    result = subprocess.run([sys.executable, str(script), '--plan-only', '--output', str(tmp_path / 'run')],
                            cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / 'run/result.json').read_text())
    assert report['status'] == 'plan_only_not_training'
    assert report['real_steps_executed'] == 0
    assert not (tmp_path / 'run/model.zip').exists()
