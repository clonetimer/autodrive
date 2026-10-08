"""Evaluate all frozen release models without selecting a model on test outcomes.

Main table uses 20 fixed held-out scenarios. Diagnostic comparisons use a
separate 20-scenario validation partition. Different training budgets are
reported rather than advertised as controlled algorithm superiority.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import _bootstrap
from src.baselines.random_agent import RandomAgent
from src.baselines.pid_controller import PIDController
from src.baselines.idm_controller import IDMController
from src.end_to_end.agent import EndToEndAgent
from src.evaluation.benchmark import generalization_suite
from src.evaluation.evaluator import evaluate
from src.scenarios.generator import split_seeds
from src.utils.config import load_config
from src.utils.logger import save_json


def run(training, output, episodes=20):
    training, output = Path(training), Path(output)
    if (output / 'summary.json').exists():
        raise FileExistsError('Use a fresh evaluation output directory')
    test_seeds = split_seeds('test', episodes)
    val_seeds = split_seeds('validation', episodes)
    base = load_config(_bootstrap.ROOT / 'configs/ppo_state.yaml').env
    results = {'protocol': {'test_seeds': test_seeds, 'validation_seeds': val_seeds,
               'selection': 'final budget checkpoints fixed before test; no test-based model selection',
               'main_single_world': asdict(base),
               'warning': 'Short-budget research measurements; budgets/architectures differ. MARL is a separate task.'},
               'single_agent_test': {}, 'multi_agent_test': {}, 'validation_diagnostics': {}}

    def record(group, label, factory, config, seeds, checkpoint=None):
        report = evaluate(factory, config, seeds, output / group / label, label)
        results[group][label] = {'metrics': report['metrics'], 'world_steps': report['world_steps'],
                                'environment': asdict(config), 'checkpoint': checkpoint}
        save_json(results, output / 'summary.json')
        print(f'{group}/{label}: success={report["metrics"]["success_rate"]:.3f}, '
              f'collision={report["metrics"]["collision_rate"]:.3f}, '
              f'offroad={report["metrics"]["out_of_road_rate"]:.3f}', flush=True)

    for label, constructor in [('random', lambda seed: RandomAgent(seed)),
                                ('pid', lambda seed: PIDController(base.dt)),
                                ('idm', lambda seed: IDMController())]:
        record('single_agent_test', label, lambda name, seed, ctor=constructor: ctor(seed), base, test_seeds)
    for name in ['ppo_state_seed7', 'ppo_state_seed8', 'ppo_state_seed9', 'sac_state', 'ppo_bev']:
        path = training / name / 'model.pt'
        agent = EndToEndAgent(path)
        cfg = replace(base, observation=agent.config.env.observation)
        record('single_agent_test', name, lambda name, seed, a=agent: a, cfg, test_seeds, str(path))
    for name in ['mappo_state', 'mappo_hybrid', 'ippo_state']:
        path = training / name / 'model.pt'
        agent = EndToEndAgent(path)
        record('multi_agent_test', name, lambda name, seed, a=agent: a, agent.config.env, test_seeds, str(path))

    # Equal-budget, same training seed reward comparisons. Reward values cannot
    # be compared across different reward definitions; physical metrics can.
    diagnostics = [
        ('balanced_8192', 'ppo_state_seed7/checkpoints/step_8192.pt'),
        ('safe_8192', 'reward_safe/model.pt'),
        ('efficient_8192', 'reward_efficient/model.pt'),
        ('mappo_8192', 'mappo_state/checkpoints/step_8192.pt'),
        ('ippo_8192', 'ippo_state/model.pt'),
        ('state_16384', 'ppo_state_seed7/checkpoints/step_16384.pt'),
        ('bev_16384', 'ppo_bev/model.pt')]
    for label, relative in diagnostics:
        path = training / relative
        agent = EndToEndAgent(path)
        record('validation_diagnostics', label, lambda name, seed, a=agent: a,
               agent.config.env, val_seeds, str(path))
    results['diagnostic_caveats'] = {
        'reward': 'Same seed, budget, optimizer, dynamics; evaluate physical outcomes, not incomparable shaped returns.',
        'ippo_vs_mappo': 'Same seed, joint-step budget and world; valid active-agent sample counts can differ.',
        'state_vs_bev': 'Same seed/budget/dynamics but rollout and optimizer minibatch sizes differ; NOT an isolated encoder ablation.'}
    results['generalization'] = generalization_suite(training / 'ppo_state_seed7/model.pt', test_seeds, output / 'generalization')
    save_json(results, output / 'summary.json')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', default='experiments/release')
    parser.add_argument('--output', default='reports/release_evaluation')
    parser.add_argument('--episodes', type=int, default=20)
    args = parser.parse_args()
    run(args.training, args.output, args.episodes)
