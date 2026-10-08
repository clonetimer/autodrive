"""Development validation of the predeclared improvement matrix.

Old test seeds 2000..2199 are historical regression data. This release uses
3000..3039 for development validation and leaves 7000..7199 untouched.
No success-rate gate or model choice uses the future holdout.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import json
import _bootstrap
import numpy as np
from src.end_to_end.agent import EndToEndAgent
from src.baselines.pid_controller import PIDController
from src.baselines.idm_controller import IDMController
from src.evaluation.evaluator import evaluate
from src.utils.config import EnvConfig
from src.utils.logger import save_json, append_csv, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', default='experiments/v2_1')
    parser.add_argument('--output', default='reports/v2_1/evaluations')
    parser.add_argument('--section', choices=['state', 'vision'], required=True)
    parser.add_argument('--episodes', type=int, default=40)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 40:
        parser.error('This frozen development protocol contains 40 scenarios per family')
    output, models = Path(args.output), Path(args.models)
    seeds = list(range(3000, 3000 + args.episodes))
    definitions = {
        'cruise': {'traffic_vehicles': 3, 'randomize': False, 'scenario': 'cruise'},
        'dense': {'traffic_vehicles': 12, 'randomize': True, 'curve_amplitude': 4.0, 'scenario': 'cruise'},
        'emergency': {'traffic_vehicles': 2, 'randomize': False, 'scenario': 'emergency'},
    }
    plan = {'seeds': seeds, 'families': definitions, 'usage': 'development_validation_not_final_holdout',
            'future_holdout_reserved_unused': [7000, 7199],
            'evaluation_reward_version': 3, 'selection': 'last checkpoint, no validation-based selection'}
    plan_path = output / f'plan_{args.section}.json'
    if plan_path.exists():
        raise FileExistsError('Choose a new output or section; previous evaluation cannot be overwritten')
    save_json(plan, plan_path)
    cases = []
    if args.section == 'state':
        for version in (2, 3):
            for seed in (7, 8, 9):
                name = f'ppo_reward_v{version}_seed{seed}'
                cases += [(name, family) for family in definitions]
        cases += [(baseline, family) for baseline in ('pid', 'idm') for family in definitions]
    else:
        cases += [(f'ppo_bev_ego_frames{frames}', 'cruise') for frames in (1, 4)]
        cases += [('mappo_bev_ego_frames4', 'multi_agent_cruise'), ('sac_resumed', 'cruise')]
    summary = {}
    for name, family in cases:
        if name in ('pid', 'idm'):
            config = EnvConfig(road_length=180, max_steps=240, target_speed=12, initial_speed=8)
            factory = (lambda agent, seed: PIDController(config.dt)) if name == 'pid' else (lambda agent, seed: IDMController())
            trained_version, model_hash = None, None
        else:
            policy = EndToEndAgent(models / name / 'model.pt')
            config = policy.config.env
            factory = lambda agent, seed: policy
            trained_version, model_hash = config.reward_version, policy.fingerprint
        config = replace(config, reward_version=3, **definitions.get(family, {'traffic_vehicles': 2}))
        case_seeds = seeds[:20] if family == 'multi_agent_cruise' else seeds
        report = evaluate(factory, config, case_seeds, output / f'{name}_{family}', name)
        row = {'name': name, 'family': family, 'trained_reward_version': trained_version,
               'evaluation_reward_version': 3, 'model_sha256': model_hash,
               'scenario_count': len(case_seeds), **report['metrics']}
        summary[name + '/' + family] = row
        save_json(summary, output / f'summary_{args.section}.json')
        append_csv(output / f'summary_{args.section}.csv', row)
        print(name, family, report['metrics']['success_rate'], report['metrics']['collision_rate'], flush=True)
    if args.section == 'state':
        effects = {}
        for family in definitions:
            per_seed, differences = [], []
            for seed in (7, 8, 9):
                records = []
                for version in (2, 3):
                    path = output / f'ppo_reward_v{version}_seed{seed}_{family}/evaluation.json'
                    records.append(json.loads(path.read_text())['episodes'])
                # All agents here are single ego; scene seeds form paired columns.
                delta = np.asarray([float(b['success']) - float(a['success']) for a, b in zip(*records)])
                differences.append(delta)
                per_seed.append(float(delta.mean()))
            # Two-way bootstrap: resample training seeds AND shared scenario IDs.
            matrix = np.stack(differences)
            rng = np.random.default_rng(20260907)
            draws = np.asarray([matrix[rng.integers(3, size=3)][:, rng.integers(len(seeds), size=len(seeds))].mean()
                                for _ in range(2000)])
            effects[family] = {'success_rate_change_v3_minus_v2': float(matrix.mean()),
                'per_training_seed_changes': per_seed, 'two_way_bootstrap_ci95': np.quantile(draws, [.025, .975]).tolist(),
                'warning': 'Three training seeds only; development validation, not a safety or final generalization claim'}
        save_json(effects, output / 'paired_effects.json')


if __name__ == '__main__':
    main()
