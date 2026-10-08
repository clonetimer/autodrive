"""Reproduce the CPU experiment matrix; each completed run has its own evidence.

No native results are labeled MetaDrive. Output directories are never silently
reused. MetaDrive has a separate real-dependency gate and run command.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import _bootstrap
from src.agents.trainer import train
from src.utils.config import load_config
from src.utils.logger import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='experiments/release')
    parser.add_argument('--scale', type=float, default=1.0)
    args = parser.parse_args()
    if args.scale <= 0:
        parser.error('scale must be positive')
    matrix = [
        ('ppo_state_seed7', 'ppo_state.yaml', 32768, {'seed': 7, 'env': {'seed': 7}}),
        ('ppo_state_seed8', 'ppo_state.yaml', 32768, {'seed': 8, 'env': {'seed': 8}}),
        ('ppo_state_seed9', 'ppo_state.yaml', 32768, {'seed': 9, 'env': {'seed': 9}}),
        ('mappo_state', 'mappo.yaml', 16384, {}),
        ('ppo_bev', 'ppo_bev.yaml', 16384, {'seed': 7, 'env': {'seed': 7, 'road_length': 180, 'max_steps': 240}}),
        ('mappo_hybrid', 'end_to_end.yaml', 16384, {}),
        ('sac_state', 'sac.yaml', 8192, {}),
        ('reward_safe', 'ppo_state.yaml', 8192, {'env': {'reward': 'safe'}}),
        ('reward_efficient', 'ppo_state.yaml', 8192, {'env': {'reward': 'efficient'}}),
        ('ippo_state', 'mappo.yaml', 8192, {'algorithm': 'ippo'}),
    ]
    results = {}
    root = Path(args.output)
    for name, filename, steps, changes in matrix:
        changes['total_steps'] = max(64, int(steps * args.scale))
        config = load_config(_bootstrap.ROOT / 'configs' / filename, changes)
        results[name] = train(config, root / name)
        save_json(results, root / 'matrix.json')
        print(f"{name}: {results[name]['world_steps_total']} world steps completed", flush=True)
    return results


if __name__ == '__main__':
    main()
