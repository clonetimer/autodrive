"""Fixed-budget v2/v3 reward ablation plus observation/resume engineering runs.

The matrix is saved BEFORE training. No checkpoint is selected on evaluation.
Recorded worlds are native CPU simulations, never labeled MetaDrive.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import time
import _bootstrap
from src.agents.trainer import train
from src.utils.config import TrainConfig, EnvConfig
from src.utils.logger import save_json, runtime_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='experiments/v2_1')
    parser.add_argument('--steps', type=int, default=32768)
    args = parser.parse_args()
    output = Path(args.output)
    if (output / 'matrix_plan.json').exists():
        raise FileExistsError('A fixed experiment matrix already exists; choose a fresh output')
    base = TrainConfig(total_steps=args.steps, rollout_steps=256, epochs=6, batch_size=128,
        entropy_coef=.003, checkpoint_every=args.steps, threads=1,
        env=EnvConfig(road_length=180, max_steps=240, traffic_vehicles=3,
                      target_speed=12, initial_speed=8, reward_version=3))
    matrix = []
    for version in (2, 3):
        for seed in (7, 8, 9):
            cfg = replace(base, seed=seed, env=replace(base.env, seed=seed, reward_version=version))
            matrix.append((f'ppo_reward_v{version}_seed{seed}', cfg))
    # Engineering verification has a separate budget and is not a fairness comparison.
    for frames in (1, 4):
        cfg = replace(base, total_steps=args.steps // 2, seed=7,
            env=replace(base.env, observation='bev_ego', bev_frames=frames, seed=7))
        matrix.append((f'ppo_bev_ego_frames{frames}', cfg))
    cfg = replace(base, algorithm='mappo', total_steps=args.steps // 4,
        env=replace(base.env, num_agents=2, observation='bev_ego', bev_frames=4, traffic_vehicles=2))
    matrix.append(('mappo_bev_ego_frames4', cfg))
    cfg = replace(base, algorithm='sac', total_steps=2048, rollout_steps=128,
        epochs=1, batch_size=64, learning_starts=128, replay_size=4096, checkpoint_every=2048)
    matrix.append(('sac_first', cfg))
    plan = {'runtime': runtime_manifest(), 'runs': {name: cfg.to_dict() for name, cfg in matrix},
            'resume': {'source': 'sac_first/training_state.pt', 'absolute_total_steps': 4096},
            'evaluation_usage': 'development_validation_only; future frozen holdout untouched'}
    save_json(plan, output / 'matrix_plan.json')
    results = {}
    started = time.perf_counter()
    for name, config in matrix:
        results[name] = train(config, output / name)
        save_json(results, output / 'matrix_results.json')
        print(name, results[name]['world_steps_total'], round(time.perf_counter() - started, 2), flush=True)
    results['sac_resumed'] = train(replace(cfg, total_steps=4096), output / 'sac_resumed',
                                  output / 'sac_first/training_state.pt')
    save_json(results, output / 'matrix_results.json')
    total = sum(result.get('world_steps_this_run', result['world_steps_total']) for result in results.values())
    save_json({'world_steps_executed_this_matrix': total, 'runs': len(results),
               'elapsed_seconds': time.perf_counter() - started, 'status': 'completed'}, output / 'totals.json')


if __name__ == '__main__':
    main()
