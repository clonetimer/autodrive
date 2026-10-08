"""Real MetaDrive + real SB3 on CPU, with OFFICIAL observations and rewards.

Single Panda3D engine per process: training engine closes before cold-model
validation creates another. No native-backend fallback or pretrained imitation.
This engineering reference is separate from the project's custom observation.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import _bootstrap
from src.envs.metadrive_contract import reference_plan
from src.envs.metadrive_wrapper import prerequisite_status, MetaDriveUnavailable
from src.utils.logger import save_json, runtime_manifest, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=32768)
    p.add_argument('--train-start', type=int, default=0)
    p.add_argument('--train-count', type=int, default=100)
    p.add_argument('--eval-start', type=int, default=1000)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--horizon', type=int, default=400)
    p.add_argument('--map', default='S')
    p.add_argument('--traffic-density', type=float, default=.1)
    p.add_argument('--plan-only', action='store_true')
    args = p.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'started', 'runtime': runtime_manifest(),
              'real_steps_executed': 0, 'training_steps': 0, 'evaluation_steps': 0}
    env, model = None, None
    start = time.perf_counter()
    try:
        report['plan'] = reference_plan(args.steps, args.train_start, args.train_count,
            args.eval_start, args.episodes, args.horizon, args.map, args.traffic_density)
        save_json(report['plan'], output / 'plan.json')
        if args.plan_only:
            report['status'] = 'plan_only_not_training'
            return 0
        if (output / 'model.zip').exists():
            raise FileExistsError('Choose a fresh experiment directory; existing model is protected')
        status = prerequisite_status()
        report['prerequisites'] = status
        if not status['ready']:
            raise MetaDriveUnavailable(f'Real runtime prerequisites unmet: {status}. No alternate backend used.')
        import numpy as np
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.logger import configure
        from gymnasium.utils.env_checker import check_env
        import gymnasium as gym
        from src.envs.metadrive_official import OfficialMetaDriveGymEnv
        torch.set_num_threads(1)
        phase = 'checking'
        class CountSteps(gym.Wrapper):
            def step(self, action):
                result = self.env.step(action)
                report['real_steps_executed'] += 1
                report[phase + '_steps'] = report.get(phase + '_steps', 0) + 1
                return result
        env = OfficialMetaDriveGymEnv(report['plan']['train_config'])
        check_env(env, skip_render_check=True, skip_close_check=True)
        env.close(); env = None
        phase = 'training'
        env = Monitor(CountSteps(OfficialMetaDriveGymEnv(report['plan']['train_config'])), str(output / 'monitor.csv'))
        model = PPO('MlpPolicy', env, device='cpu', seed=args.seed, n_steps=512,
                    batch_size=128, n_epochs=5, learning_rate=3e-4,
                    policy_kwargs={'net_arch': {'pi': [64, 64], 'vf': [64, 64]}}, verbose=0)
        model.set_logger(configure(str(output / 'logs'), ['csv', 'tensorboard']))
        before = {n: v.detach().cpu().clone() for n, v in model.policy.state_dict().items()}
        model.learn(total_timesteps=args.steps, callback=CheckpointCallback(
            save_freq=4096, save_path=str(output / 'checkpoints'), name_prefix='ppo'))
        model.save(output / 'model')
        report['actual_sb3_timesteps'] = model.num_timesteps
        report['parameter_delta_l1'] = sum(float(torch.sum(torch.abs(v.cpu() - before[n])))
            for n, v in model.policy.state_dict().items())
        env.close(); env = None
        # Cold-load means the artifact, rather than the live training object, is evaluated.
        frozen = PPO.load(output / 'model.zip', device='cpu')
        phase = 'evaluation'
        env = CountSteps(OfficialMetaDriveGymEnv(report['plan']['evaluation_config']))
        episodes = []
        for seed in report['plan']['evaluation_seeds']:
            observation, _ = env.reset(seed=seed)
            reward_sum, collision, offroad, speeds = 0., False, False, []
            last_info, terminated, truncated = {}, False, False
            for i in range(args.horizon):
                action, _ = frozen.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
                collision |= any(bool(info.get(k, False)) for k in ('crash_vehicle', 'crash_object', 'crash_building', 'crash_human'))
                offroad |= bool(info.get('out_of_road', False))
                speeds.append(float(env.unwrapped.raw_env.vehicle.speed))
                last_info = info
                if terminated or truncated:
                    break
            episodes.append({'seed': seed, 'steps': i + 1, 'return': reward_sum,
                'collision': collision, 'out_of_road': offroad,
                'success': bool(last_info.get('arrive_dest', False)) and not collision and not offroad,
                'terminated': bool(terminated), 'truncated': bool(truncated),
                'mean_speed_mps': float(np.mean(speeds))})
            save_json(episodes, output / 'episodes.json')
        report.update(status='real_training_and_cold_evaluation_completed',
            model_sha256=sha256(output / 'model.zip'),
            metrics={key + '_rate': float(np.mean([r[key] for r in episodes]))
                     for key in ('success', 'collision', 'out_of_road')})
        return 0
    except Exception as exc:
        report.update(status='blocked_or_failed', exception=type(exc).__name__, message=str(exc))
        return 2
    finally:
        if env is not None:
            env.close()
        report['elapsed_seconds'] = time.perf_counter() - start
        save_json(report, output / 'result.json')
        print(report['status'], output / 'result.json')


if __name__ == '__main__':
    raise SystemExit(main())
