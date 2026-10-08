"""Real SB3 PPO CPU reference; no fallback to the project's native optimizer.

State-input/single-learning-agent diagnostic only. Both implementations can use
the same physical environment and reward, but SB3 Gaussian action clipping and
this project's tanh-squashed actions differ: do not claim implementation parity.
"""
from dataclasses import asdict
from functools import partial
import argparse
from pathlib import Path
import time
import _bootstrap
from src.utils.config import load_config, EnvConfig
from src.utils.logger import save_json, sha256, runtime_manifest


def make_gym_environment(env_dict, rank=0):
    from dataclasses import replace
    from src.envs.simple_drive_env import SimpleDriveEnv
    from src.envs.metadrive_wrapper import MetaDriveWrapper
    cfg = EnvConfig(**env_dict)
    cfg = replace(cfg, seed=(cfg.seed + rank) % 1000)
    return MetaDriveWrapper(cfg) if cfg.backend == 'metadrive' else SimpleDriveEnv(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-envs', type=int, choices=[1, 2, 4], default=1)
    parser.add_argument('--episodes', type=int, default=20)
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = None
    try:
        if config.env.observation != 'state' or config.env.num_agents != 1 or config.env.backend not in {'native', 'metadrive'}:
            raise ValueError('The reference is state-only and single-agent, using native or real MetaDrive')
        if (output / 'model.zip').exists():
            raise FileExistsError('Choose a fresh reference output directory')
        try:
            import torch
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
            from stable_baselines3.common.env_checker import check_env
            from stable_baselines3.common.logger import configure
        except ImportError as exc:
            raise RuntimeError('Real SB3/Gymnasium required. Install requirements-validation.txt; no substitute training executed.') from exc
        torch.set_num_threads(config.threads)
        # Check once, close before constructing another real MetaDrive engine.
        checked = make_gym_environment(asdict(config.env))
        try:
            check_env(checked, warn=True)
        finally:
            checked.close()
        factories = [partial(make_gym_environment, asdict(config.env), rank) for rank in range(args.n_envs)]
        env = DummyVecEnv(factories) if args.n_envs == 1 else SubprocVecEnv(factories, start_method='spawn')
        policy = PPO('MlpPolicy', env, device='cpu', seed=config.seed,
            learning_rate=config.learning_rate, n_steps=config.rollout_steps, batch_size=config.batch_size,
            n_epochs=config.epochs, gamma=config.gamma, gae_lambda=config.gae_lambda,
            clip_range=config.clip_range, ent_coef=config.entropy_coef, vf_coef=config.value_coef,
            max_grad_norm=config.max_grad_norm, target_kl=config.target_kl,
            policy_kwargs={'net_arch': {'pi': [config.hidden_size, config.hidden_size],
                                         'vf': [config.hidden_size, config.hidden_size]}})
        policy.set_logger(configure(str(output / 'logs'), ['csv']))
        started = time.perf_counter()
        policy.learn(total_timesteps=config.total_steps)
        policy.save(output / 'model')
        env.close(); env = None
        from src.evaluation.evaluator import evaluate
        class Agent:
            def predict(self, obs):
                return policy.predict(obs, deterministic=True)[0]
        report = evaluate(lambda name, seed: Agent(), config.env,
                          list(range(3000, 3000 + args.episodes)), output / 'evaluation', 'SB3 PPO reference')
        save_json({'status': 'real_training_completed', 'runtime': runtime_manifest(), 'config': config.to_dict(),
                   'requested_steps': config.total_steps, 'actual_sb3_timesteps': policy.num_timesteps,
                   'n_envs': args.n_envs, 'elapsed_seconds': time.perf_counter() - started,
                   'model_sha256': sha256(output / 'model.zip'), 'metrics': report['metrics'],
                   'comparison_scope': 'reference diagnosis; network/action-distribution differences are not controlled'},
                  output / 'result.json')
        return 0
    except Exception as exc:
        save_json({'status': 'blocked_or_failed', 'exception': type(exc).__name__, 'message': str(exc),
                   'requested_backend': config.env.backend, 'config': config.to_dict()}, output / 'failure.json')
        print(str(exc))
        return 2
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    raise SystemExit(main())
