"""Record a fixed-seed actual rollout as GIF and a verifiable JSON trajectory."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import _bootstrap
from src.end_to_end.agent import EndToEndAgent
from src.navigation.learned_route_adapter import LearnedRouteAdapterAgent
from src.multi_agent.marl_env import make_parallel_env
from src.utils.config import load_config
from src.utils.logger import save_json


def record(checkpoint, output, seed=2000, every=2, env_config=None, steering_adapter=None):
    if every < 1:
        raise ValueError('Frame interval must be positive')
    policy = (LearnedRouteAdapterAgent(checkpoint, steering_adapter) if steering_adapter else EndToEndAgent(checkpoint))
    base_config = policy.base.config.env if steering_adapter else policy.config.env
    config = env_config or base_config
    if config.observation != base_config.observation or config.num_agents != base_config.num_agents:
        raise ValueError('Demo override must preserve policy observation type and learning-agent count')
    env = make_parallel_env(config)
    frames, trajectory = [], []
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    interventions = 0
    try:
        observations, _ = env.reset(seed=seed)
        while env.agents:
            if env.steps % every == 0:
                active = 'agent0' if 'agent0' in env.agents else env.agents[0]
                image = Image.fromarray(env.render(active)).resize((384, 384), Image.Resampling.NEAREST)
                canvas = Image.new('RGB', (384, 430))
                canvas.paste(image, (0, 46))
                draw = ImageDraw.Draw(canvas)
                draw.text((10, 8), f'{env.config.backend.upper()} | {env.config.observation} | seed={seed}', fill='white')
                mode = 'supervised-system' if env.config.safety_supervisor else 'raw-policy'
                draw.text((10, 25), f't={env.steps * env.config.dt:.1f}s | {mode} | interventions={interventions}', fill='white')
                frames.append(canvas)
            actions = policy.act(observations)
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            interventions += sum(bool(info.get('safety_intervention', False)) for info in infos.values())
            trajectory.append({'step': env.steps, 'actions': actions, 'infos': infos,
                               'rewards': rewards, 'terminated': terms, 'truncated': truncs})
            observations = {name: next_obs[name] for name in env.agents}
        if not frames:
            raise RuntimeError('No frames captured')
        frames[0].save(output, save_all=True, append_images=frames[1:],
                       duration=round(1000 * env.config.dt * every), loop=0)
        save_json({'backend': env.config.backend, 'seed': seed, 'model_sha256': policy.fingerprint,
                   'steering_adapter': str(steering_adapter) if steering_adapter else None,
                   'trajectory': trajectory}, output.with_suffix('.json'))
        return len(frames)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=2000)
    parser.add_argument('--every', type=int, default=2)
    parser.add_argument('--config', help='optional environment override; must preserve policy observation/agent schema')
    parser.add_argument('--steering-adapter', help='optional learned route-adapter artifact matched to --model')
    args = parser.parse_args()
    env_config = load_config(args.config).env if args.config else None
    print(f'Captured {record(args.model, args.output, args.seed, args.every, env_config, args.steering_adapter)} frames')


if __name__ == '__main__':
    main()
