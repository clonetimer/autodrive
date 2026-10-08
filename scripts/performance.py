"""CPU timings with separate physics step, full loop and policy-call semantics."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
import numpy as np
import torch
import _bootstrap
from src.envs.native_world import NativeWorld
from src.utils.config import EnvConfig
from src.utils.logger import save_json, runtime_manifest


def measure(steps=3000, repeats=3, modes=None, bev_frames=1):
    if min(steps, repeats) < 1:
        raise ValueError('Counts must be positive')
    modes = list(modes or ['state', 'bev', 'hybrid'])
    if len(set(modes)) != len(modes) or any(mode not in {'state', 'bev', 'hybrid', 'bev_ego'} for mode in modes):
        raise ValueError('Modes must be distinct supported observation names')
    if not isinstance(bev_frames, int) or not 1 <= bev_frames <= 8:
        raise ValueError('bev_frames must be an integer from 1 to 8')
    torch.set_num_threads(1)
    runs = []
    for mode in modes:
        for repeat in range(repeats):
            config = EnvConfig(observation=mode, num_agents=2, traffic_vehicles=8,
                               road_length=180, max_steps=240,
                               bev_frames=1 if mode == "state" else bev_frames)
            world = NativeWorld(config)
            observations, _ = world.reset(seed=500 + repeat)
            # Fixed neutral control: this measures simulator work, not driving quality.
            elapsed_step = 0
            resets = 0
            begin = time.perf_counter()
            try:
                for _ in range(steps):
                    actions = {name: np.array([0., 0.3], np.float32) for name in world.agents}
                    start = time.perf_counter()
                    world.step(actions)
                    elapsed_step += time.perf_counter() - start
                    if not world.agents:
                        world.reset()
                        resets += 1
            finally:
                world.close()
            elapsed_loop = time.perf_counter() - begin
            print(f'{mode} repeat={repeat}: completed {steps} joint steps', flush=True)
            runs.append({'mode': mode, 'bev_frames': config.bev_frames, 'reward_version': config.reward_version,
                         'repeat': repeat, 'agents': 2, 'idm_requested': 8,
                         'world_steps': steps, 'resets': resets,
                         'step_call_steps_per_second': steps / elapsed_step,
                         'loop_including_resets_steps_per_second': steps / elapsed_loop})
    return {'runtime': runtime_manifest(),
            'definition': 'Joint world steps, dt=0.1 s; step-call includes physics, collision, reward and observations. Not rendered FPS.',
            'runs': runs,
            'summary': {mode: {'step_call_median_sps': float(np.median([r['step_call_steps_per_second'] for r in runs if r['mode'] == mode])),
                               'loop_median_sps': float(np.median([r['loop_including_resets_steps_per_second'] for r in runs if r['mode'] == mode]))}
                        for mode in modes}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', default='reports/performance.json')
    parser.add_argument('--modes', nargs='+', choices=['state', 'bev', 'hybrid', 'bev_ego'])
    parser.add_argument('--bev-frames', type=int, default=1)
    args = parser.parse_args()
    result = measure(args.steps, args.repeats, args.modes, args.bev_frames)
    save_json(result, args.output)
    print(result['summary'])
