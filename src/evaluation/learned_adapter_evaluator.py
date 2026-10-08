"""Paired held-out scenarios; all outcomes are retained, including failures."""
from __future__ import annotations
from dataclasses import replace, asdict
from collections import deque
import math
from pathlib import Path
import time
import torch
from src.evaluation.learned_adapter_metrics import EpisodeMetrics, summarize
from src.multi_agent.marl_env import make_parallel_env
from src.utils.logger import save_json


def evaluate(agent_factory, env_config, seeds: list[int], output=None, label='agent', save_trajectories=False, failure_tail_seconds=4.0):
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('Evaluation requires non-empty unique seeds')
    if not math.isfinite(failure_tail_seconds) or failure_tail_seconds < 0:
        raise ValueError('failure_tail_seconds must be finite and nonnegative')
    torch.set_num_threads(1)
    env = make_parallel_env(env_config)
    episodes, latencies, trajectories, failures = [], [], [], []
    start = time.perf_counter()
    total_world_steps = 0
    try:
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            agents = {name: agent_factory(name, seed) for name in env.possible_agents}
            for agent in agents.values():
                if hasattr(agent, 'reset'):
                    agent.reset()
            # Warm up neural inference, then reset again. Feed-forward adapters
            # are unaffected; temporal adapters must not inherit three copies of
            # the reset frame as fabricated episode history.
            for name, agent in agents.items():
                if hasattr(agent, 'policy'):
                    for _ in range(3):
                        agent.predict(obs[name])
            for agent in agents.values():
                if hasattr(agent, 'reset'):
                    agent.reset()
            accumulators = {name: EpisodeMetrics(seed, name) for name in env.possible_agents}
            tails = {name: deque() for name in env.possible_agents}
            while env.agents:
                actions, timing = {}, {}
                for name in env.agents:
                    before = time.perf_counter_ns()
                    actions[name] = agents[name].predict(obs[name])
                    timing[name] = (time.perf_counter_ns() - before) / 1e6
                next_obs, rewards, terms, truncs, infos = env.step(actions)
                total_world_steps += 1
                for name in actions:
                    diagnostics = getattr(agents[name], 'last_diagnostics', None)
                    if diagnostics:
                        overlap = set(diagnostics).intersection(infos[name])
                        if overlap:
                            raise KeyError(f'Agent diagnostics collide with environment info keys: {sorted(overlap)}')
                        infos[name].update(diagnostics)
                    accumulators[name].update(rewards[name], infos[name], terms[name], truncs[name], timing[name])
                    latencies.append(timing[name])
                    if save_trajectories:
                        trajectories.append({'seed': seed, 'agent': name, 'action': actions[name],
                                             'reward': rewards[name], 'terminated': terms[name],
                                             'truncated': truncs[name], **infos[name]})
                    record = {'action': actions[name], 'reward': rewards[name], 'terminated': terms[name],
                              'truncated': truncs[name], **infos[name]}
                    if failure_tail_seconds:
                        tails[name].append(record)
                        now = float(infos[name].get('time', accumulators[name].duration_s))
                        while len(tails[name]) > 1 and tails[name][0].get('time', now) < now - failure_tail_seconds:
                            tails[name].popleft()
                    if terms[name] or truncs[name]:
                        episode = accumulators[name].result()
                        episodes.append(episode)
                        if not episode['success']:
                            failures.append({'seed': seed, 'agent': name, 'backend': env_config.backend,
                                'model_sha256': getattr(agents[name], 'fingerprint', None),
                                'collision': episode['collision'], 'out_of_road': episode['out_of_road'],
                                'truncated': episode['truncated'], 'tail_seconds_requested': failure_tail_seconds,
                                'trace': list(tails[name])})
                obs = {name: next_obs[name] for name in env.agents}
        elapsed = time.perf_counter() - start
        report = {'label': label, 'backend': env_config.backend, 'observation': env_config.observation,
                  'env_config': asdict(env_config), 'environment_class': type(env).__module__ + '.' + type(env).__name__,
                  'failure_count': len(failures), 'seeds': seeds, 'metrics': summarize(episodes, latencies), 'episodes': episodes,
                  'world_steps': total_world_steps, 'elapsed_seconds': elapsed,
                  'closed_loop_world_steps_per_second': total_world_steps / max(elapsed, 1e-9)}
        if output:
            from src.evaluation.report import export_report
            export_report(report, output)
            save_json(failures, Path(output) / 'failure_traces.json')
            if save_trajectories:
                save_json(trajectories, Path(output) / 'trajectories.json')
            # Commit marker is deliberately written LAST, after every report,
            # failure trace and optional trajectory has reached disk. Fresh-process
            # MetaDrive evaluators may hang later in Panda3D teardown; supervisors
            # can safely terminate the worker once this marker is valid.
            save_json({'status': 'complete', 'world_steps': total_world_steps,
                       'episodes': len(episodes)}, Path(output) / 'evaluation.complete.json')
        return report
    finally:
        env.close()
