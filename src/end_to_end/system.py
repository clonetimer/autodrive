"""A real closed-loop deployment runner with atomic trajectory evidence."""
from pathlib import Path
from src.multi_agent.marl_env import make_parallel_env
from src.end_to_end.agent import EndToEndAgent
from src.utils.logger import save_json


def run_episode(checkpoint, env_config=None, seed=2000, output=None):
    agent = EndToEndAgent(checkpoint)
    env = make_parallel_env(env_config or agent.config.env)
    records = []
    try:
        observations, _ = env.reset(seed=seed)
        while env.agents:
            actions = agent.act(observations)
            next_obs, rewards, terms, truncs, infos = env.step(actions)
            records.append({'step': env.steps, 'actions': actions, 'rewards': rewards,
                            'terminated': terms, 'truncated': truncs, 'infos': infos})
            observations = {name: next_obs[name] for name in env.agents}
        result = {'checkpoint_sha256': agent.fingerprint, 'seed': seed,
                  'backend': env.config.backend, 'scope': 'semantic_observation_to_control', 'trajectory': records}
        if output:
            save_json(result, output)
        return result
    finally:
        env.close()
