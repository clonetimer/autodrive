"""Policy-weight transfer across reward objectives without pretending to resume training.

Only architecture-compatible encoder/actor parameters are copied. Critic and
optimizer are reinitialized because reward semantics changed.
"""
from __future__ import annotations
from pathlib import Path
import torch
from src.agents.trainer import load_checkpoint, model_from_config, save_checkpoint
from src.utils.config import TrainConfig
from src.utils.logger import sha256


def transfer_actor(source: str | Path, target_config: TrainConfig, output: str | Path,
                   *, include_log_std: bool = True) -> dict:
    source_model, source_config, _ = load_checkpoint(source, target_config.device)
    architecture_source = (source_config.env.observation, source_config.env.neighbors,
                           source_config.hidden_size, source_config.env.bev_frames,
                           source_config.env.num_agents, source_config.algorithm)
    architecture_target = (target_config.env.observation, target_config.env.neighbors,
                           target_config.hidden_size, target_config.env.bev_frames,
                           target_config.env.num_agents, target_config.algorithm)
    if architecture_source != architecture_target:
        raise ValueError(f'Actor transfer requires matching architecture: {architecture_source} != {architecture_target}')
    target = model_from_config(target_config, source_model.central_dim).to(target_config.device)
    source_state = source_model.state_dict()
    target_state = target.state_dict()
    copied = []
    for name, value in source_state.items():
        if name.startswith('encoder.') or name.startswith('actor.') or (include_log_std and name == 'log_std'):
            if name not in target_state or target_state[name].shape != value.shape:
                raise ValueError(f'Incompatible transferred tensor {name}')
            target_state[name] = value.detach().clone()
            copied.append(name)
    target.load_state_dict(target_state)
    optimizer = torch.optim.Adam(target.parameters(), lr=target_config.learning_rate, eps=1e-5)
    extra = {
        'transfer': {
            'source_sha256': sha256(source),
            'source_reward_version': source_config.env.reward_version,
            'source_reward_source': source_config.env.reward_source,
            'target_reward_version': target_config.env.reward_version,
            'target_reward_source': target_config.env.reward_source,
            'copied_tensors': copied,
            'critic_reinitialized': True,
            'optimizer_reinitialized': True,
            'semantics': 'actor/encoder warm start only; not a continuation of source reward training',
        }
    }
    save_checkpoint(output, target, target_config, optimizer, 0, extra=extra)
    return extra['transfer']
