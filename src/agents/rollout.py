"""GAE preserves termination vs time-limit bootstrap and masks inactive agents."""
from __future__ import annotations
import numpy as np


def generalized_advantage(rewards, values, next_values, terminated, truncated, valid,
                          gamma: float = 0.99, gae_lambda: float = 0.95):
    arrays = [np.asarray(x) for x in (rewards, values, next_values, terminated, truncated, valid)]
    shape = arrays[0].shape
    if len(shape) != 2 or any(x.shape != shape for x in arrays):
        raise ValueError('GAE inputs must all have shape (time, agent)')
    rewards, values, next_values, terminated, truncated, valid = arrays
    advantage = np.zeros(shape, dtype=np.float32)
    carry = np.zeros(shape[1], dtype=np.float32)
    for t in range(shape[0] - 1, -1, -1):
        bootstrap = (~terminated[t].astype(bool)).astype(np.float32)
        continuation = (~(terminated[t].astype(bool) | truncated[t].astype(bool))).astype(np.float32)
        delta = rewards[t] + gamma * next_values[t] * bootstrap - values[t]
        carry = (delta + gamma * gae_lambda * continuation * carry) * valid[t]
        advantage[t] = carry
    returns = advantage + values
    return advantage, returns
