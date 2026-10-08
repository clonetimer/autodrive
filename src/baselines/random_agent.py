import numpy as np


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def predict(self, observation, deterministic=True):
        return self.rng.uniform(-1, 1, 2).astype(np.float32)

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
