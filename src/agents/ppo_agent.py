"""Public actor interface shared with the end-to-end deployment loop."""
from src.agents.trainer import load_checkpoint


class PPOAgent:
    def __init__(self, checkpoint, device='cpu'):
        self.policy, self.config, self.metadata = load_checkpoint(checkpoint, device)

    def predict(self, observation, deterministic=True):
        return self.policy.predict(observation, deterministic)

    def reset(self):
        pass
