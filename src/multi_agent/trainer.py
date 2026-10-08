"""Explicit centralized-training/decentralized-execution entry point."""
from src.agents.trainer import PPOTrainer


class MAPPOTrainer(PPOTrainer):
    def __init__(self, config, output, resume=None):
        if config.algorithm != 'mappo' or config.env.num_agents < 2:
            raise ValueError('MAPPOTrainer requires algorithm=mappo and at least two learning vehicles')
        super().__init__(config, output, resume)
