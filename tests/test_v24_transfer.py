from pathlib import Path
import torch
from src.agents.trainer import model_from_config, save_checkpoint, load_checkpoint
from src.agents.transfer import transfer_actor
from src.utils.config import TrainConfig, EnvConfig


def test_actor_transfer_copies_actor_but_reinitializes_critic(tmp_path: Path):
    source_cfg = TrainConfig(total_steps=10, env=EnvConfig())
    source = model_from_config(source_cfg)
    with torch.no_grad():
        for p in source.encoder.parameters():
            p.add_(0.5)
        source.actor.weight.add_(0.25)
        source.critic.weight.add_(3.0)
    source_file = tmp_path / 'source.pt'
    save_checkpoint(source_file, source, source_cfg, torch.optim.Adam(source.parameters()), 10)
    target_cfg = TrainConfig(total_steps=10, env=EnvConfig(reward_version=4))
    target_file = tmp_path / 'target.pt'
    evidence = transfer_actor(source_file, target_cfg, target_file)
    target, _, payload = load_checkpoint(target_file)
    assert torch.equal(target.actor.weight, source.actor.weight)
    assert all(torch.equal(a, b) for a, b in zip(target.encoder.parameters(), source.encoder.parameters()))
    assert not torch.equal(target.critic.weight, source.critic.weight)
    assert payload['steps'] == 0
    assert evidence['critic_reinitialized'] and evidence['optimizer_reinitialized']


def test_release_holdout_is_disjoint_from_all_development_splits():
    from src.scenarios.generator import SPLITS, split_seeds
    release = set(split_seeds('release', 200))
    for name in ('train', 'validation', 'test', 'final'):
        lo, hi = SPLITS[name]
        assert release.isdisjoint(range(lo, hi + 1))


def test_second_release_holdout_is_disjoint_and_reserved():
    from src.scenarios.generator import SPLITS, split_seeds
    release2 = set(split_seeds('release2', 200))
    for name in ('train', 'validation', 'test', 'final', 'release'):
        lo, hi = SPLITS[name]
        assert release2.isdisjoint(range(lo, hi + 1))


def test_third_release_holdout_is_disjoint_and_reserved():
    from src.scenarios.generator import SPLITS, split_seeds
    release3 = set(split_seeds('release3', 200))
    for name in ('train', 'validation', 'test', 'final', 'release', 'release2'):
        lo, hi = SPLITS[name]
        assert release3.isdisjoint(range(lo, hi + 1))
