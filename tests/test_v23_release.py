from __future__ import annotations

import hashlib
import json
from pathlib import Path

import src

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def test_release_version_and_curated_metadrive_models_are_consistent():
    registry = json.loads((ROOT / 'models/registry.json').read_text(encoding='utf-8'))
    expected = {
        'metadrive_ppo_state_32768': ('metadrive', 'ppo', 32768),
        'metadrive_mappo_hybrid_8192': ('metadrive', 'mappo', 8192),
        'metadrive_ppo_bev_ego4_4096': ('metadrive', 'ppo', 4096),
        'metadrive_official_sb3_ppo_4096': ('metadrive', 'sb3_ppo', 4096),
    }
    for name, (backend, algorithm, steps) in expected.items():
        row = registry[name]
        path = ROOT / row['path']
        assert path.is_file() and path.stat().st_size > 0
        assert row['backend'] == backend
        assert row['algorithm'] == algorithm
        assert row['world_steps'] == steps
        assert row['sha256'] == _sha256(path)


def test_release_readme_does_not_claim_bev_or_rgb_safety_completion():
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert 'v2.3.0' in readme
    assert 'raw-RGB' in readme
    assert '禁止用于实车控制' in readme
