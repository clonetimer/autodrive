from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest
from src.utils.config import EnvConfig
from src.envs.native_world import NativeWorld


@pytest.fixture
def world():
    env = NativeWorld(EnvConfig(traffic_vehicles=0, max_steps=80, road_length=70))
    env.reset(seed=42)
    yield env
    env.close()
