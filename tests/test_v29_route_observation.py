import math
import numpy as np
import pytest

from src.envs.observation import Snapshot, build_state, route_preview
from src.scenarios.traffic import Vehicle
from src.utils.config import EnvConfig


def _snapshot(route, *, x=0.0, y=0.0, yaw=0.0):
    ego = Vehicle('ego', x, y, yaw, 10.0, 10.0, 4.5, 1.9)
    return Snapshot(
        ego=ego, vehicles=[ego], boundaries=[], route=np.asarray(route, dtype=np.float64),
        road_heading=0.0, lane_offset=0.0, road_offset=0.0, lane_width=3.6,
        num_lanes=2, lane_index=0, completion=0.1,
    )


def test_state_schema_version_validation():
    assert EnvConfig(state_schema_version=3).state_schema_version == 3
    assert EnvConfig(state_schema_version=4).state_schema_version == 4
    with pytest.raises(ValueError, match='state_schema_version'):
        EnvConfig(state_schema_version=5)


def test_v4_arc_preview_is_monotonic_along_curve():
    theta = np.linspace(0.0, math.pi / 2, 80)
    radius = 25.0
    route = np.column_stack((radius * np.sin(theta), radius * (1 - np.cos(theta))))
    snapshot = _snapshot(route)
    preview = route_preview(snapshot, 4)
    # The four targets progress forward along the ordered route and bend left.
    assert preview.shape == (4, 2)
    assert np.all(np.diff(np.linalg.norm(preview, axis=1)) > 1.0)
    assert np.all(np.diff(preview[:, 1]) > 0.0)
    state = build_state(snapshot, neighbors=1, schema_version=4)
    assert state.shape == (27,)
    np.testing.assert_allclose(state[12:20].reshape(4, 2) * [60, 20], preview, atol=1e-5)


def test_v4_preview_uses_route_order_not_matching_local_x():
    # Hairpin-like polyline: the same ego-frame x occurs on different route legs.
    route = np.array([[0, 0], [10, 0], [20, 0], [20, 10], [10, 10], [0, 10]], dtype=float)
    snapshot = _snapshot(route)
    legacy = route_preview(snapshot, 3)
    v4 = route_preview(snapshot, 4)
    # Legacy x matching can jump to the earliest/nearest x match; v4 must advance
    # along the ordered route, eventually reaching the upper return leg.
    assert v4[-1, 1] > 5.0
    assert not np.allclose(legacy, v4)


def test_v3_state_contract_remains_unchanged_by_default():
    route = np.column_stack((np.linspace(-20, 80, 45), np.zeros(45)))
    snapshot = _snapshot(route)
    implicit = build_state(snapshot, neighbors=2)
    explicit = build_state(snapshot, neighbors=2, schema_version=3)
    np.testing.assert_array_equal(implicit, explicit)


def test_distillation_allows_lateral_priority_weight_above_one():
    import torch
    from src.agents.policy import DrivingPolicy
    from src.safety.distillation import distill_safety_actor
    model = DrivingPolicy('state', neighbors=1, hidden_size=16)
    states = np.zeros((8, 27), dtype=np.float32)
    targets = np.tile(np.array([0.5, 0.1], dtype=np.float32), (8, 1))
    result = distill_safety_actor(model, states, targets, np.ones(8, bool), np.ones(8, np.float32),
                                  epochs=1, batch_size=4, steering_weight=4.0, seed=2)
    assert result.steering_weight == 4.0
