"""Simulator-independent local observation contract (schema versions 3/4).

State: 12 ego/road scalars, four local route points (8), N neighbors (7 each).
Schema v3 keeps historical x-matched route samples for checkpoint compatibility.
Schema v4 samples the ordered route by forward arc length, which remains well
defined on curves and near merge transitions where ego-frame x is not monotonic.
Neighbor features are ego-frame position/velocity, relative heading and mask.
Absolute world position is NOT fed into the actor. BEV is privileged semantic
rasterization, not a camera perception model.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from src.scenarios.traffic import Vehicle, wrap_angle
from src.utils.spaces import Box, Dict


@dataclass
class Snapshot:
    ego: Vehicle
    vehicles: list[Vehicle]
    boundaries: list[np.ndarray]
    route: np.ndarray
    road_heading: float
    lane_offset: float
    road_offset: float
    lane_width: float
    num_lanes: int
    lane_index: int
    completion: float


def to_local(points: np.ndarray, ego: Vehicle) -> np.ndarray:
    delta = np.asarray(points, dtype=np.float64) - [ego.x, ego.y]
    c, s = math.cos(ego.yaw), math.sin(ego.yaw)
    return delta @ np.array([[c, -s], [s, c]])


def state_dimension(neighbors: int = 6) -> int:
    return 20 + 7 * neighbors


def _arc_length_route_preview(snapshot: Snapshot, distances=(6.0, 12.0, 24.0, 42.0)) -> np.ndarray:
    """Interpolate ordered route points at forward arc-length distances.

    The legacy contract searched for a point whose *ego-frame x* matched each
    lookahead.  That is ambiguous on tight curves and can jump across branch
    geometry at merges.  v4 instead starts from the closest ordered route
    sample and walks forward by polyline arc length.  The result remains four
    ego-frame (x, y) points so the policy input dimensionality is unchanged.
    """
    route = np.asarray(snapshot.route, dtype=np.float64)
    if route.ndim != 2 or route.shape[1] != 2 or len(route) < 2 or not np.isfinite(route).all():
        raise ValueError('Snapshot route must contain at least two finite XY points')
    ego_xy = np.array([snapshot.ego.x, snapshot.ego.y], dtype=np.float64)
    nearest = int(np.argmin(np.linalg.norm(route - ego_xy, axis=1)))
    forward = route[nearest:]
    if len(forward) < 2:
        forward = route[-2:]
    segments = np.linalg.norm(np.diff(forward, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    # Remove repeated points before interpolation.  MetaDrive can duplicate a
    # lane-connection endpoint when current and next reference lanes are joined.
    keep = np.concatenate(([True], np.diff(cumulative) > 1e-7))
    forward = forward[keep]
    cumulative = cumulative[keep]
    if len(forward) < 2 or cumulative[-1] <= 1e-7:
        return np.repeat(to_local(forward[:1], snapshot.ego), len(tuple(distances)), axis=0)
    result = []
    for distance in distances:
        d = float(np.clip(distance, 0.0, cumulative[-1]))
        j = int(np.searchsorted(cumulative, d, side='right'))
        if j <= 0:
            point = forward[0]
        elif j >= len(forward):
            point = forward[-1]
        else:
            lo, hi = cumulative[j - 1], cumulative[j]
            alpha = 0.0 if hi <= lo else (d - lo) / (hi - lo)
            point = forward[j - 1] * (1.0 - alpha) + forward[j] * alpha
        result.append(point)
    return to_local(np.asarray(result), snapshot.ego)


def route_preview(snapshot: Snapshot, schema_version: int = 3) -> np.ndarray:
    local_route = to_local(snapshot.route, snapshot.ego)
    if schema_version == 3:
        return np.asarray([local_route[int(np.argmin(abs(local_route[:, 0] - d)))]
                           for d in (8, 16, 30, 50)], dtype=np.float64)
    if schema_version == 4:
        return _arc_length_route_preview(snapshot)
    raise ValueError(f'Unsupported state schema version {schema_version}')


def build_state(snapshot: Snapshot, neighbors: int = 6, schema_version: int = 3) -> np.ndarray:
    ego = snapshot.ego
    error = wrap_angle(ego.yaw - snapshot.road_heading)
    state = np.zeros(state_dimension(neighbors), dtype=np.float32)
    state[:12] = [ego.speed / 30, math.sin(error), math.cos(error),
                  snapshot.lane_offset / snapshot.lane_width, ego.acceleration / 6,
                  ego.target_speed / 30, snapshot.road_offset / (snapshot.num_lanes * snapshot.lane_width / 2),
                  snapshot.num_lanes / 4, snapshot.lane_width / 5, 1 - snapshot.completion,
                  ego.steer / 0.45, snapshot.lane_index / max(1, snapshot.num_lanes - 1)]
    preview = route_preview(snapshot, schema_version)
    for i, point in enumerate(preview):
        state[12 + 2 * i:14 + 2 * i] = point / [60, 20]
    nearby = [v for v in snapshot.vehicles if v.id != ego.id and v.active]
    nearby.sort(key=lambda v: ((v.x - ego.x) ** 2 + (v.y - ego.y) ** 2, v.id))
    c, s = math.cos(ego.yaw), math.sin(ego.yaw)
    ego_velocity = ego.speed * np.array([c, s])
    for i, vehicle in enumerate(nearby[:neighbors]):
        dx, dy = to_local(np.array([[vehicle.x, vehicle.y]]), ego)[0]
        velocity = vehicle.speed * np.array([math.cos(vehicle.yaw), math.sin(vehicle.yaw)]) - ego_velocity
        dvx, dvy = velocity @ np.array([[c, -s], [s, c]])
        angle = wrap_angle(vehicle.yaw - ego.yaw)
        state[20 + i * 7:27 + i * 7] = [dx / 60, dy / 20, dvx / 30, dvy / 30, math.sin(angle), math.cos(angle), 1]
    return np.clip(state, -5, 5).astype(np.float32)


def build_ego_motion(snapshot: Snapshot, schema_version: int = 3) -> np.ndarray:
    """20 normalized ego/navigation scalars; no privileged neighbor velocities.

    speed, acceleration, yaw rate, steer, target speed, heading sin/cos,
    lane/road offset, lane width, remaining route fraction, lane count,
    followed by four local navigation points (x,y). Yaw rate is in rad/s.
    """
    ego = snapshot.ego
    legacy = build_state(snapshot, 0, schema_version)
    motion = np.array([ego.speed / 30, ego.acceleration / 6, ego.yaw_rate / 2,
                       ego.steer / 0.45, ego.target_speed / 30, legacy[1], legacy[2],
                       legacy[3], legacy[6], legacy[8], legacy[9], legacy[7],
                       *legacy[12:20]], dtype=np.float32)
    return np.clip(motion, -5, 5)


def observation_space(mode: str, neighbors: int, size: int, frames: int = 1, rgb_width: int = 96, rgb_height: int = 48):
    state = Box(-5, 5, shape=(state_dimension(neighbors),), dtype=np.float32)
    image = Box(0, 255, shape=(5 * frames, size, size), dtype=np.uint8)
    if mode == 'state':
        return state
    if mode == 'rgb_ego':
        return Dict({'rgb': Box(0, 255, shape=(3, rgb_height, rgb_width), dtype=np.uint8),
                     'motion': Box(-5, 5, shape=(20,), dtype=np.float32)})
    spaces = {'bev': image}
    if mode == 'hybrid':
        spaces['state'] = state
    elif mode == 'bev_ego':
        spaces['motion'] = Box(-5, 5, shape=(20,), dtype=np.float32)
    elif mode != 'bev':
        raise ValueError(mode)
    if frames > 1:
        spaces['history'] = Box(0, 1, shape=(2 * frames,), dtype=np.float32)
    return Dict(spaces)


def build_observation(snapshot: Snapshot, config, rng: np.random.Generator, history=None):
    state = build_state(snapshot, config.neighbors, config.state_schema_version)
    if config.observation_noise:
        noise = rng.normal(0, config.observation_noise, state.shape).astype(np.float32)
        noise[26::7] = 0
        for i in range(config.neighbors):
            if state[26 + i * 7] == 0:
                noise[20 + i * 7:27 + i * 7] = 0
        state = np.clip(state + noise, -5, 5)
    if config.observation == 'state':
        return state
    if config.observation == 'rgb_ego':
        raise ValueError('rgb_ego is supplied by MetaDrive camera pixels, not semantic snapshot rasterization')
    # Standalone callers obtain current-only history, with valid padding masks.
    if history is None:
        from src.perception.history import BEVHistory
        history = BEVHistory(config.bev_frames, config.bev_size, config.dt)
        history.capture(snapshot, 0.0)
    image, validity = history.observation(snapshot)
    result = {'bev': image}
    if config.observation == 'hybrid':
        result['state'] = state
    elif config.observation == 'bev_ego':
        motion = build_ego_motion(snapshot, config.state_schema_version)
        if config.observation_noise:
            motion[:3] = np.clip(motion[:3] + rng.normal(0, config.observation_noise, 3), -5, 5)
        result['motion'] = motion
    if config.bev_frames > 1:
        result['history'] = validity
    return result
