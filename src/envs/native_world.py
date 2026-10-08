"""Coupled traffic simulator with simultaneous actions and individual lifetimes.

The same physical world is shared by all agents. This is NOT a vector of
independent single-car environments. Termination removes a car; truncation is
reserved for the time limit. Final observations are always returned once.
"""
from __future__ import annotations
from dataclasses import replace
from typing import Any
import math
import numpy as np
from src.envs.observation import Snapshot, build_observation, build_state, observation_space, state_dimension
from src.scenarios.generator import generate_road
from src.scenarios.traffic import Vehicle, collision, traffic_action
from src.utils.config import EnvConfig
from src.utils.spaces import Box

from src.envs.reward import REWARD_WEIGHTS, reward_terms, traffic_risk
from src.perception.history import BEVHistory



class NativeWorld:
    metadata = {'name': 'native_parallel_drive_v2', 'is_parallelizable': True}

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.possible_agents = [f'agent{i}' for i in range(self.config.num_agents)]
        self.agents: list[str] = []
        self.rng = np.random.default_rng(self.config.seed)
        self.action_space = Box(-1, 1, shape=(2,), dtype=np.float32)
        self.observation_space = observation_space(self.config.observation, self.config.neighbors, self.config.bev_size, self.config.bev_frames)
        self.global_state_dim = self.config.num_agents * (state_dimension(self.config.neighbors) + 1)
        self.steps = 0
        self.vehicles: dict[str, Vehicle] = {}
        self._seed = self.config.seed
        self.closed = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = int(seed)
            self.rng = np.random.default_rng(seed)
        self.closed = False
        self.road = generate_road(self.config, self.rng)
        self.steps = 0
        self.vehicles = {}
        self.agents = self.possible_agents.copy()
        center_lane = self.config.lanes // 2
        for i, name in enumerate(self.possible_agents):
            lane = (center_lane + i) % self.config.lanes
            x = 5.0 + (i // self.config.lanes) * 12
            y = float(self.road.lane_y(x, lane)) + float(self.rng.uniform(-0.15, 0.15))
            speed = self.config.initial_speed * (float(self.rng.uniform(0.85, 1.15)) if self.config.randomize else 1.0)
            vehicle = Vehicle(name, x, y, self.road.heading(x), speed, self.config.target_speed,
                              learning=True, target_lane=lane)
            self.vehicles[name] = vehicle
        if self.config.scenario != 'cruise':
            ego = self.vehicles['agent0']
            leader_speed = 5.0 if self.config.scenario in {'follow', 'overtake'} else 10.0
            self.vehicles['leader'] = Vehicle('leader', ego.x + 32, float(self.road.lane_y(ego.x + 32, center_lane)),
                self.road.heading(ego.x + 32), leader_speed, leader_speed, target_lane=center_lane)
        for i in range(self.config.traffic_vehicles):
            for _ in range(100):
                x = float(self.rng.uniform(38, max(45, self.road.length - 10)))
                lane = int(self.rng.integers(self.config.lanes))
                vehicle = Vehicle(f'traffic{i}', x, float(self.road.lane_y(x, lane)), self.road.heading(x),
                    float(self.rng.uniform(7, 13)), float(self.rng.uniform(8, 16)), target_lane=lane)
                if all(abs(v.x - x) > 13 or abs(v.y - vehicle.y) > 2.5 for v in self.vehicles.values()):
                    self.vehicles[vehicle.id] = vehicle
                    break
        self.returns = {name: 0.0 for name in self.agents}
        self.previous_lane = {name: self.vehicles[name].target_lane for name in self.agents}
        self.lane_changes = {name: 0 for name in self.agents}
        self.histories = {name: BEVHistory(self.config.bev_frames, self.config.bev_size, self.config.dt)
                          for name in self.possible_agents} if self.config.observation != 'state' else {}
        for name in self.agents:
            self._capture_history(name)
        observations = {name: self.observe(name) for name in self.agents}
        infos = {name: self._info(name, False, False, False, 0, 0) for name in self.agents}
        return observations, infos

    def snapshot(self, name: str) -> Snapshot:
        ego = self.vehicles[name]
        lane = self.road.lane_index(ego.x, ego.y)
        # Route is the current lane; longitudinal goal is independent of lane.
        boundaries, route = self.road.polylines(ego.x - 24, ego.x + 90, lane)
        return Snapshot(ego, list(self.vehicles.values()), boundaries, route, self.road.heading(ego.x),
            self.road.offset(ego.x, ego.y), float(ego.y - self.road.center(ego.x)), self.road.lane_width,
            self.road.lanes, lane, float(np.clip(ego.x / self.road.length, 0, 1)))

    def _capture_history(self, name: str):
        if name in self.histories:
            self.histories[name].capture(self.snapshot(name), self.steps * self.config.dt)

    def observe(self, name: str):
        return build_observation(self.snapshot(name), self.config, self.rng, self.histories.get(name))

    def global_state(self) -> np.ndarray:
        state = []
        for name in self.possible_agents:
            valid = name in self.agents
            features = build_state(self.snapshot(name), self.config.neighbors, self.config.state_schema_version) if valid else np.zeros(state_dimension(self.config.neighbors), dtype=np.float32)
            state.extend(features)
            state.append(float(valid))
        return np.asarray(state, dtype=np.float32)

    def step(self, actions: dict[str, np.ndarray]):
        if not self.agents or self.closed:
            raise RuntimeError('reset() is required before step() and after episode completion')
        active = self.agents.copy()
        if set(actions) != set(active):
            raise ValueError(f'Action keys must match active agents: {active}')
        processed = {}
        for name, action in actions.items():
            arr = np.asarray(action, dtype=np.float32)
            if arr.shape != (2,) or not np.isfinite(arr).all():
                raise ValueError('Every action must contain two finite scalars')
            processed[name] = np.clip(arr, -1, 1)
        old_x = {name: self.vehicles[name].x for name in active}
        old_acc = {name: self.vehicles[name].acceleration for name in active}
        old_lat = {name: self.vehicles[name].speed * self.vehicles[name].yaw_rate for name in active}
        old_steer = {name: self.vehicles[name].steer for name in active}
        hit = set()
        self.steps += 1
        if self.config.scenario == 'emergency' and self.steps * self.config.dt >= 2 and 'leader' in self.vehicles:
            self.vehicles['leader'].target_speed = 0.0
        # All policies see the same pre-substep world; integrate simultaneously.
        for substep in range(self.config.substeps):
            sub_dt = self.config.dt / self.config.substeps
            simulation_time = (self.steps - 1) * self.config.dt + (substep + 1) * sub_dt
            self.before_substep(simulation_time)
            cars = [v for v in self.vehicles.values() if v.active]
            controls = {v.id: processed[v.id] if v.learning else traffic_action(v, cars, self.road) for v in cars}
            for v in cars:
                sub_dt = self.config.dt / self.config.substeps
                self.advance_vehicle(v, controls[v.id], sub_dt, (self.steps - 1) * self.config.dt + (substep + 1) * sub_dt)
            for i, first in enumerate(cars):
                for second in cars[i + 1:]:
                    if collision(first, second):
                        hit.update((first.id, second.id))
        observations, rewards, terminated, truncated, infos = {}, {}, {}, {}, {}
        weights = REWARD_WEIGHTS[self.config.reward]
        for name in active:
            car = self.vehicles[name]
            crash = name in hit
            offroad = abs(car.y - self.road.center(car.x)) + car.width / 2 > self.road.lanes * self.road.lane_width / 2
            success = car.x >= self.road.length and not crash and not offroad
            jerk = (car.acceleration - old_acc[name]) / self.config.dt
            progress = car.x - old_x[name]
            snapshot = self.snapshot(name)
            risk = traffic_risk(snapshot)
            terms = reward_terms(snapshot, progress, jerk, self.config.dt, self.config.reward,
                                 self.config.reward_version, collision=crash, offroad=offroad,
                                 success=success, risk=risk)
            reward = float(sum(terms.values()))
            terminated[name] = bool(crash or offroad or success)
            truncated[name] = bool(self.steps >= self.config.max_steps and not terminated[name])
            self.returns[name] += reward
            lane = self.road.lane_index(car.x, car.y)
            if lane != self.previous_lane[name]:
                self.lane_changes[name] += 1
                self.previous_lane[name] = lane
            infos[name] = self._info(name, crash, offroad, success, jerk, progress)
            infos[name]['reward_terms'] = terms
            infos[name]['reward_version'] = self.config.reward_version
            infos[name]['risk'] = risk.to_dict()
            infos[name]['lateral_acceleration'] = car.speed * car.yaw_rate
            infos[name]['lateral_jerk'] = (car.speed * car.yaw_rate - old_lat[name]) / self.config.dt
            infos[name]['steering_rate'] = (car.steer - old_steer[name]) / self.config.dt
            self._capture_history(name)
            infos[name]['episode_return'] = self.returns[name]
            rewards[name] = reward
            observations[name] = self.observe(name)  # final pre-removal observation
        self.transition_state = self.global_state()
        for name in active:
            if terminated[name] or truncated[name]:
                self.vehicles[name].active = False
                self.agents.remove(name)
        return observations, rewards, terminated, truncated, infos

    def before_substep(self, simulation_time):
        """Optional deterministic source hook; native IDM traffic needs no override."""

    def advance_vehicle(self, vehicle, action, dt, simulation_time):
        vehicle.integrate(action, dt)

    def _info(self, name, crash, offroad, success, jerk, progress) -> dict[str, Any]:
        car = self.vehicles[name]
        return {'backend': 'native', 'scenario_seed': self._seed, 'collision': bool(crash),
                'out_of_road': bool(offroad), 'success': bool(success), 'speed': car.speed,
                'acceleration': car.acceleration, 'jerk': float(jerk),
                'lane_offset': self.road.offset(car.x, car.y), 'lane_width': self.road.lane_width,
                'progress': float(progress), 'route_completion': float(np.clip(car.x / self.road.length, 0, 1)),
                'step': self.steps, 'dt': self.config.dt, 'time': self.steps * self.config.dt,
                'lane_changes': self.lane_changes.get(name, 0), 'x': car.x, 'y': car.y, 'yaw': car.yaw}

    def render(self, name: str = 'agent0') -> np.ndarray:
        from src.perception.bev_generator import rasterize, semantic_rgb
        return semantic_rgb(rasterize(self.snapshot(name), 192))

    def close(self) -> None:
        self.closed = True
        self.agents = []
