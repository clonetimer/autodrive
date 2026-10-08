"""Closed-loop learning ego with NON-reactive recorded background trajectories."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from src.datasets.scenario import load_scenario
from src.envs.native_world import NativeWorld
from src.scenarios.traffic import Vehicle, wrap_angle


class ReplayWorld(NativeWorld):
    def __init__(self, config):
        if not config.dataset:
            raise ValueError('Replay backend requires dataset=portable_scenario.json')
        self.data = load_scenario(config.dataset)
        self.tracks = {str(track['id']): np.asarray(track['samples'], dtype=float) for track in self.data['tracks']}
        super().__init__(replace(config, traffic_vehicles=0))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        for track in self.data['tracks']:
            time, x, y, yaw, speed = self.tracks[str(track['id'])][0]
            name = str(track['id'])
            self.vehicles[name] = Vehicle(name, x, y, yaw, speed, speed,
                length=float(track.get('length', 4.5)), width=float(track.get('width', 1.8)),
                target_lane=self.road.lane_index(x, y), active=(time <= 0 <= self.tracks[name][-1, 0]))
        return ({name: self.observe(name) for name in self.agents},
                {name: dict(self._info(name, False, False, False, 0, 0), backend='replay') for name in self.agents})

    def before_substep(self, simulation_time):
        for name, samples in self.tracks.items():
            self.vehicles[name].active = bool(samples[0, 0] <= simulation_time <= samples[-1, 0])

    def advance_vehicle(self, vehicle, action, dt, simulation_time):
        if vehicle.id not in self.tracks:
            return super().advance_vehicle(vehicle, action, dt, simulation_time)
        samples = self.tracks[vehicle.id]
        old_speed = vehicle.speed
        if simulation_time > samples[-1, 0] or simulation_time < samples[0, 0]:
            vehicle.active = False
            return
        vehicle.x = float(np.interp(simulation_time, samples[:, 0], samples[:, 1]))
        vehicle.y = float(np.interp(simulation_time, samples[:, 0], samples[:, 2]))
        vehicle.yaw = wrap_angle(float(np.interp(simulation_time, samples[:, 0], np.unwrap(samples[:, 3]))))
        vehicle.speed = float(np.interp(simulation_time, samples[:, 0], samples[:, 4]))
        vehicle.acceleration = (vehicle.speed - old_speed) / dt

    def _info(self, *args, **kwargs):
        return dict(super()._info(*args, **kwargs), backend='replay')
