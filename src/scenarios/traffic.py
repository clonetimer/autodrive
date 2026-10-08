"""Kinematic bicycle vehicles, IDM following and oriented-box collision tests.

This is a deliberately simplified *native* simulator, not MetaDrive physics.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from src.scenarios.generator import Road


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Vehicle:
    id: str
    x: float
    y: float
    yaw: float
    speed: float
    target_speed: float = 15.0
    length: float = 4.5
    width: float = 1.8
    wheelbase: float = 2.7
    acceleration: float = 0.0
    steer: float = 0.0
    learning: bool = False
    active: bool = True
    target_lane: int = 1
    yaw_rate: float = 0.0

    def corners(self) -> np.ndarray:
        local = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]], dtype=float) * [self.length / 2, self.width / 2]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return local @ np.array([[c, s], [-s, c]]) + [self.x, self.y]

    def integrate(self, action, dt: float) -> None:
        steer, pedal = np.asarray(action, dtype=float)
        self.steer = float(np.clip(steer, -1, 1)) * 0.45
        requested_acc = float(pedal) * (3.0 if pedal >= 0 else 6.0)
        # First-order actuator lag; longitudinal drag prevents endless acceleration.
        self.acceleration += min(1.0, dt / 0.25) * (requested_acc - 0.015 * self.speed - self.acceleration)
        new_speed = float(np.clip(self.speed + self.acceleration * dt, 0.0, 35.0))
        mean_speed = (new_speed + self.speed) / 2
        yaw_rate = mean_speed / self.wheelbase * math.tan(self.steer)
        self.yaw_rate = yaw_rate
        middle_yaw = self.yaw + yaw_rate * dt / 2
        self.x += mean_speed * math.cos(middle_yaw) * dt
        self.y += mean_speed * math.sin(middle_yaw) * dt
        self.yaw = wrap_angle(self.yaw + yaw_rate * dt)
        self.speed = new_speed


def collision(a: Vehicle, b: Vehicle) -> bool:
    if abs(a.x - b.x) > (a.length + b.length) / 2 + 1 or abs(a.y - b.y) > (a.length + b.length) / 2 + 1:
        return False
    p, q = a.corners(), b.corners()
    for polygon in (p, q):
        for i in (0, 1):
            edge = polygon[i + 1] - polygon[i]
            axis = np.array([-edge[1], edge[0]])
            ap, bp = p @ axis, q @ axis
            if ap.max() < bp.min() or bp.max() < ap.min():
                return False
    return True


def idm_acceleration(speed: float, target: float, gap: float = float('inf'), closing: float = 0.0,
                     headway: float = 1.5, max_acc: float = 2.0, comfortable_brake: float = 2.5) -> float:
    if target <= 0:
        return -6.0 if speed > 0 else 0.0
    desired_gap = 2.0 + max(0.0, speed * headway + speed * closing / (2 * math.sqrt(max_acc * comfortable_brake)))
    interaction = 0.0 if math.isinf(gap) else (desired_gap / max(gap, 0.1)) ** 2
    return float(np.clip(max_acc * (1 - (speed / max(target, 0.1)) ** 4 - interaction), -6.0, 3.0))


def lane_control(vehicle: Vehicle, road: Road, lane: int, acceleration: float) -> np.ndarray:
    lookahead = max(7.0, vehicle.speed * 0.8)
    target_x = vehicle.x + lookahead
    target_y = float(road.lane_y(target_x, lane))
    alpha = wrap_angle(math.atan2(target_y - vehicle.y, target_x - vehicle.x) - vehicle.yaw)
    steer_angle = math.atan2(2 * vehicle.wheelbase * math.sin(alpha), lookahead)
    pedal = acceleration / (3.0 if acceleration >= 0 else 6.0)
    return np.clip([steer_angle / 0.45, pedal], -1, 1).astype(np.float32)


def traffic_action(vehicle: Vehicle, vehicles: list[Vehicle], road: Road) -> np.ndarray:
    lane = road.lane_index(vehicle.x, vehicle.y)
    front = [v for v in vehicles if v.id != vehicle.id and v.active and v.x > vehicle.x
             and road.lane_index(v.x, v.y) == lane]
    leader = min(front, key=lambda v: v.x) if front else None
    gap = leader.x - vehicle.x - (vehicle.length + leader.length) / 2 if leader else float('inf')
    closing = vehicle.speed - leader.speed if leader else 0
    acceleration = idm_acceleration(vehicle.speed, vehicle.target_speed, gap, closing)
    return lane_control(vehicle, road, vehicle.target_lane, acceleration)
