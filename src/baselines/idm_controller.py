"""IDM longitudinal baseline plus explicit gap-checked lane selection.

This is NOT claimed to be the full MOBIL model. Lane change has hysteresis and
requires front/rear clearance in the adjacent lane. It uses only the shared
state observation (no direct world access).
"""
import math
import numpy as np
from src.scenarios.traffic import idm_acceleration


class IDMController:
    def __init__(self, allow_lane_change: bool = True):
        self.allow_lane_change = allow_lane_change
        self.target_lane = None
        self.cooldown = 0

    def reset(self):
        self.target_lane = None
        self.cooldown = 0

    def predict(self, observation, deterministic=True):
        state = observation['state'] if isinstance(observation, dict) else observation
        if isinstance(state, dict):
            raise ValueError('IDM requires state input, not BEV-only pixels')
        speed, target = float(state[0] * 30), float(state[5] * 30)
        width = float(state[8] * 5)
        lanes = max(1, round(float(state[7]) * 4))
        current_lane = round(float(state[11]) * max(1, lanes - 1))
        offset = float(state[3]) * width
        neighbors = np.asarray(state[20:]).reshape(-1, 7)
        visible = [row for row in neighbors if row[6] > 0.5]

        def lane_traffic(lane):
            center = (lane - current_lane) * width - offset
            return [row for row in visible if abs(float(row[1] * 20) - center) < width * 0.48]

        if self.target_lane is None or not 0 <= self.target_lane < lanes:
            self.target_lane = current_lane
        self.cooldown = max(0, self.cooldown - 1)
        current_cars = lane_traffic(current_lane)
        fronts = [row for row in current_cars if row[0] > 0]
        nearest = min(fronts, key=lambda row: row[0]) if fronts else None
        if (self.allow_lane_change and self.cooldown == 0 and self.target_lane == current_lane
                and abs(offset) < width * 0.25 and nearest is not None
                and nearest[0] * 60 < max(30, speed * 3) and nearest[2] * 30 < -1):
            for lane in (current_lane + 1, current_lane - 1):
                if not 0 <= lane < lanes:
                    continue
                clear = all((row[0] * 60 > max(20, speed * 2.0)) or
                            (row[0] * 60 < -max(15, (speed + row[2] * 30) * 1.5)) for row in lane_traffic(lane))
                if clear:
                    self.target_lane = lane
                    self.cooldown = 60
                    break
        fronts = [row for row in lane_traffic(self.target_lane) if row[0] > 0]
        nearest = min(fronts, key=lambda row: row[0]) if fronts else None
        gap = float(nearest[0] * 60 - 4.5) if nearest is not None else float('inf')
        closing = float(-nearest[2] * 30) if nearest is not None else 0.0
        acceleration = idm_acceleration(speed, target, gap, closing)
        x, y = float(state[14] * 60), float(state[15] * 20)
        y += (self.target_lane - current_lane) * width
        distance = max(math.hypot(x, y), 2.0)
        steering = math.atan2(2 * 2.7 * math.sin(math.atan2(y, x)), distance) / 0.45
        return np.clip([steering, acceleration / (3 if acceleration >= 0 else 6)], -1, 1).astype(np.float32)
