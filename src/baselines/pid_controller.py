"""State-only lateral preview controller with longitudinal PI speed regulation."""
import math
import numpy as np


class PIDController:
    def __init__(self, dt: float = 0.1):
        self.dt = dt
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def predict(self, observation, deterministic=True):
        state = observation['state'] if isinstance(observation, dict) else observation
        speed, target = float(state[0] * 30), float(state[5] * 30)
        error = target - speed
        self.integral = float(np.clip(self.integral + error * self.dt, -5, 5))
        acceleration = float(np.clip(0.7 * error + 0.08 * self.integral, -6, 3))
        # Route point at 8 m in ego frame; pure pursuit lateral geometry.
        forward, lateral = float(state[12] * 60), float(state[13] * 20)
        lookahead = max(math.hypot(forward, lateral), 2)
        alpha = math.atan2(lateral, forward)
        steer = math.atan2(2 * 2.7 * math.sin(alpha), lookahead) / 0.45
        return np.clip([steer, acceleration / (3 if acceleration >= 0 else 6)], -1, 1).astype(np.float32)
