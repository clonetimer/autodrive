"""Training-only stage advancement based on measured recent episode outcomes."""
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Curriculum:
    min_episodes: int = 20
    success_threshold: float = 0.75
    max_collision_rate: float = 0.15
    stage: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=40))

    def observe(self, success: bool, collision: bool) -> bool:
        self.history.append((bool(success), bool(collision)))
        if self.stage >= 2 or len(self.history) < self.min_episodes:
            return False
        success_rate = sum(x[0] for x in self.history) / len(self.history)
        crash_rate = sum(x[1] for x in self.history) / len(self.history)
        if success_rate >= self.success_threshold and crash_rate <= self.max_collision_rate:
            self.stage += 1
            self.history.clear()
            return True
        return False

    def parameters(self) -> dict:
        return [dict(traffic_vehicles=0, curve_amplitude=0.0),
                dict(traffic_vehicles=5, curve_amplitude=2.0),
                dict(traffic_vehicles=12, curve_amplitude=5.0)][self.stage]
