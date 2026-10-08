"""Seeded road generation with explicit train/validation/test seed partitions."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from src.utils.config import EnvConfig

SPLITS = {
    'train': (0, 999),
    'validation': (1000, 1199),
    'test': (2000, 2199),
    'final': (3000, 3199),
    'release': (4000, 4199),
    'release2': (5000, 5199),
    'release3': (6000, 6199),
    'release4': (7000, 7199),
    'safety_train': (8000, 8999),
    'release5': (9000, 9199),
    # v2.7+ uses fresh domains so earlier validation/release observations cannot
    # influence the new privileged-teacher experiments.
    'expert_train': (10000, 10999),
    'validation2': (11000, 11199),
    'release6': (12000, 12199),
    # v2.8+ capability-specialization domains. These remain disjoint from all
    # earlier safety/expert experiments so dense-traffic, merge and yield
    # tuning cannot reuse validation2/release6 outcomes.
    'capability_train': (13000, 13999),
    'validation3': (14000, 14199),
    'release7': (15000, 15199),
    # validation3 was consumed while calibrating the first v2.8 capability-suite
    # draft. Final candidate selection therefore moves to validation4 and a
    # completely fresh release8 domain.
    'validation4': (16000, 16199),
    'release8': (17000, 17199),
    # validation4 became a development screen for component-wise distillation.
    # v2.8 final promotion therefore uses a new validation5/release9 pair.
    'validation5': (18000, 18199),
    'release9': (19000, 19199),
    # validation5 exposed a collision-to-offroad trade-off in unbounded
    # component-wise imitation. Reference-bounded adaptation is therefore
    # selected only on a new validation6 and, if promoted, release10.
    'validation6': (20000, 20199),
    'release10': (21000, 21199),
    # validation6 was consumed by the bounded-component v2.8 screen. Since no
    # candidate passed the promotion gate, release10 remains unopened. Any future
    # tuning must move to validation7 and may open release11 only after a freeze.
    'validation7': (22000, 22199),
    'release11': (23000, 23199),
    # v2.9 route-observation work. 24000-24999 is reserved for diagnostics only
    # because seed 24000 was already used during the schema-v4 smoke audit.
    'route_dev': (24000, 24999),
    'route_train': (25000, 25999),
    'validation8': (26000, 26199),
    'release12': (27000, 27199),
    # v3.0 adaptive route-authority handoff. release12 is consumed; all handoff
    # tuning moves to a new training/validation domain and release13 remains
    # unopened until the selected system is frozen.
    'handoff_train': (28000, 28999),
    'validation9': (29000, 29199),
    'release13': (30000, 30199),
    # v3.1 learned route-adapter distillation.  The v3.0 shared-control holdout
    # is consumed, so online DAgger collection and model selection move to fresh
    # domains. release14 must remain unopened until the adapter artifact and its
    # selection evidence are frozen.
    'adapter_train': (31000, 31999),
    'validation10': (32000, 32199),
    'release14': (33000, 33199),
    # v3.2 curve-specialist rule-to-policy transfer. release14 is consumed;
    # curve-focused DAgger, selection and final holdout use new disjoint domains.
    'curve_train': (34000, 34999),
    'validation11': (35000, 35199),
    'release15': (36000, 36199),
    # v3.3 temporal/scene-aware learned adapter. release15 is consumed and must
    # remain historical evidence. validation12 seed 38000 was later used by an
    # engineering evaluator smoke, so validation12 is now development-only.
    # Candidate selection moves to validation13; release16 remains untouched.
    'scene_train': (37000, 37999),
    'validation12': (38000, 38199),
    'release16': (39000, 39199),
    'validation13': (40000, 40199),
    # v4.0 joint-action / support-correction experiments. v3.3 failed its
    # validation13 promotion gate, so release16 remains permanently unopened.
    # v4 uses entirely fresh train/selection/final domains.
    'joint_train': (41000, 41999),
    'validation14': (42000, 42199),
    'release17': (43000, 43199),
    # v4.1 factorized longitudinal learning. v4.0 failed validation14 because
    # relearning steering regressed the already-strong v3.2 curve specialist;
    # release17 therefore remains unopened. v4.1 freezes v3.2 lateral control
    # and uses fresh domains for longitudinal-only expert transfer.
    'longitudinal_train': (44000, 44999),
    'validation15': (45000, 45199),
    'release18': (46000, 46199),
    # v4.2 authority-gated longitudinal learning. v4.1 improved interaction on
    # validation15 but failed standard capability non-regression; release18
    # remains unopened. v4.2 uses a fresh development/train/selection/release
    # chain and keeps the registered v3.2 lateral controller frozen.
    'authority_train': (47000, 47999),
    'validation16': (48000, 48199),
    'release19': (49000, 49199),
}


@dataclass
class Road:
    length: float
    lanes: int
    lane_width: float
    amplitude: float = 0.0
    wavelength: float = 220.0

    def center(self, x):
        return self.amplitude * np.sin(2 * np.pi * np.asarray(x) / self.wavelength)

    def heading(self, x: float) -> float:
        return math.atan(self.amplitude * 2 * math.pi / self.wavelength * math.cos(2 * math.pi * x / self.wavelength))

    def lane_y(self, x, lane: int):
        return self.center(x) + (lane - (self.lanes - 1) / 2) * self.lane_width

    def lane_index(self, x: float, y: float) -> int:
        return int(np.clip(np.rint((y - self.center(x)) / self.lane_width + (self.lanes - 1) / 2), 0, self.lanes - 1))

    def offset(self, x: float, y: float) -> float:
        return float(y - self.lane_y(x, self.lane_index(x, y)))

    def polylines(self, start: float, end: float, route_lane: int):
        xs = np.linspace(start, end, 45)
        boundaries = [np.column_stack((xs, self.center(xs) + (i - self.lanes / 2) * self.lane_width))
                      for i in range(self.lanes + 1)]
        route = np.column_stack((xs, self.lane_y(xs, route_lane)))
        return boundaries, route


def generate_road(config: EnvConfig, rng: np.random.Generator) -> Road:
    amplitude = config.curve_amplitude
    width = config.lane_width
    if config.randomize:
        amplitude = float(rng.uniform(-max(abs(amplitude), 4), max(abs(amplitude), 4)))
        width = float(rng.uniform(max(2.8, width - 0.25), width + 0.25))
    return Road(config.road_length, config.lanes, width, amplitude, config.curve_wavelength)


def split_seeds(split: str, count: int) -> list[int]:
    if split not in SPLITS:
        raise ValueError(f'Unknown split {split}')
    lo, hi = SPLITS[split]
    if not 1 <= count <= hi - lo + 1:
        raise ValueError('Invalid split size')
    return list(range(lo, lo + count))


def split_seed_block(split: str, start: int, count: int) -> list[int]:
    """Return a deterministic contiguous block inside a named split.

    ``start`` is an offset from the split lower bound.  Keeping this helper in
    the scenario module makes training scripts and tests share the same split
    arithmetic instead of reimplementing it ad hoc.
    """
    if split not in SPLITS:
        raise ValueError(f'Unknown split {split}')
    lo, hi = SPLITS[split]
    begin = lo + int(start)
    if count <= 0 or start < 0 or begin + count - 1 > hi:
        raise ValueError('Requested seed block is out of range')
    return list(range(begin, begin + count))
