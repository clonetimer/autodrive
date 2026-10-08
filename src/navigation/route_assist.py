"""Transparent lateral route assistance for hybrid/autonomy-handoff research.

Two modes are intentionally explicit:

``fixed``
    Historical v2.9 behavior. The configured blend is applied on every step.
``adaptive``
    Rule authority is continuously gated by *policy-visible* route/tracking
    difficulty. Straight, well-tracked road can be left to the learned actor;
    curvature, heading error or lateral tracking error progressively hands
    steering authority to the geometric pure-pursuit controller.

The assist is never hidden inside the neural policy. Evaluation records its
actual time-varying blend, intervention rate and correction magnitude so a
hybrid result cannot be reported as raw-policy performance.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class RouteAssistDecision:
    proposed: np.ndarray
    applied: np.ndarray
    target_steering: float
    correction: float
    blend: float
    intervened: bool
    need_score: float = 1.0


def pure_pursuit_steering(state, *, wheelbase_m: float = 2.7, max_steering_rad: float = 0.45) -> float:
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 1 or len(state) < 20 or not np.isfinite(state).all():
        raise ValueError('pure_pursuit_steering expects a finite state vector with route preview')
    if wheelbase_m <= 0 or max_steering_rad <= 0:
        raise ValueError('vehicle geometry must be positive')
    # First preview point is normalized by [60m, 20m] in both state schemas.
    forward, lateral = float(state[12] * 60.0), float(state[13] * 20.0)
    lookahead = max(math.hypot(forward, lateral), 2.0)
    alpha = math.atan2(lateral, forward)
    steering_rad = math.atan2(2.0 * wheelbase_m * math.sin(alpha), lookahead)
    return float(np.clip(steering_rad / max_steering_rad, -1.0, 1.0))


def _linear_gate(value: float, start: float, full: float) -> float:
    """Map |value| to [0,1] with an explicit dead zone and saturation point."""
    if not (0 <= start < full):
        raise ValueError('adaptive route-assist thresholds require 0 <= start < full')
    return float(np.clip((abs(float(value)) - start) / (full - start), 0.0, 1.0))


def route_assist_need_score(
    state,
    target_steering: float | None = None,
    *,
    curvature_start: float = 0.06,
    curvature_full: float = 0.28,
    heading_start_rad: float = 0.035,
    heading_full_rad: float = 0.18,
    lane_start: float = 0.06,
    lane_full: float = 0.30,
) -> float:
    """Return policy-visible lateral difficulty in [0,1].

    The score deliberately excludes privileged simulator geometry. It uses the
    same normalized route preview, heading error and lane-offset features fed to
    the state policy. This makes the handoff auditable and prevents the adaptive
    layer from quietly receiving information unavailable to the actor.
    """
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 1 or len(state) < 20 or not np.isfinite(state).all():
        raise ValueError('route_assist_need_score expects a finite state vector')
    target = pure_pursuit_steering(state) if target_steering is None else float(target_steering)
    if not math.isfinite(target):
        raise ValueError('target steering must be finite')
    heading_error = math.atan2(float(state[1]), float(state[2]))
    lane_error_normalized = float(state[3])
    return max(
        _linear_gate(target, curvature_start, curvature_full),
        _linear_gate(heading_error, heading_start_rad, heading_full_rad),
        _linear_gate(lane_error_normalized, lane_start, lane_full),
    )


class RouteAssist:
    """Blend learned steering with a policy-visible pure-pursuit target."""

    def __init__(
        self,
        blend: float = 1.0,
        correction_deadband: float = 1e-4,
        *,
        mode: str = 'fixed',
        min_blend: float = 0.0,
        curvature_start: float = 0.06,
        curvature_full: float = 0.28,
        heading_start_rad: float = 0.035,
        heading_full_rad: float = 0.18,
        lane_start: float = 0.06,
        lane_full: float = 0.30,
    ):
        if mode not in {'fixed', 'adaptive'}:
            raise ValueError("route-assist mode must be 'fixed' or 'adaptive'")
        if not (0.0 <= min_blend <= blend <= 1.0) or correction_deadband < 0:
            raise ValueError('Route-assist blends must satisfy 0 <= min <= max <= 1 and deadband nonnegative')
        # Validate thresholds eagerly even in fixed mode so a serialized config
        # cannot become invalid only when adaptive mode is toggled later.
        for start, full in ((curvature_start, curvature_full),
                            (heading_start_rad, heading_full_rad),
                            (lane_start, lane_full)):
            if not (0 <= start < full):
                raise ValueError('adaptive route-assist thresholds require 0 <= start < full')
        self.blend = float(blend)
        self.min_blend = float(min_blend)
        self.correction_deadband = float(correction_deadband)
        self.mode = mode
        self.curvature_start = float(curvature_start)
        self.curvature_full = float(curvature_full)
        self.heading_start_rad = float(heading_start_rad)
        self.heading_full_rad = float(heading_full_rad)
        self.lane_start = float(lane_start)
        self.lane_full = float(lane_full)

    def decide(self, proposed_action, state) -> RouteAssistDecision:
        proposed = np.asarray(proposed_action, dtype=np.float32)
        if proposed.shape != (2,) or not np.isfinite(proposed).all():
            raise ValueError('RouteAssist expects finite [steering,longitudinal] action')
        target = pure_pursuit_steering(state)
        if self.mode == 'fixed':
            need = 1.0
            effective_blend = self.blend
        else:
            need = route_assist_need_score(
                state,
                target,
                curvature_start=self.curvature_start,
                curvature_full=self.curvature_full,
                heading_start_rad=self.heading_start_rad,
                heading_full_rad=self.heading_full_rad,
                lane_start=self.lane_start,
                lane_full=self.lane_full,
            )
            effective_blend = self.min_blend + (self.blend - self.min_blend) * need
        applied = proposed.copy()
        applied[0] = np.clip(
            (1.0 - effective_blend) * float(proposed[0]) + effective_blend * target,
            -1.0,
            1.0,
        )
        correction = abs(float(applied[0]) - float(proposed[0]))
        return RouteAssistDecision(
            proposed.copy(), applied, target, correction, effective_blend,
            bool(correction > self.correction_deadband), need,
        )
