"""Transparent lateral lane-stabilization teacher for offline/DAgger supervision.

The teacher is *training-only*. It never runs in deployment evaluation unless a
caller explicitly asks for teacher targets. It uses only the same normalized
state vector exposed to the learned policy: lane/heading error and local route
preview. The goal is to provide bounded corrective steering labels on states
where the learned policy is already drifting or approaching a curve, without
silently turning the runtime SafetySupervisor into a rule-based steering stack.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class LaneTeacherDecision:
    proposed_steer: float
    teacher_steer: float
    target_steer: float
    intervened: bool
    severity: float
    lane_offset_norm: float
    heading_error_sin: float
    preview_lateral_norm: float


def _pure_pursuit_steer(state: np.ndarray) -> float:
    # First route point is ~8m ahead in ego coordinates, normalized by [60,20].
    forward = float(state[12]) * 60.0
    lateral = float(state[13]) * 20.0
    lookahead = max(math.hypot(forward, lateral), 2.0)
    alpha = math.atan2(lateral, forward)
    # Same wheelbase / nominal max steering convention as PIDController.
    steer = math.atan2(2.0 * 2.7 * math.sin(alpha), lookahead) / 0.45
    return float(np.clip(steer, -1.0, 1.0))


def lane_teacher_action(proposed_action, state, *, max_blend: float = 0.70,
                        activation: float = 0.20) -> LaneTeacherDecision:
    """Return a bounded steering target while preserving longitudinal control.

    ``severity`` is built from three policy-visible signals:
    lane-center offset, heading error and preview lateral displacement.  The
    teacher remains inactive in a centered straight-lane state.  It becomes
    progressively stronger near lane departure or on meaningful upcoming
    curvature.  No privileged simulator coordinates are used.
    """
    proposed = np.asarray(proposed_action, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    if proposed.shape != (2,) or state.ndim != 1 or len(state) < 20:
        raise ValueError('Lane teacher expects action [2] and state observation >=20')
    if not np.isfinite(proposed).all() or not np.isfinite(state).all():
        raise ValueError('Lane teacher inputs must be finite')
    if not (0 < max_blend <= 1 and 0 <= activation < 1):
        raise ValueError('Invalid lane-teacher blend/activation')

    lane = abs(float(state[3]))
    heading = abs(float(state[1]))
    # Use the first three route-preview y components. The furthest point can
    # reflect a junction branch too early and destabilize otherwise safe travel.
    preview = max(abs(float(state[13])), abs(float(state[15])), abs(float(state[17])))
    lane_score = float(np.clip(lane / 0.75, 0.0, 1.0))
    heading_score = float(np.clip(heading / 0.40, 0.0, 1.0))
    preview_score = float(np.clip(preview / 0.12, 0.0, 1.0))
    raw = max(lane_score, heading_score, preview_score)
    severity = float(np.clip((raw - activation) / max(1.0 - activation, 1e-6), 0.0, 1.0))
    teacher = _pure_pursuit_steer(state)
    blend = max_blend * severity
    target = float(np.clip((1.0 - blend) * float(proposed[0]) + blend * teacher, -1.0, 1.0))
    intervened = bool(abs(target - float(proposed[0])) > 1e-7)
    return LaneTeacherDecision(float(proposed[0]), teacher, target, intervened, severity,
                               lane, heading, preview)
