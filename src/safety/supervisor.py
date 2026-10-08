"""Transparent longitudinal safety supervisor for learned driving policies.

The supervisor is intentionally narrow: it can only limit the policy's
longitudinal command and never changes steering.  It is a system-level safety
layer, not part of the learned policy, so every modification is exposed in the
step info and must be reported separately from raw-policy metrics.

Version 2 uses a small amount of state to avoid the pathological throttle/brake
oscillation of a purely stateless emergency clamp.  It starts by trimming
throttle at a predictive warning threshold, ramps braking with risk, holds the
intervention until the scene is clearly safe, and rate-limits recovery.  Truly
critical TTC can still request immediate hard braking.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.envs.observation import Snapshot
from src.envs.reward import TrafficRisk, traffic_risk


@dataclass(frozen=True)
class SafetyDecision:
    proposed: np.ndarray
    applied: np.ndarray
    intervened: bool
    reason: str
    severity: float
    risk: TrafficRisk


def _risk_state(snapshot: Snapshot, ttc_threshold_s: float, risk_threshold: float):
    risk = traffic_risk(snapshot)
    # The warning region deliberately starts before the emergency threshold so
    # braking can be progressive instead of an impulse at the last moment.
    warning_ttc = 2.0 * ttc_threshold_s
    warning_risk = 0.5 * risk_threshold
    ttc_warning = (0.0 if risk.ttc_s is None else
                   float(np.clip((warning_ttc - risk.ttc_s) / warning_ttc, 0.0, 1.0)))
    risk_warning = float(np.clip((risk.front_risk - warning_risk) /
                                 max(1.0 - warning_risk, 1e-6), 0.0, 1.0))
    speed_urgency = float(np.clip((snapshot.ego.speed - risk.safe_target_mps) /
                                  max(snapshot.ego.target_speed, 1.0), 0.0, 1.0))
    severity = max(ttc_warning, risk_warning, speed_urgency)
    soft_trigger = ((risk.ttc_s is not None and risk.ttc_s < warning_ttc) or
                    risk.front_risk >= warning_risk or speed_urgency > 0.10)
    critical = ((risk.ttc_s is not None and risk.ttc_s < ttc_threshold_s) or
                risk.front_risk >= risk_threshold or speed_urgency > 0.25)
    clear = ((risk.ttc_s is None or risk.ttc_s > 2.4 * ttc_threshold_s) and
             risk.front_risk < 0.35 * risk_threshold and speed_urgency < 0.06)
    return risk, float(severity), bool(soft_trigger), bool(critical), bool(clear)


class SafetySupervisor:
    """Stateful longitudinal action filter with hysteresis and slew limiting."""

    def __init__(self, *, ttc_threshold_s: float = 1.5, risk_threshold: float = 0.75,
                 release_steps: int = 4, max_slew: float = 0.28):
        if not np.isfinite([ttc_threshold_s, risk_threshold, max_slew]).all() or ttc_threshold_s <= 0:
            raise ValueError('Invalid safety-supervisor thresholds')
        if not 0 < risk_threshold <= 1 or not 0 < max_slew <= 2 or release_steps < 1:
            raise ValueError('Invalid safety-supervisor thresholds')
        self.ttc_threshold_s = float(ttc_threshold_s)
        self.risk_threshold = float(risk_threshold)
        self.release_steps = int(release_steps)
        self.max_slew = float(max_slew)
        self.active = False
        self.clear_steps = 0
        self.last_longitudinal: float | None = None

    def reset(self):
        self.active = False
        self.clear_steps = 0
        self.last_longitudinal = None

    def decide(self, action, snapshot: Snapshot) -> SafetyDecision:
        proposed = np.asarray(action, dtype=np.float32)
        if proposed.shape != (2,) or not np.isfinite(proposed).all():
            raise ValueError('Safety supervisor expects two finite normalized controls')
        proposed = np.clip(proposed, -1, 1).astype(np.float32, copy=True)
        risk, severity, soft, critical, clear = _risk_state(
            snapshot, self.ttc_threshold_s, self.risk_threshold)

        if soft or critical:
            self.active = True
            self.clear_steps = 0
        elif self.active:
            self.clear_steps = self.clear_steps + 1 if clear else 0
            if self.clear_steps >= self.release_steps:
                self.active = False
                self.clear_steps = 0

        desired = float(proposed[1])
        reason = 'none'
        if self.active:
            # At the warning boundary the supervisor merely trims throttle.
            # Progressively move through coasting to braking as risk increases.
            longitudinal_cap = float(np.clip(0.30 - 1.30 * severity, -1.0, 0.30))
            desired = min(desired, longitudinal_cap)
            reason = 'predictive_front_risk'
            # Imminent impact is allowed to bypass the comfort ramp downward.
            imminent = risk.ttc_s is not None and risk.ttc_s < 0.55 * self.ttc_threshold_s
            if imminent:
                desired = -1.0
                reason = 'critical_ttc'
        elif self.last_longitudinal is not None and proposed[1] > self.last_longitudinal + self.max_slew:
            # Smooth release back to the actor to avoid brake/throttle chattering.
            desired = self.last_longitudinal + self.max_slew
            reason = 'recovery_slew'

        if self.last_longitudinal is not None and reason != 'critical_ttc':
            # Riskier states are permitted to ramp braking more quickly, while
            # throttle recovery remains conservative.
            downward = self.max_slew + 0.45 * severity
            desired = float(np.clip(desired,
                                    self.last_longitudinal - downward,
                                    self.last_longitudinal + self.max_slew))

        applied = proposed.copy()
        applied[1] = float(np.clip(desired, -1.0, 1.0))
        self.last_longitudinal = float(applied[1])
        intervened = bool(abs(float(applied[1]) - float(proposed[1])) > 1e-7)
        if not intervened:
            reason = 'none'
        return SafetyDecision(proposed, applied, intervened, reason, severity, risk)


def supervise_action(action, snapshot: Snapshot, *, ttc_threshold_s: float = 1.5,
                     risk_threshold: float = 0.75) -> SafetyDecision:
    """One-step convenience API; runtime environments use :class:`SafetySupervisor`."""
    return SafetySupervisor(ttc_threshold_s=ttc_threshold_s, risk_threshold=risk_threshold).decide(action, snapshot)
