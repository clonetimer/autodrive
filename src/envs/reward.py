"""Versioned driving objectives. v3 integrates reward *rates* in physical time.

Geometry-derived risk is a training diagnostic, not a safety certificate. The
risk model uses a road-tangent approximation and does not predict intentions.
Historical v2 equations are kept byte-for-byte in spirit for fair ablations.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import numpy as np
from src.envs.observation import Snapshot

REWARD_WEIGHTS = {
    'balanced': dict(progress=1.0, lane=0.08, jerk=0.015, collision=12.0, offroad=10.0, success=12.0),
    'safe': dict(progress=0.8, lane=0.1, jerk=0.03, collision=24.0, offroad=15.0, success=12.0),
    'efficient': dict(progress=1.4, lane=0.04, jerk=0.005, collision=10.0, offroad=8.0, success=12.0),
}
# Rates are scaled to retain a similar magnitude at the original 0.1 s step.
REFERENCE_DT = 0.1
RISK_WEIGHTS = {'balanced': (0.5, 0.75, 1.8), 'safe': (0.6, 1.0, 2.5), 'efficient': (0.4, 0.6, 1.4)}
# v4 adds explicit finite-horizon collision-avoidance shaping. Tuple fields:
# (ttc_weight, closing_weight, unsafe_accel_weight, emergency_brake_credit).
EMERGENCY_WEIGHTS = {
    'balanced': (1.2, 0.35, 0.8, 0.20),
    'safe': (1.8, 0.50, 1.1, 0.25),
    'efficient': (0.9, 0.25, 0.6, 0.15),
}


@dataclass(frozen=True)
class TrafficRisk:
    front_id: str | None = None
    gap_m: float | None = None
    closing_mps: float = 0.0
    ttc_s: float | None = None
    front_risk: float = 0.0
    rear_risk: float = 0.0
    safe_target_mps: float = 0.0
    anticipated_overlap: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def traffic_risk(snapshot: Snapshot, horizon: float = 2.0) -> TrafficRisk:
    """Use bumper gap, closing speed and lateral overlap, not Euclidean distance.

    Include vehicles whose lateral swept intervals overlap within ``horizon``;
    do not classify a parallel neighboring vehicle as the ego's leader. This
    also exposes fast rear approaches during a cut-in. All velocities are
    projected onto the local road tangent (not the global x axis).
    """
    ego = snapshot.ego
    c, s = math.cos(snapshot.road_heading), math.sin(snapshot.road_heading)
    tangent, lateral = np.array([c, s]), np.array([-s, c])
    ego_v = ego.speed * np.array([math.cos(ego.yaw), math.sin(ego.yaw)])
    ego_forward = max(float(ego_v @ tangent), 0.0)
    candidates, rear_risk = [], 0.0
    for vehicle in snapshot.vehicles:
        if vehicle.id == ego.id or not vehicle.active:
            continue
        delta = np.array([vehicle.x - ego.x, vehicle.y - ego.y])
        dx, dy = float(delta @ tangent), float(delta @ lateral)
        if abs(dx) > 100:
            continue
        velocity = vehicle.speed * np.array([math.cos(vehicle.yaw), math.sin(vehicle.yaw)])
        relative = velocity - ego_v
        dvy = float(relative @ lateral)
        half_width = (ego.width + vehicle.width) / 2 + 0.2
        currently_overlaps = abs(dy) <= half_width
        # A swept lateral interval; static adjacent lanes remain excluded.
        future_dy = dy + dvy * horizon
        may_overlap = min(dy, future_dy) <= half_width and max(dy, future_dy) >= -half_width
        if not may_overlap:
            continue
        gap = abs(dx) - (ego.length + vehicle.length) / 2
        if dx >= 0:
            closing = -float(relative @ tangent)
            ttc = max(gap, 0.0) / closing if closing > 1e-6 else None
            # A stopped ego at a normal standstill gap is not unsafe.
            desired_gap = 2.0 + 1.5 * ego_forward
            headway_risk = float(np.clip(1 - max(gap, 0) / desired_gap, 0, 1))
            ttc_risk = 0.0 if ttc is None else float(np.clip(1 - ttc / 3.0, 0, 1))
            lead_speed = max(float(velocity @ tangent), 0.0)
            available = max(gap - 2.0, 0.0)
            safe = min(ego.target_speed, available / 1.5,
                       math.sqrt(lead_speed ** 2 + 2 * 2.5 * available))
            candidates.append((gap, vehicle.id, closing, ttc, max(headway_risk, ttc_risk),
                               safe, not currently_overlaps))
        else:
            rear_closing = float(relative @ tangent)
            if rear_closing > 1e-6:
                rear_ttc = max(gap, 0.0) / rear_closing
                rear_risk = max(rear_risk, float(np.clip(1 - rear_ttc / 3.0, 0, 1)))
    if not candidates:
        return TrafficRisk(rear_risk=rear_risk, safe_target_mps=float(ego.target_speed))
    front = min(candidates, key=lambda item: (item[0], item[1]))
    gap, name, closing, ttc, risk, safe, anticipated = front
    return TrafficRisk(name, float(gap), float(closing), ttc, float(risk), rear_risk, float(safe), anticipated)


def reward_terms(snapshot: Snapshot, progress_m: float, jerk: float, dt: float,
                 profile: str = 'balanced', version: int = 3, *, collision: bool = False,
                 offroad: bool = False, success: bool = False,
                 risk: TrafficRisk | None = None) -> dict[str, float]:
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Reward dt must be finite and positive')
    if version not in {2, 3, 4}:
        raise ValueError('Unknown reward version')
    weights = REWARD_WEIGHTS[profile]
    car = snapshot.ego
    reference_speed = max(car.target_speed, 1e-6)
    progress = progress_m / (reference_speed * dt)
    lane = abs(snapshot.lane_offset) / snapshot.lane_width
    if version == 2:
        terms = {
            'progress': weights['progress'] * progress,
            'lane': -weights['lane'] * lane,
            'comfort': -weights['jerk'] * min(abs(jerk) / 10, 5),
            'speed': -0.1 * max(car.speed / reference_speed - 1, 0),
        }
    else:
        risk = risk or traffic_risk(snapshot)
        speed_weight, overspeed_weight, risk_weight = RISK_WEIGHTS[profile]
        speed_error = (car.speed - risk.safe_target_mps) / reference_speed
        overspeed = max(car.speed / reference_speed - 1, 0.0)
        # Emergency braking may be uncomfortable but must not be dominated by a
        # comfort objective while the speed/gap safety objective asks for braking.
        urgency = max(risk.front_risk, max(car.speed - risk.safe_target_mps, 0) / reference_speed)
        comfort_multiplier = 1 - min(urgency, 1.0) if car.acceleration < 0 else 1.0
        rates = {
            'progress': weights['progress'] * float(np.clip(progress, -1, 1)),
            'lane': -weights['lane'] * lane,
            'comfort': -weights['jerk'] * min(abs(jerk) / 10, 5) * comfort_multiplier,
            'speed': -speed_weight * min(speed_error ** 2, 9.0),
            'overspeed': -overspeed_weight * min(overspeed ** 2, 9.0),
            'risk': -risk_weight * max(risk.front_risk, risk.rear_risk),
        }
        if version == 4:
            ttc_weight, closing_weight, unsafe_accel_weight, brake_credit_weight = EMERGENCY_WEIGHTS[profile]
            # TTC cost grows smoothly inside a four-second horizon and reaches one
            # at impact. Missing TTC (not closing) contributes no false penalty.
            ttc_cost = 0.0 if risk.ttc_s is None else float(np.clip((4.0 - risk.ttc_s) / 4.0, 0.0, 1.0) ** 2)
            closing_ratio = float(np.clip(max(risk.closing_mps, 0.0) / reference_speed, 0.0, 2.0))
            positive_accel = float(np.clip(max(car.acceleration, 0.0) / 3.0, 0.0, 1.0))
            braking = float(np.clip(max(-car.acceleration, 0.0) / 4.0, 0.0, 1.0))
            # The credit is gated by risk and only offsets a fraction of the
            # safety penalty; it cannot make repeated hard braking profitable.
            rates.update(
                ttc=-ttc_weight * ttc_cost,
                closing=-closing_weight * risk.front_risk * closing_ratio ** 2,
                unsafe_accel=-unsafe_accel_weight * urgency * positive_accel,
                emergency_brake=brake_credit_weight * urgency * braking,
            )
        terms = {key: float(value * dt / REFERENCE_DT) for key, value in rates.items()}
    # One-off terminal events are not rates and must not be scaled by dt.
    terms.update(collision=-weights['collision'] * bool(collision),
                 offroad=-weights['offroad'] * bool(offroad),
                 success=weights['success'] * bool(success and not collision and not offroad))
    return {key: float(value) for key, value in terms.items()}
