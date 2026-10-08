"""Episode metrics with physical units and scenario-clustered uncertainty."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np


def weighted_quantile_abs(values: list[float], weights: list[float], q: float) -> float:
    """Inverse weighted empirical CDF, so variable dt does not overweight short steps."""
    order = np.argsort(np.abs(values))
    sorted_values = np.abs(np.asarray(values))[order]
    mass = np.cumsum(np.asarray(weights)[order])
    return float(sorted_values[min(int(np.searchsorted(mass, q * mass[-1], side='left')), len(order) - 1)])


def physical_summary(values: list[float], weights: list[float], duration: float) -> dict:
    if not values:
        return {'mean_abs': None, 'rms': None, 'p95_abs': None, 'peak_abs': None, 'coverage': 0.0}
    return {'mean_abs': float(np.average(np.abs(values), weights=weights)),
            'rms': float(np.sqrt(np.average(np.square(values), weights=weights))),
            'p95_abs': weighted_quantile_abs(values, weights, .95),
            'peak_abs': float(np.max(np.abs(values))),
            'coverage': float(sum(weights) / duration)}


@dataclass
class EpisodeMetrics:
    seed: int
    agent: str
    steps: int = 0
    duration_s: float = 0.0
    return_sum: float = 0.0
    speed_integral: float = 0.0
    lane_abs_integral: float = 0.0
    jerk_abs_integral: float = 0.0
    jerk_square_integral: float = 0.0
    lane_crossing_time: float = 0.0
    collision: bool = False
    out_of_road: bool = False
    success: bool = False
    truncated: bool = False
    completion: float = 0.0
    lane_changes: int = 0
    safety_intervention_time: float = 0.0
    safety_interventions: int = 0
    safety_correction_integral: float = 0.0
    safety_severity_integral: float = 0.0
    safety_correction_peak: float = 0.0
    route_assist_intervention_time: float = 0.0
    route_assist_interventions: int = 0
    route_assist_correction_integral: float = 0.0
    route_assist_correction_peak: float = 0.0
    route_assist_blend_integral: float = 0.0
    route_assist_need_integral: float = 0.0
    stalled_time: float = 0.0
    signed_progress_m: float = 0.0
    forward_progress_m: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)

    physical_samples: dict = field(default_factory=lambda: {key: ([], []) for key in
        ['longitudinal_jerk', 'lateral_acceleration', 'lateral_jerk', 'steering_rate']})
    minimum_ttc_s: float | None = None

    def update(self, reward: float, info: dict, terminated: bool, truncated: bool, latency_ms: float):
        dt = float(info['dt'])
        if not math.isfinite(dt) or not dt > 0:
            raise ValueError('Metric timestep must be positive')
        self.steps += 1
        self.duration_s += dt
        self.return_sum += float(reward)
        self.speed_integral += float(info['speed']) * dt
        self.lane_abs_integral += abs(float(info['lane_offset'])) * dt
        jerk = float(info['jerk'])
        self.jerk_abs_integral += abs(jerk) * dt
        self.jerk_square_integral += jerk * jerk * dt
        self.lane_crossing_time += (abs(info['lane_offset']) > info['lane_width'] * 0.4) * dt
        progress_m = float(info.get('progress_m', 0.0))
        self.signed_progress_m += progress_m
        self.forward_progress_m += max(progress_m, 0.0)
        # A deadlock/stall is low-speed time before route completion. Collision
        # or off-road terminal frames are excluded so failures are not double
        # counted as congestion.
        if (float(info['speed']) < 0.5 and float(info.get('route_completion', 0.0)) < 0.95
                and not bool(info.get('collision', False)) and not bool(info.get('out_of_road', False))):
            self.stalled_time += dt
        self.collision |= bool(info['collision'])
        self.out_of_road |= bool(info['out_of_road'])
        self.success |= bool(info['success']) and terminated
        self.truncated |= bool(truncated)
        self.completion = float(info['route_completion'])
        self.lane_changes = int(info.get('lane_changes', 0))
        if bool(info.get('safety_intervention', False)):
            self.safety_intervention_time += dt
            self.safety_interventions += 1
        proposed, applied = info.get('proposed_action'), info.get('applied_action')
        if proposed is not None and applied is not None:
            correction = abs(float(applied[1]) - float(proposed[1]))
            self.safety_correction_integral += correction * dt
            self.safety_correction_peak = max(self.safety_correction_peak, correction)
        self.safety_severity_integral += float(info.get('safety_severity', 0.0)) * dt
        if bool(info.get('route_assist_intervention', False)):
            self.route_assist_intervention_time += dt
            self.route_assist_interventions += 1
        route_correction = float(info.get('route_assist_steering_correction', 0.0))
        self.route_assist_correction_integral += route_correction * dt
        self.route_assist_correction_peak = max(self.route_assist_correction_peak, route_correction)
        route_blend = float(info.get('route_assist_blend', 0.0))
        route_need = float(info.get('route_assist_need_score', 0.0))
        if not (0.0 <= route_blend <= 1.0 and 0.0 <= route_need <= 1.0):
            raise ValueError('Route-assist blend/need metrics must lie in [0,1]')
        self.route_assist_blend_integral += route_blend * dt
        self.route_assist_need_integral += route_need * dt
        self.latencies_ms.append(float(latency_ms))
        for key, (values, weights) in self.physical_samples.items():
            value = info.get('jerk' if key == 'longitudinal_jerk' else key)
            if value is not None:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f'Nonfinite physical metric: {key}')
                values.append(value)
                weights.append(dt)
        ttc = info.get('risk', {}).get('ttc_s')
        if ttc is not None:
            self.minimum_ttc_s = float(ttc) if self.minimum_ttc_s is None else min(self.minimum_ttc_s, float(ttc))

    def result(self) -> dict:
        if self.steps == 0:
            raise ValueError('Cannot report an unexecuted episode')
        duration = self.duration_s
        result = {'seed': self.seed, 'agent': self.agent, 'steps': self.steps, 'duration_s': duration,
                'return': self.return_sum, 'success': self.success and not self.collision and not self.out_of_road,
                'collision': self.collision, 'out_of_road': self.out_of_road, 'truncated': self.truncated,
                'route_completion': self.completion, 'avg_speed_mps': self.speed_integral / duration,
                'lane_abs_error_m': self.lane_abs_integral / duration,
                'jerk_mean_abs_mps3': self.jerk_abs_integral / duration,
                'jerk_rms_mps3': math.sqrt(self.jerk_square_integral / duration),
                'lane_crossing_time_fraction': self.lane_crossing_time / duration,
                'stall_time_fraction': self.stalled_time / duration,
                'signed_progress_mps': self.signed_progress_m / duration,
                'forward_progress_mps': self.forward_progress_m / duration,
                'lane_changes': self.lane_changes,
                'safety_interventions': self.safety_interventions,
                'safety_intervention_time_fraction': self.safety_intervention_time / duration,
                'safety_mean_longitudinal_correction': self.safety_correction_integral / duration,
                'safety_peak_longitudinal_correction': self.safety_correction_peak,
                'safety_mean_severity': self.safety_severity_integral / duration,
                'route_assist_interventions': self.route_assist_interventions,
                'route_assist_intervention_time_fraction': self.route_assist_intervention_time / duration,
                'route_assist_mean_steering_correction': self.route_assist_correction_integral / duration,
                'route_assist_peak_steering_correction': self.route_assist_correction_peak,
                'route_assist_mean_blend': self.route_assist_blend_integral / duration,
                'route_assist_mean_need_score': self.route_assist_need_integral / duration,
                'learned_steering_authority_fraction': 1.0 - self.route_assist_blend_integral / duration,
                'inference_mean_ms': float(np.mean(self.latencies_ms)),
                'inference_p95_ms': float(np.quantile(self.latencies_ms, 0.95))}

        units = {'longitudinal_jerk': 'mps3', 'lateral_jerk': 'mps3',
                 'lateral_acceleration': 'mps2', 'steering_rate': 'radps'}
        for key, (values, weights) in self.physical_samples.items():
            summary = physical_summary(values, weights, duration)
            for statistic, value in summary.items():
                name = f'{key}_{statistic}' if statistic == 'coverage' else f'{key}_{statistic}_{units[key]}'
                result[name] = value
        result['minimum_ttc_s'] = self.minimum_ttc_s
        return result


def clustered_interval(episodes: list[dict], key: str, seed: int = 0) -> list[float]:
    clusters = sorted({row['seed'] for row in episodes})
    values = np.array([np.mean([float(row[key]) for row in episodes if row['seed'] == group]) for group in clusters])
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(len(values), size=(2000, len(values)))].mean(axis=1)
    return np.quantile(sampled, [0.025, 0.975]).tolist()


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    """Binomial score interval for independent scenarios, including 0/n and n/n."""
    if trials <= 0 or not 0 <= successes <= trials or not z > 0:
        raise ValueError('Invalid binomial counts or confidence quantile')
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(episodes: list[dict], latencies: list[float] | None = None) -> dict:
    if not episodes:
        raise ValueError('No evaluated episodes')
    durations = np.asarray([row['duration_s'] for row in episodes])
    groups = sorted({row['seed'] for row in episodes})
    result = {'agent_episodes': len(episodes), 'scenario_episodes': len(groups),
              'success_rate': float(np.mean([row['success'] for row in episodes])),
              'collision_rate': float(np.mean([row['collision'] for row in episodes])),
              'out_of_road_rate': float(np.mean([row['out_of_road'] for row in episodes])),
              'timeout_rate': float(np.mean([row['truncated'] for row in episodes])),
              'mean_route_completion': float(np.mean([row['route_completion'] for row in episodes])),
              'mean_episode_return': float(np.mean([row['return'] for row in episodes])),
              'joint_success_rate': float(np.mean([all(row['success'] for row in episodes if row['seed'] == group) for group in groups])),
              'scenario_any_collision_rate': float(np.mean([any(row['collision'] for row in episodes if row['seed'] == group) for group in groups]))}
    for key in ['avg_speed_mps', 'lane_abs_error_m', 'jerk_mean_abs_mps3', 'lane_crossing_time_fraction',
                'stall_time_fraction', 'signed_progress_mps', 'forward_progress_mps']:
        result[key] = float(np.average([row[key] for row in episodes], weights=durations))
    result['safety_intervention_time_fraction'] = float(np.average(
        [row.get('safety_intervention_time_fraction', 0.0) for row in episodes], weights=durations))
    result['safety_interventions_total'] = int(sum(row.get('safety_interventions', 0) for row in episodes))
    result['safety_mean_longitudinal_correction'] = float(np.average(
        [row.get('safety_mean_longitudinal_correction', 0.0) for row in episodes], weights=durations))
    result['safety_peak_longitudinal_correction'] = float(max(
        row.get('safety_peak_longitudinal_correction', 0.0) for row in episodes))
    result['safety_mean_severity'] = float(np.average(
        [row.get('safety_mean_severity', 0.0) for row in episodes], weights=durations))
    result['route_assist_intervention_time_fraction'] = float(np.average(
        [row.get('route_assist_intervention_time_fraction', 0.0) for row in episodes], weights=durations))
    result['route_assist_interventions_total'] = int(sum(row.get('route_assist_interventions', 0) for row in episodes))
    result['route_assist_mean_steering_correction'] = float(np.average(
        [row.get('route_assist_mean_steering_correction', 0.0) for row in episodes], weights=durations))
    result['route_assist_peak_steering_correction'] = float(max(
        row.get('route_assist_peak_steering_correction', 0.0) for row in episodes))
    result['route_assist_mean_blend'] = float(np.average(
        [row.get('route_assist_mean_blend', 0.0) for row in episodes], weights=durations))
    result['route_assist_mean_need_score'] = float(np.average(
        [row.get('route_assist_mean_need_score', 0.0) for row in episodes], weights=durations))
    result['learned_steering_authority_fraction'] = float(np.average(
        [row.get('learned_steering_authority_fraction', 1.0) for row in episodes], weights=durations))
    result['jerk_rms_mps3'] = float(np.sqrt(np.average([row['jerk_rms_mps3'] ** 2 for row in episodes], weights=durations)))

    for key, unit in [('lateral_acceleration', 'mps2'), ('lateral_jerk', 'mps3'), ('steering_rate', 'radps')]:
        observed = [row for row in episodes if row.get(f'{key}_rms_{unit}') is not None]
        covered = [row['duration_s'] * row[f'{key}_coverage'] for row in observed]
        coverage_duration = sum(covered)
        result[f'{key}_coverage'] = float(coverage_duration / durations.sum())
        result[f'{key}_rms_{unit}'] = (float(np.sqrt(np.average(
            [row[f'{key}_rms_{unit}'] ** 2 for row in observed], weights=covered))) if observed else None)
        result[f'{key}_peak_abs_{unit}'] = max((row[f'{key}_peak_abs_{unit}'] for row in observed), default=None)
        result[f'{key}_mean_episode_p95_abs_{unit}'] = (float(np.average(
            [row[f'{key}_p95_abs_{unit}'] for row in observed], weights=covered)) if observed else None)
    result['success_rate_cluster_ci95'] = clustered_interval(episodes, 'success')
    result['collision_rate_cluster_ci95'] = clustered_interval(episodes, 'collision')
    if latencies:
        result['inference_p50_ms'] = float(np.quantile(latencies, 0.5))
        result['inference_p95_ms'] = float(np.quantile(latencies, 0.95))
    joint_successes = sum(all(row['success'] for row in episodes if row['seed'] == group) for group in groups)
    scenario_collisions = sum(any(row['collision'] for row in episodes if row['seed'] == group) for group in groups)
    result['joint_success_wilson_ci95'] = wilson_interval(joint_successes, len(groups))
    result['scenario_any_collision_wilson_ci95'] = wilson_interval(scenario_collisions, len(groups))
    result['uncertainty_note'] = ('Wilson intervals use one binary outcome per independent scenario; '
        'agent-rate percentile bootstrap clusters correlated MARL agents by scenario and can degenerate at 0/1. '
        'Neither interval represents uncertainty across training seeds.')
    return result
