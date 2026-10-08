"""Build the v2.9 hybrid route-control validation/release12 report.

The v2.9 learned-policy experiments are intentionally retained as negative
results.  The promoted *system* is the unchanged v2.8 neural policy plus an
explicit policy-visible pure-pursuit route assist and the independently audited
longitudinal SafetySupervisor.  release12 is evaluated only after an explicit
pre-release freeze manifest exists.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import _bootstrap
from src.utils.logger import save_json, sha256


def exact_mcnemar_p(improved: int, regressed: int) -> float:
    n = int(improved) + int(regressed)
    if n <= 0:
        return 1.0
    k = min(int(improved), int(regressed))
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes'}


def _episode_map(report_dir: Path) -> dict[tuple[str, int], dict]:
    result: dict[tuple[str, int], dict] = {}
    for csv_path in sorted(report_dir.glob('*/episodes.csv')):
        stage = csv_path.parent.name
        with csv_path.open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                key = (stage, int(row['seed']))
                if key in result:
                    raise ValueError(f'duplicate episode row {key}')
                row = dict(row)
                for field in ('success', 'collision', 'out_of_road', 'truncated'):
                    row[field] = _bool(row[field])
                result[key] = row
    if not result:
        raise ValueError(f'No episode rows found below {report_dir}')
    return result


def _paired(base: dict, candidate: dict, good) -> dict:
    if set(base) != set(candidate):
        raise ValueError('Reports are not paired on identical stage/seed scenarios')
    improved = regressed = both_good = both_bad = 0
    for key in sorted(base):
        b, c = bool(good(base[key])), bool(good(candidate[key]))
        if not b and c:
            improved += 1
        elif b and not c:
            regressed += 1
        elif b and c:
            both_good += 1
        else:
            both_bad += 1
    return {
        'improved': improved,
        'regressed': regressed,
        'both_good': both_good,
        'both_bad': both_bad,
        'exact_mcnemar_p_two_sided': exact_mcnemar_p(improved, regressed),
    }


def _subset(metrics: dict) -> dict:
    fields = (
        'success_rate', 'collision_rate', 'out_of_road_rate', 'timeout_rate',
        'mean_route_completion', 'avg_speed_mps', 'lane_abs_error_m',
        'stall_time_fraction', 'jerk_mean_abs_mps3',
        'safety_intervention_time_fraction', 'safety_mean_longitudinal_correction',
        'route_assist_intervention_time_fraction', 'route_assist_mean_steering_correction',
    )
    return {key: metrics.get(key) for key in fields if key in metrics}


def _stage_summary(report: dict) -> dict:
    output = {}
    for stage in report['stages']:
        metrics = stage['metrics']
        output[stage['spec']['name']] = {
            'success_rate': metrics['success_rate'],
            'collision_rate': metrics['collision_rate'],
            'out_of_road_rate': metrics['out_of_road_rate'],
            'timeout_rate': metrics['timeout_rate'],
            'catastrophic_rate': metrics['collision_rate'] + metrics['out_of_road_rate'],
            'mean_route_completion': metrics['mean_route_completion'],
        }
    return output


def _suite_signature(report: dict):
    return tuple(
        (
            stage['name'], stage['metadrive_map'], float(stage['traffic_density']),
            float(stage['target_speed']), int(stage['max_steps']), tuple(stage['seeds']),
        )
        for stage in report['suite']
    )


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def build(root: Path, output: Path) -> dict:
    root, output = Path(root), Path(output)
    validation_raw = _load(root / 'validation8_source/suite_report.json')
    validation_route = _load(root / 'validation8_route/suite_report.json')
    validation_hybrid = _load(root / 'validation8_route_safety/suite_report.json')
    release_raw = _load(root / 'release12_raw/suite_report.json')
    release_hybrid = _load(root / 'release12_hybrid/suite_report.json')
    freeze_path = root / 'release12_freeze.json'
    freeze = _load(freeze_path)

    validation_signatures = {_suite_signature(x) for x in (validation_raw, validation_route, validation_hybrid)}
    release_signatures = {_suite_signature(x) for x in (release_raw, release_hybrid)}
    if len(validation_signatures) != 1 or len(release_signatures) != 1:
        raise ValueError('Each comparison must use an identical capability suite')
    validation_seeds = [s for stage in next(iter(validation_signatures)) for s in stage[-1]]
    release_seeds = [s for stage in next(iter(release_signatures)) for s in stage[-1]]
    if validation_seeds != list(range(26000, 26015)):
        raise ValueError('Expected validation8 slice 26000-26014')
    if release_seeds != list(range(27000, 27015)):
        raise ValueError('Expected release12 slice 27000-27014')
    if freeze.get('release12_used_for_tuning') is not False or freeze.get('release_split') != 'release12':
        raise ValueError('Missing valid pre-release12 freeze declaration')

    # The explicit freeze must predate the first release request and every frozen
    # source hash must still match the release implementation.
    first_requests = list((root / 'release12_raw').glob('*/stage_request.json')) + list((root / 'release12_hybrid').glob('*/stage_request.json'))
    if not first_requests:
        raise ValueError('Missing release12 stage request evidence')
    freeze_mtime = freeze_path.stat().st_mtime_ns
    first_release_request = min(path.stat().st_mtime_ns for path in first_requests)
    freeze_predates_release = freeze_mtime < first_release_request
    source_hashes_ok = True
    source_hash_observed = {}
    for rel, expected in freeze['source_files_sha256'].items():
        observed = sha256(_bootstrap.ROOT / rel)
        source_hash_observed[rel] = observed
        source_hashes_ok &= observed == expected
    if not freeze_predates_release or not source_hashes_ok:
        raise ValueError('release12 freeze chronology or frozen source hashes failed')

    raw_rows = _episode_map(root / 'release12_raw')
    hybrid_rows = _episode_map(root / 'release12_hybrid')
    paired_success = _paired(raw_rows, hybrid_rows, lambda row: row['success'])
    paired_noncat = _paired(raw_rows, hybrid_rows,
                            lambda row: not row['collision'] and not row['out_of_road'])

    rm, hm = release_raw['metrics'], release_hybrid['metrics']
    raw_stage, hybrid_stage = _stage_summary(release_raw), _stage_summary(release_hybrid)
    stage_delta = {
        name: {
            'success_change_pp': 100 * (hybrid_stage[name]['success_rate'] - raw_stage[name]['success_rate']),
            'catastrophic_change_pp': 100 * (hybrid_stage[name]['catastrophic_rate'] - raw_stage[name]['catastrophic_rate']),
            'route_completion_change_pp': 100 * (hybrid_stage[name]['mean_route_completion'] - raw_stage[name]['mean_route_completion']),
        }
        for name in raw_stage
    }

    summary = {
        'version': '2.9.0',
        'release_name': 'Transparent Hybrid Route-Control Research Release',
        'learned_model_promoted': False,
        'frozen_learned_model': freeze['systems']['raw_v28']['model'],
        'frozen_learned_model_sha256': freeze['systems']['raw_v28']['model_sha256'],
        'system': {
            'name': 'v2.9 transparent hybrid route + longitudinal safety',
            'neural_policy': freeze['systems']['raw_v28']['model'],
            'lateral_control': 'pure-pursuit route assist using policy-visible route preview',
            'route_assist_blend': freeze['systems']['hybrid_v29']['route_blend'],
            'longitudinal_control': 'neural policy with independent SafetySupervisor correction',
            'safety_ttc_threshold_s': freeze['systems']['hybrid_v29']['ttc_threshold_s'],
            'safety_risk_threshold': freeze['systems']['hybrid_v29']['risk_threshold'],
            'runtime_teacher': False,
            'claim_boundary': 'This is a hybrid neural+rule system; route-assist results are never raw learned-policy results.',
        },
        'route_observation_v4': {
            'implemented': True,
            'description': 'forward ordered route preview sampled by route arc length instead of legacy ego-frame x ordering',
            'legacy_schema3_default_preserved': True,
        },
        'learned_lateral_experiments': {
            'status': 'negative_not_promoted',
            'experiments': [
                'experiments/v2_9/route_schema4_expert_small',
                'experiments/v2_9/route_schema4_steer4',
                'experiments/v2_9/route_schema4_dagger35',
                'experiments/v2_9/route_head_ridge',
            ],
            'interpretation': 'Schema-v4 expert/DAgger/ridge-head variants did not reliably solve raw closed-loop curve/ramp failures on validation8; none is curated or used in release12.',
        },
        'selection': {
            'domain': 'validation8 seeds 26000-26014 (3 per capability stage)',
            'release12_used_for_selection': False,
            'raw': _subset(validation_raw['metrics']),
            'route_assist_only': _subset(validation_route['metrics']),
            'hybrid_route_plus_safety': _subset(validation_hybrid['metrics']),
            'decision': 'promote transparent hybrid system configuration; do not promote a new learned checkpoint',
        },
        'pre_release_freeze': {
            'path': str(freeze_path.relative_to(_bootstrap.ROOT)),
            'sha256': sha256(freeze_path),
            'freeze_predates_first_release_request': freeze_predates_release,
            'frozen_source_hashes_match': source_hashes_ok,
            'first_release_request_mtime_ns': first_release_request,
            'freeze_mtime_ns': freeze_mtime,
            'source_hashes_observed': source_hash_observed,
        },
        'release12': {
            'slice': '27000-27014 evaluated; domain reserved 27000-27199',
            'scenarios': 15,
            'consumed': True,
            'raw': _subset(rm),
            'hybrid': _subset(hm),
            'success_change_pp': 100 * (hm['success_rate'] - rm['success_rate']),
            'collision_change_pp': 100 * (hm['collision_rate'] - rm['collision_rate']),
            'offroad_change_pp': 100 * (hm['out_of_road_rate'] - rm['out_of_road_rate']),
            'catastrophic_change_pp': 100 * ((hm['collision_rate'] + hm['out_of_road_rate']) - (rm['collision_rate'] + rm['out_of_road_rate'])),
            'route_completion_change_pp': 100 * (hm['mean_route_completion'] - rm['mean_route_completion']),
            'paired_success': paired_success,
            'paired_noncatastrophic': paired_noncat,
            'raw_by_stage': raw_stage,
            'hybrid_by_stage': hybrid_stage,
            'stage_deltas': stage_delta,
        },
        'interpretation': (
            'The fresh release12 result supports a large finite-simulation system-level improvement for the '
            'transparent hybrid controller. It does not demonstrate a safer raw neural policy: route assistance '
            'replaces/blends steering on almost every step and longitudinal SafetySupervisor also intervenes. '
            'The learned v2.8 model is unchanged.'
        ),
        'future_tuning_rule': 'release12 is consumed. Future tuning must use fresh validation9/release13 or later domains.',
        'safety_boundary': 'Research simulation only; not a safety certificate, real-road risk estimate, or real-vehicle controller.',
    }
    output.mkdir(parents=True, exist_ok=True)
    save_json(summary, output / 'comparison.json')
    save_json({
        'version': '2.9.0',
        'status': 'metadrive_hybrid_research_release_not_safety_certified',
        'release12_consumed': True,
        'learned_model_promoted': False,
        'pre_release_freeze_verified': freeze_predates_release and source_hashes_ok,
        'raw_success_rate': rm['success_rate'],
        'hybrid_success_rate': hm['success_rate'],
        'raw_collision_rate': rm['collision_rate'],
        'hybrid_collision_rate': hm['collision_rate'],
        'raw_out_of_road_rate': rm['out_of_road_rate'],
        'hybrid_out_of_road_rate': hm['out_of_road_rate'],
        'hybrid_route_assist_intervention_time_fraction': hm.get('route_assist_intervention_time_fraction'),
        'hybrid_safety_intervention_time_fraction': hm.get('safety_intervention_time_fraction'),
        'paired_success': paired_success,
        'claim_boundary': summary['system']['claim_boundary'],
    }, output / 'RELEASE_STATUS.json')

    lines = [
        '# v2.9 Transparent Hybrid Route-Control Research Release', '',
        'v2.9 does **not** promote a new learned driving checkpoint. Several schema-v4 / lateral-distillation '
        'experiments were executed and retained as negative evidence because they did not reliably solve raw '
        'closed-loop curve/ramp failures. The released system therefore makes its rule components explicit:', '',
        '```text', 'Frozen v2.8 neural actor', '      │', '      ├─ steering → pure-pursuit RouteAssist (policy-visible route preview)',
        '      └─ longitudinal → SafetySupervisor when risk requires correction', '                        ↓', '                  MetaDrive 0.4.3', '```', '',
        'The route layer is not hidden inside the neural policy, and every route/safety intervention is recorded.', '',
        '## Validation8 selection (26000-26014)', '',
        '| System | Success | Collision | Off-road | Route completion | Lane error | Route intervention | Safety intervention |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for label, report in [('raw v2.8', validation_raw), ('+ route assist', validation_route), ('+ route + safety', validation_hybrid)]:
        m = report['metrics']
        lines.append(
            f"| {label} | {m['success_rate']:.1%} | {m['collision_rate']:.1%} | {m['out_of_road_rate']:.1%} | "
            f"{m['mean_route_completion']:.1%} | {m['lane_abs_error_m']:.3f} m | "
            f"{m.get('route_assist_intervention_time_fraction',0):.1%} | {m.get('safety_intervention_time_fraction',0):.1%} |"
        )
    lines += ['', 'The transparent hybrid was frozen after validation8. `release12_freeze.json` was written **before** '
                   'any release12 stage request, and all frozen source hashes still match.', '',
              '## Fresh release12 (27000-27014)', '',
              '| System | Success | Collision | Off-road | Catastrophic | Route completion | Lane error | Route intervention | Safety intervention |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
              f"| raw v2.8 | {rm['success_rate']:.1%} | {rm['collision_rate']:.1%} | {rm['out_of_road_rate']:.1%} | {(rm['collision_rate']+rm['out_of_road_rate']):.1%} | {rm['mean_route_completion']:.1%} | {rm['lane_abs_error_m']:.3f} m | 0.0% | 0.0% |",
              f"| v2.9 hybrid | {hm['success_rate']:.1%} | {hm['collision_rate']:.1%} | {hm['out_of_road_rate']:.1%} | {(hm['collision_rate']+hm['out_of_road_rate']):.1%} | {hm['mean_route_completion']:.1%} | {hm['lane_abs_error_m']:.3f} m | {hm['route_assist_intervention_time_fraction']:.1%} | {hm['safety_intervention_time_fraction']:.1%} |", '',
              f"Paired success: **{paired_success['improved']} improved / {paired_success['regressed']} regressed**, exact McNemar p={paired_success['exact_mcnemar_p_two_sided']:.4f}. "
              f"Aggregate success changes {summary['release12']['success_change_pp']:+.1f}pp and collision-or-off-road {summary['release12']['catastrophic_change_pp']:+.1f}pp.", '',
              '### Per capability', '',
              '| Stage | Raw success | Hybrid success | Raw catastrophic | Hybrid catastrophic | Route completion Δ |',
              '|---|---:|---:|---:|---:|---:|']
    for stage in raw_stage:
        b, c, d = raw_stage[stage], hybrid_stage[stage], stage_delta[stage]
        lines.append(f"| {stage} | {b['success_rate']:.1%} | {c['success_rate']:.1%} | {b['catastrophic_rate']:.1%} | {c['catastrophic_rate']:.1%} | {d['route_completion_change_pp']:+.1f}pp |")
    lines += ['',
              '## Interpretation boundary', '',
              f"- RouteAssist intervenes on **{hm['route_assist_intervention_time_fraction']:.1%}** of evaluated time steps; this is therefore a rule-dominant lateral system result, not raw-policy performance.",
              f"- SafetySupervisor intervenes on **{hm['safety_intervention_time_fraction']:.1%}** of evaluated time steps and only changes longitudinal control.",
              '- The learned v2.8 checkpoint is unchanged; no failed v2.9 lateral student was added to the curated learned-model registry.',
              '- 15 fresh simulated scenarios are finite research evidence, not a safety certificate or real-road risk estimate.',
              '- release12 is consumed and must not be used for further tuning. Use new validation9/release13 domains.', '']
    (output / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(_bootstrap.ROOT / 'reports/v2_9'))
    parser.add_argument('--output', default=str(_bootstrap.ROOT / 'reports/v2_9'))
    args = parser.parse_args()
    print(json.dumps(build(Path(args.root), Path(args.output)), indent=2))
