"""Build and audit the v3.0 adaptive shared-control release evidence.

This script is deliberately post-evaluation: the release13 freeze manifest was
written before any release13 stage request. The report verifies frozen hashes,
chronology, seed isolation, paired outcomes and the distinction between raw,
fixed-rule and adaptive shared-control systems.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json
import math
import _bootstrap
from src.utils.logger import save_json, sha256

ROOT = _bootstrap.ROOT
R = ROOT / 'reports/v3_0'


def _load(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _episodes(report: dict) -> dict[int, dict]:
    rows = report['episodes']
    result = {int(row['seed']): row for row in rows}
    if len(result) != len(rows):
        raise ValueError('v3.0 release evidence expects one agent episode per unique scenario seed')
    return result


def _exact_mcnemar(improved: int, regressed: int) -> float:
    n = improved + regressed
    if n == 0:
        return 1.0
    k = min(improved, regressed)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _paired(left: dict, right: dict) -> dict:
    a, b = _episodes(left), _episodes(right)
    if set(a) != set(b):
        raise ValueError('Paired release systems must evaluate identical seeds')
    improved = regressed = both_good = both_bad = 0
    for seed in sorted(a):
        l, r = bool(a[seed]['success']), bool(b[seed]['success'])
        if not l and r: improved += 1
        elif l and not r: regressed += 1
        elif l and r: both_good += 1
        else: both_bad += 1
    return {'improved': improved, 'regressed': regressed, 'both_good': both_good,
            'both_bad': both_bad, 'exact_mcnemar_p_two_sided': _exact_mcnemar(improved, regressed)}


def _catastrophic(metrics: dict) -> float:
    # Collision and out-of-road are mutually exclusive terminal outcomes in the
    # current single-agent MetaDrive wrapper. Keep both components in output so
    # this assumption remains auditable.
    return float(metrics['collision_rate'] + metrics['out_of_road_rate'])


def _stage_table(report: dict) -> dict:
    return {row['spec']['name']: {
        'success_rate': row['metrics']['success_rate'],
        'collision_rate': row['metrics']['collision_rate'],
        'out_of_road_rate': row['metrics']['out_of_road_rate'],
        'catastrophic_rate': _catastrophic(row['metrics']),
        'mean_route_completion': row['metrics']['mean_route_completion'],
        'route_assist_mean_blend': row['metrics'].get('route_assist_mean_blend', 0.0),
        'learned_steering_authority_fraction': row['metrics'].get('learned_steering_authority_fraction', 1.0),
    } for row in report['stages']}


def main():
    selection = _load(R / 'selection_validation9.json')
    freeze = _load(R / 'release13_freeze.json')
    raw = _load(R / 'release13_raw/suite_report.json')
    fixed = _load(R / 'release13_fixed/suite_report.json')
    adaptive = _load(R / 'release13_adaptive/suite_report.json')
    validation_fixed = _load(R / 'validation9_fixed/suite_report.json')
    validation_adaptive = _load(R / 'validation9_adaptive050/suite_report.json')

    if not selection.get('gate_passed') or selection.get('release13_used_for_selection') is not False:
        raise ValueError('Invalid validation9 selection evidence')
    if freeze.get('release_split') != 'release13' or freeze.get('release13_used_for_tuning') is not False:
        raise ValueError('Invalid release13 freeze declaration')
    bad_hashes = {path: {'expected': expected, 'actual': sha256(ROOT / path)}
                  for path, expected in freeze['frozen_files'].items()
                  if not (ROOT / path).is_file() or sha256(ROOT / path) != expected}
    if bad_hashes:
        raise ValueError(f'Frozen source hash mismatch: {bad_hashes}')

    freeze_time = datetime.fromisoformat(freeze['created_at_utc']).timestamp()
    requests = []
    for name in ('release13_raw', 'release13_fixed', 'release13_adaptive'):
        requests.extend((R / name).glob('*/stage_request.json'))
    if len(requests) != 15:
        raise ValueError('Expected exactly 15 release13 stage requests (5 stages x 3 frozen systems)')
    earliest_request = min(path.stat().st_mtime for path in requests)
    chronology_ok = earliest_request >= freeze_time
    if not chronology_ok:
        raise ValueError('release13 request predates the pre-release freeze')

    expected_release = list(range(30000, 30015))
    expected_validation = list(range(29000, 29015))
    for label, report, expected in [
        ('validation_fixed', validation_fixed, expected_validation),
        ('validation_adaptive', validation_adaptive, expected_validation),
        ('release_raw', raw, expected_release), ('release_fixed', fixed, expected_release),
        ('release_adaptive', adaptive, expected_release)]:
        seeds = sorted(int(row['seed']) for row in report['episodes'])
        if seeds != expected:
            raise ValueError(f'{label} seeds differ from frozen expected slice')

    rm, fm, am = raw['metrics'], fixed['metrics'], adaptive['metrics']
    summary = {
        'version': '3.0.0',
        'status': 'adaptive_shared_control_metadrive_research_release_not_safety_certified',
        'validation9': {
            'consumed': True,
            'fixed': validation_fixed['metrics'],
            'adaptive': validation_adaptive['metrics'],
            'selection_gate_passed': selection['gate_passed'],
        },
        'release13': {
            'consumed': True,
            'scenarios': len(expected_release),
            'raw': rm,
            'fixed_v29_semantics': fm,
            'adaptive_v30': am,
            'raw_to_adaptive_paired_success': _paired(raw, adaptive),
            'fixed_to_adaptive_paired_success': _paired(fixed, adaptive),
            'raw_stages': _stage_table(raw),
            'fixed_stages': _stage_table(fixed),
            'adaptive_stages': _stage_table(adaptive),
            'adaptive_vs_fixed': {
                'success_change_pp': 100 * (am['success_rate'] - fm['success_rate']),
                'catastrophic_change_pp': 100 * (_catastrophic(am) - _catastrophic(fm)),
                'route_completion_change_pp': 100 * (am['mean_route_completion'] - fm['mean_route_completion']),
                'rule_blend_reduction_relative': 1.0 - am['route_assist_mean_blend'] / max(fm['route_assist_mean_blend'], 1e-12),
                'learned_steering_authority_fraction': am['learned_steering_authority_fraction'],
            },
        },
        'freeze_audit': {
            'freeze_created_at_utc': freeze['created_at_utc'],
            'earliest_release13_stage_request_mtime': datetime.fromtimestamp(earliest_request).astimezone().isoformat(),
            'chronology_ok': chronology_ok,
            'frozen_file_hashes_ok': True,
            'frozen_file_count': len(freeze['frozen_files']),
        },
        'claim_boundary': (
            'The v3.0 candidate does not change the neural checkpoint. It is a transparent adaptive shared-control '
            'system: policy-visible pure-pursuit contributes a time-varying steering blend capped at 0.5 and the '
            'SafetySupervisor may modify longitudinal control. System metrics are not raw-policy metrics.'),
        'future_tuning_rule': 'validation9 and release13 are consumed. Any further tuning must use validation10/release14 or later fresh domains.'
    }
    save_json(summary, R / 'comparison.json')
    status = {
        'version': '3.0.0', 'status': summary['status'], 'release13_consumed': True,
        'learned_model_promoted': False, 'system_promoted': True,
        'pre_release_freeze_verified': True,
        'release13_raw_success_rate': rm['success_rate'],
        'release13_fixed_success_rate': fm['success_rate'],
        'release13_adaptive_success_rate': am['success_rate'],
        'release13_adaptive_collision_rate': am['collision_rate'],
        'release13_adaptive_out_of_road_rate': am['out_of_road_rate'],
        'release13_adaptive_route_assist_mean_blend': am['route_assist_mean_blend'],
        'release13_adaptive_learned_steering_authority_fraction': am['learned_steering_authority_fraction'],
        'claim_boundary': summary['claim_boundary'],
    }
    save_json(status, R / 'RELEASE_STATUS.json')

    pair = summary['release13']['fixed_to_adaptive_paired_success']
    lines = [
        '# v3.0 Adaptive Shared-Control MetaDrive Research Release', '',
        'v3.0 keeps the frozen v2.8 neural checkpoint and changes only the **transparent system handoff**. '
        'Instead of replacing steering with pure pursuit on every step, adaptive RouteAssist uses only '
        'policy-visible curvature/heading/lane-error features to choose a time-varying blend capped at 0.5.', '',
        '## Validation9 selection (29000-29014)', '',
        '| System | Success | Collision | Off-road | Mean route blend | Learned steering authority |',
        '|---|---:|---:|---:|---:|---:|',
        f"| fixed v2.9 semantics | {validation_fixed['metrics']['success_rate']:.1%} | {validation_fixed['metrics']['collision_rate']:.1%} | {validation_fixed['metrics']['out_of_road_rate']:.1%} | {validation_fixed['metrics']['route_assist_mean_blend']:.1%} | {validation_fixed['metrics']['learned_steering_authority_fraction']:.1%} |",
        f"| **adaptive v3.0** | **{validation_adaptive['metrics']['success_rate']:.1%}** | {validation_adaptive['metrics']['collision_rate']:.1%} | **{validation_adaptive['metrics']['out_of_road_rate']:.1%}** | **{validation_adaptive['metrics']['route_assist_mean_blend']:.1%}** | **{validation_adaptive['metrics']['learned_steering_authority_fraction']:.1%}** |", '',
        '`release13_freeze.json` was written before any release13 stage request and all frozen hashes still match.', '',
        '## Fresh release13 (30000-30014)', '',
        '| System | Success | Collision | Off-road | Route completion | Mean route blend | Learned steering authority |',
        '|---|---:|---:|---:|---:|---:|---:|',
        f"| raw v2.8 actor | {rm['success_rate']:.1%} | {rm['collision_rate']:.1%} | {rm['out_of_road_rate']:.1%} | {rm['mean_route_completion']:.1%} | 0.0% | 100.0% |",
        f"| fixed route+safety | {fm['success_rate']:.1%} | {fm['collision_rate']:.1%} | {fm['out_of_road_rate']:.1%} | {fm['mean_route_completion']:.1%} | {fm['route_assist_mean_blend']:.1%} | {fm['learned_steering_authority_fraction']:.1%} |",
        f"| **adaptive route+safety** | **{am['success_rate']:.1%}** | **{am['collision_rate']:.1%}** | **{am['out_of_road_rate']:.1%}** | {am['mean_route_completion']:.1%} | **{am['route_assist_mean_blend']:.1%}** | **{am['learned_steering_authority_fraction']:.1%}** |", '',
        f"Fixed→adaptive paired success: **{pair['improved']} improved / {pair['regressed']} regressed / {pair['both_good']} both-success / {pair['both_bad']} both-fail**, exact McNemar p={pair['exact_mcnemar_p_two_sided']:.4g}.", '',
        'Aggregate fixed and adaptive success are equal on release13, but adaptive reduces mean geometric steering '
        f"blend from 100% to **{am['route_assist_mean_blend']:.1%}**, giving the learned actor **{am['learned_steering_authority_fraction']:.1%} mean steering authority**. "
        'This is a system-authority result, not evidence that the raw actor learned the missing lateral skill.', '',
        '### Per capability', '',
        '| Stage | Raw success | Fixed success | Adaptive success | Adaptive route blend |',
        '|---|---:|---:|---:|---:|']
    for stage in summary['release13']['adaptive_stages']:
        lines.append(f"| {stage} | {summary['release13']['raw_stages'][stage]['success_rate']:.1%} | {summary['release13']['fixed_stages'][stage]['success_rate']:.1%} | {summary['release13']['adaptive_stages'][stage]['success_rate']:.1%} | {summary['release13']['adaptive_stages'][stage]['route_assist_mean_blend']:.1%} |")
    lines += ['', '## Interpretation boundary', '',
        '- The neural checkpoint is unchanged from v2.8; no v3.0 learned model is promoted.',
        '- Adaptive RouteAssist contributes steering on many steps, although its mean blend is low; this is still hybrid shared control.',
        '- SafetySupervisor remains enabled for the two hybrid systems and can modify longitudinal control.',
        '- release13 contains only 15 MetaDrive scenarios and is not a safety certificate or real-road risk estimate.',
        '- validation9 and release13 are now consumed; future tuning must move to fresh domains.', '']
    (ROOT / 'docs/ADAPTIVE_SHARED_CONTROL_v3.0.0.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
