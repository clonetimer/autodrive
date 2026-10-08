"""Render the v2.6 validation-selected / release5 safety-suite comparison.

The script refuses to compare mismatched release suites and records that the
selected candidate was chosen on validation-only seeds before release5 was
opened.  Release5 outcomes are diagnostic evidence only: they cannot re-rank
or promote a model.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import _bootstrap
from src.utils.logger import save_json


def exact_mcnemar_p(improved: int, regressed: int) -> float:
    n = int(improved) + int(regressed)
    if n <= 0:
        return 1.0
    k = min(int(improved), int(regressed))
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def _release_signature(report: dict) -> tuple:
    signature = []
    for stage in report.get('suite', []):
        signature.append((stage['name'], stage['metadrive_map'], float(stage['traffic_density']),
                          float(stage['target_speed']), int(stage['max_steps']), tuple(stage['seeds'])))
    if not signature:
        raise ValueError('Missing safety-suite specification')
    return tuple(signature)


def _episode_map(report: dict) -> dict[int, dict]:
    rows = report.get('episodes', [])
    mapped = {int(row['seed']): row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError('Expected exactly one episode per release5 seed')
    return mapped


def _binary_transitions(base: dict[int, dict], candidate: dict[int, dict], key) -> dict:
    seeds = sorted(base)
    if seeds != sorted(candidate):
        raise ValueError('Paired reports must contain the same seeds')
    improved = regressed = both_good = both_bad = 0
    for seed in seeds:
        b_good = bool(key(base[seed]))
        c_good = bool(key(candidate[seed]))
        if not b_good and c_good:
            improved += 1
        elif b_good and not c_good:
            regressed += 1
        elif b_good and c_good:
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


def _subset_metrics(metrics: dict, supervised: bool = False) -> dict:
    keys = ['success_rate', 'collision_rate', 'out_of_road_rate', 'timeout_rate',
            'mean_route_completion', 'avg_speed_mps', 'lane_abs_error_m', 'jerk_mean_abs_mps3']
    if supervised:
        keys.extend(['safety_intervention_time_fraction', 'safety_interventions_total',
                     'safety_mean_longitudinal_correction', 'safety_peak_longitudinal_correction',
                     'safety_mean_severity'])
    return {key: metrics.get(key) for key in keys}


def build(root: Path, selection_path: Path, output: Path) -> dict:
    names = ['release5_base_raw', 'release5_selected_raw',
             'release5_base_supervised', 'release5_selected_supervised']
    reports = {name: json.loads((root / name / 'suite_report.json').read_text(encoding='utf-8'))
               for name in names}
    signatures = {_release_signature(report) for report in reports.values()}
    if len(signatures) != 1:
        raise ValueError('All release5 reports must use the exact same frozen safety suite')
    signature = next(iter(signatures))
    seeds = [seed for stage in signature for seed in stage[-1]]
    expected = list(range(9000, 9040))
    if seeds != expected:
        raise ValueError(f'Expected first frozen release5 slice {expected[0]}-{expected[-1]}, got {seeds}')

    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if selection.get('selection_domain') != 'validation_seeds_1000_1199_only':
        raise ValueError('Candidate selection is not validation-only')
    validation_seeds = [int(seed) for stage in selection.get('suite_signature', []) for seed in stage[-1]]
    if not validation_seeds or any(seed < 1000 or seed > 1199 for seed in validation_seeds):
        raise ValueError('Candidate selection contains a non-validation seed')
    if set(validation_seeds) & set(seeds):
        raise ValueError('Validation selection and release5 must be disjoint')
    promotion = selection.get('promotion')
    if not isinstance(promotion, dict):
        raise ValueError('Selection report must contain the predeclared promotion decision')

    base_raw = _episode_map(reports['release5_base_raw'])
    selected_raw = _episode_map(reports['release5_selected_raw'])
    base_sup = _episode_map(reports['release5_base_supervised'])
    selected_sup = _episode_map(reports['release5_selected_supervised'])

    success_pairs = _binary_transitions(base_raw, selected_raw, lambda row: row.get('success'))
    # For catastrophic outcomes, "good" means no collision and no off-road.
    catastrophic_pairs = _binary_transitions(
        base_raw, selected_raw,
        lambda row: not bool(row.get('collision')) and not bool(row.get('out_of_road')),
    )
    intervention_deltas = [selected_sup[s]['safety_intervention_time_fraction'] -
                           base_sup[s]['safety_intervention_time_fraction'] for s in seeds]

    bm = reports['release5_base_raw']['metrics']
    cm = reports['release5_selected_raw']['metrics']
    bsm = reports['release5_base_supervised']['metrics']
    csm = reports['release5_selected_supervised']['metrics']
    base_cat = sum(bool(base_raw[s].get('collision')) or bool(base_raw[s].get('out_of_road')) for s in seeds) / len(seeds)
    cand_cat = sum(bool(selected_raw[s].get('collision')) or bool(selected_raw[s].get('out_of_road')) for s in seeds) / len(seeds)

    stages = {}
    for report_name, report in reports.items():
        stages[report_name] = {
            row['spec']['name']: {
                'success_rate': row['metrics']['success_rate'],
                'collision_rate': row['metrics']['collision_rate'],
                'out_of_road_rate': row['metrics']['out_of_road_rate'],
                'timeout_rate': row['metrics']['timeout_rate'],
                'mean_route_completion': row['metrics']['mean_route_completion'],
                'safety_intervention_time_fraction': row['metrics'].get('safety_intervention_time_fraction', 0.0),
            }
            for row in report.get('stages', [])
        }

    summary = {
        'version': '2.6.0',
        'selection': {
            'selected_candidate': selection['selected'],
            'selection_domain': selection['selection_domain'],
            'validation_seeds': validation_seeds,
            'promotion': promotion,
            'release5_not_used_for_selection_or_promotion': True,
        },
        'release5': {'slice': '9000-9039', 'seeds': seeds, 'scenarios': len(seeds)},
        'raw_policy': {
            'base': _subset_metrics(bm),
            'selected': _subset_metrics(cm),
            'base_catastrophic_rate': base_cat,
            'selected_catastrophic_rate': cand_cat,
            'success_change_pp': 100.0 * (cm['success_rate'] - bm['success_rate']),
            'collision_change_pp': 100.0 * (cm['collision_rate'] - bm['collision_rate']),
            'offroad_change_pp': 100.0 * (cm['out_of_road_rate'] - bm['out_of_road_rate']),
            'catastrophic_change_pp': 100.0 * (cand_cat - base_cat),
            'paired_success': success_pairs,
            'paired_noncatastrophic': catastrophic_pairs,
        },
        'supervised_system': {
            'base': _subset_metrics(bsm, True),
            'selected': _subset_metrics(csm, True),
            'intervention_fraction_relative_change': (
                csm['safety_intervention_time_fraction'] / bsm['safety_intervention_time_fraction'] - 1.0
                if bsm['safety_intervention_time_fraction'] else None),
            'mean_correction_relative_change': (
                csm['safety_mean_longitudinal_correction'] / bsm['safety_mean_longitudinal_correction'] - 1.0
                if bsm['safety_mean_longitudinal_correction'] else None),
            'scenarios_with_lower_intervention_fraction': sum(delta < 0 for delta in intervention_deltas),
            'scenarios_with_higher_intervention_fraction': sum(delta > 0 for delta in intervention_deltas),
        },
        'stages': stages,
        'release5_consumed': True,
        'future_tuning_rule': 'release5 is now observed and must never be used for future model tuning or selection',
        'claim_boundary': (
            'The validation winner did not pass the predeclared promotion gate. Release5 shows only a marginal raw-policy '
            'difference: lower collision is partly exchanged for off-road failure, with one paired improvement and no '
            'paired regression in success/non-catastrophic outcome. The longitudinal supervisor reduces collisions but '
            'cannot repair lateral/path-following failures. This is diagnostic research evidence, not a safety claim.'),
    }
    output.mkdir(parents=True, exist_ok=True)
    save_json(summary, output / 'comparison.json')

    p = promotion
    lines = [
        '# v2.6 release5 frozen safety-suite comparison', '',
        f"Candidate `{selection['selected']}` was selected on validation seeds 1000-1019. "
        f"The predeclared promotion gate returned **{p['promoted']}** before release5 was opened. "
        'The release5 results below are diagnostic only and may not alter that decision.', '',
        '| System | Success | Collision | Off-road | Timeout | Route completion | Supervisor intervention |',
        '|---|---:|---:|---:|---:|---:|---:|',
        f"| v2.5 base raw | {bm['success_rate']:.1%} | {bm['collision_rate']:.1%} | {bm['out_of_road_rate']:.1%} | {bm['timeout_rate']:.1%} | {bm['mean_route_completion']:.1%} | 0% |",
        f"| v2.6 selected raw | {cm['success_rate']:.1%} | {cm['collision_rate']:.1%} | {cm['out_of_road_rate']:.1%} | {cm['timeout_rate']:.1%} | {cm['mean_route_completion']:.1%} | 0% |",
        f"| v2.5 base + longitudinal supervisor | {bsm['success_rate']:.1%} | {bsm['collision_rate']:.1%} | {bsm['out_of_road_rate']:.1%} | {bsm['timeout_rate']:.1%} | {bsm['mean_route_completion']:.1%} | {bsm['safety_intervention_time_fraction']:.1%} |",
        f"| v2.6 selected + longitudinal supervisor | {csm['success_rate']:.1%} | {csm['collision_rate']:.1%} | {csm['out_of_road_rate']:.1%} | {csm['timeout_rate']:.1%} | {csm['mean_route_completion']:.1%} | {csm['safety_intervention_time_fraction']:.1%} |",
        '',
        f"Raw-policy collision changed by {summary['raw_policy']['collision_change_pp']:+.1f} pp, but off-road changed by "
        f"{summary['raw_policy']['offroad_change_pp']:+.1f} pp and collision-or-off-road changed by "
        f"{summary['raw_policy']['catastrophic_change_pp']:+.1f} pp. Paired success: "
        f"{success_pairs['improved']} improved / {success_pairs['regressed']} regressed, exact McNemar "
        f"p={success_pairs['exact_mcnemar_p_two_sided']:.4f}. This finite test does not establish statistical safety improvement.", '',
        f"Under the same longitudinal supervisor, intervention time changed from "
        f"{bsm['safety_intervention_time_fraction']:.1%} to {csm['safety_intervention_time_fraction']:.1%}; "
        f"{summary['supervised_system']['scenarios_with_lower_intervention_fraction']}/40 scenarios required less intervention. "
        f"However, supervised success remained {csm['success_rate']:.1%}: dense traffic mostly timed out while curve/mixed stages mostly failed laterally.", '',
        '## Promotion decision', '',
        f"- promoted: **{p['promoted']}**",
        f"- reasons: {', '.join(p['reasons']) if p['reasons'] else 'none'}",
        f"- validation success gain: {p['deltas']['success_gain']:+.1%}",
        f"- validation catastrophic reduction: {p['deltas']['catastrophic_reduction']:+.1%}",
        f"- validation off-road increase: {p['deltas']['offroad_increase']:+.1%}",
        '',
        '## Boundary', '', summary['claim_boundary'], '',
        '**Release hygiene:** release5 is now consumed. All future tuning must use new training/validation data and a new untouched release split.', ''
    ]
    (output / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')

    try:
        import matplotlib.pyplot as plt
        labels = ['Base raw', 'Selected raw', 'Base + supervisor', 'Selected + supervisor']
        success = [bm['success_rate'], cm['success_rate'], bsm['success_rate'], csm['success_rate']]
        collision = [bm['collision_rate'], cm['collision_rate'], bsm['collision_rate'], csm['collision_rate']]
        offroad = [bm['out_of_road_rate'], cm['out_of_road_rate'], bsm['out_of_road_rate'], csm['out_of_road_rate']]
        x = list(range(len(labels))); width = .24
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar([v-width for v in x], success, width, label='Success')
        ax.bar(x, collision, width, label='Collision')
        ax.bar([v+width for v in x], offroad, width, label='Off-road')
        ax.set_xticks(x, labels, rotation=12)
        ax.set_ylim(0, 1.05); ax.set_ylabel('Rate')
        ax.set_title('MetaDrive release5: validation-selected candidate diagnostic')
        ax.legend(); fig.tight_layout()
        figure = output / 'safety_suite_release5.png'
        fig.savefig(figure, dpi=160); plt.close(fig)
        summary['figure'] = figure.name
    except ImportError:
        summary['figure'] = None
    save_json(summary, output / 'comparison.json')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=str(_bootstrap.ROOT / 'reports/v2_6'))
    p.add_argument('--selection', default=str(_bootstrap.ROOT / 'reports/v2_6/candidate_selection.json'))
    p.add_argument('--output', default=str(_bootstrap.ROOT / 'reports/v2_6'))
    args = p.parse_args()
    print(json.dumps(build(Path(args.root), Path(args.selection), Path(args.output)), indent=2))
