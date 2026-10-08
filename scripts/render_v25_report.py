"""Build the v2.5 paired safety-distillation report from frozen evaluation JSONs."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import _bootstrap
from src.utils.logger import save_json


def exact_mcnemar_p(improved: int, regressed: int) -> float:
    n = improved + regressed
    if n == 0:
        return 1.0
    k = min(improved, regressed)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def build(root: Path, output: Path):
    names = ['base_raw', 'base_supervised', 'curated_raw', 'curated_supervised']
    reports = {name: json.loads((root / name / 'evaluation.json').read_text()) for name in names}
    seed_lists = [tuple(report['seeds']) for report in reports.values()]
    if len(set(seed_lists)) != 1:
        raise ValueError('v2.5 paired report requires identical ordered scenario seeds')
    seeds = list(seed_lists[0])
    if len(seeds) != 40 or seeds != list(range(7000, 7040)):
        raise ValueError('Expected the frozen release4 7000-7039 scenario set')

    base_ep = {row['seed']: row for row in reports['base_raw']['episodes']}
    dist_ep = {row['seed']: row for row in reports['curated_raw']['episodes']}
    improved = sum(not base_ep[s]['success'] and dist_ep[s]['success'] for s in seeds)
    regressed = sum(base_ep[s]['success'] and not dist_ep[s]['success'] for s in seeds)
    unchanged_success = sum(base_ep[s]['success'] and dist_ep[s]['success'] for s in seeds)
    unchanged_failure = sum(not base_ep[s]['success'] and not dist_ep[s]['success'] for s in seeds)

    base_sup = {row['seed']: row for row in reports['base_supervised']['episodes']}
    dist_sup = {row['seed']: row for row in reports['curated_supervised']['episodes']}
    intervention_deltas = [dist_sup[s]['safety_intervention_time_fraction'] -
                           base_sup[s]['safety_intervention_time_fraction'] for s in seeds]
    bm = reports['base_raw']['metrics']; dm = reports['curated_raw']['metrics']
    bsm = reports['base_supervised']['metrics']; dsm = reports['curated_supervised']['metrics']
    summary = {
        'version': '2.5.0', 'split': 'release4', 'seeds': seeds,
        'selection_note': 'candidate was selected on validation 1000-1009 before release4 was evaluated',
        'raw_policy': {
            'base': {k: bm[k] for k in ['success_rate','collision_rate','avg_speed_mps','mean_route_completion','jerk_mean_abs_mps3']},
            'distilled': {k: dm[k] for k in ['success_rate','collision_rate','avg_speed_mps','mean_route_completion','jerk_mean_abs_mps3']},
            'success_rate_change_pp': 100 * (dm['success_rate'] - bm['success_rate']),
            'collision_rate_change_pp': 100 * (dm['collision_rate'] - bm['collision_rate']),
            'paired_success_transitions': {'improved': improved, 'regressed': regressed,
                'success_both': unchanged_success, 'failure_both': unchanged_failure},
            'paired_exact_mcnemar_p_two_sided': exact_mcnemar_p(improved, regressed),
        },
        'supervised_system': {
            'base': {k: bsm[k] for k in ['success_rate','collision_rate','avg_speed_mps','safety_intervention_time_fraction',
                                          'safety_interventions_total','safety_mean_longitudinal_correction','safety_mean_severity']},
            'distilled': {k: dsm[k] for k in ['success_rate','collision_rate','avg_speed_mps','safety_intervention_time_fraction',
                                              'safety_interventions_total','safety_mean_longitudinal_correction','safety_mean_severity']},
            'intervention_fraction_change_pp': 100 * (dsm['safety_intervention_time_fraction'] - bsm['safety_intervention_time_fraction']),
            'intervention_fraction_relative_change': (dsm['safety_intervention_time_fraction'] /
                                                       bsm['safety_intervention_time_fraction'] - 1),
            'mean_correction_relative_change': (dsm['safety_mean_longitudinal_correction'] /
                                                bsm['safety_mean_longitudinal_correction'] - 1),
            'seeds_with_lower_intervention_fraction': sum(delta < 0 for delta in intervention_deltas),
            'seeds_with_higher_intervention_fraction': sum(delta > 0 for delta in intervention_deltas),
        },
        'claim_boundary': ('Distillation reduced supervisor dependence and raw collision rate on this finite holdout, '
                           'but the raw policy still collided in 40% of scenarios; this is not a safety guarantee.'),
    }
    output.mkdir(parents=True, exist_ok=True)
    save_json(summary, output / 'comparison.json')

    lines = [
        '# v2.5 release4 paired safety-distillation comparison', '',
        '| System | Success | Collision | Speed m/s | Supervisor intervention |',
        '|---|---:|---:|---:|---:|',
        f"| Base raw PPO | {bm['success_rate']:.1%} | {bm['collision_rate']:.1%} | {bm['avg_speed_mps']:.2f} | 0% |",
        f"| Distilled raw PPO | {dm['success_rate']:.1%} | {dm['collision_rate']:.1%} | {dm['avg_speed_mps']:.2f} | 0% |",
        f"| Base PPO + supervisor | {bsm['success_rate']:.1%} | {bsm['collision_rate']:.1%} | {bsm['avg_speed_mps']:.2f} | {bsm['safety_intervention_time_fraction']:.1%} |",
        f"| Distilled PPO + supervisor | {dsm['success_rate']:.1%} | {dsm['collision_rate']:.1%} | {dsm['avg_speed_mps']:.2f} | {dsm['safety_intervention_time_fraction']:.1%} |",
        '',
        f"Paired raw-policy outcomes: {improved} scenarios improved, {regressed} regressed, "
        f"{unchanged_success} succeeded under both and {unchanged_failure} failed under both. "
        f"The descriptive exact McNemar p-value is {summary['raw_policy']['paired_exact_mcnemar_p_two_sided']:.4f}; "
        'the 40-scenario holdout is too small for a safety claim.', '',
        f"The supervised system remained 0-collision on this holdout while intervention time fell from "
        f"{bsm['safety_intervention_time_fraction']:.1%} to {dsm['safety_intervention_time_fraction']:.1%}. "
        f"Intervention fraction fell on {summary['supervised_system']['seeds_with_lower_intervention_fraction']}/40 scenarios.", '',
        '**Boundary:** ' + summary['claim_boundary'], ''
    ]
    (output / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')

    try:
        import matplotlib.pyplot as plt
        labels = ['Base raw', 'Distilled raw', 'Base + supervisor', 'Distilled + supervisor']
        success = [bm['success_rate'], dm['success_rate'], bsm['success_rate'], dsm['success_rate']]
        collision = [bm['collision_rate'], dm['collision_rate'], bsm['collision_rate'], dsm['collision_rate']]
        x = list(range(len(labels)))
        width = .36
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar([v - width/2 for v in x], success, width, label='Success rate')
        ax.bar([v + width/2 for v in x], collision, width, label='Collision rate')
        ax.set_xticks(x, labels, rotation=12)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel('Rate')
        ax.set_title('MetaDrive release4: safety-distillation paired comparison')
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / 'safety_distillation_release4.png', dpi=160)
        plt.close(fig)
        summary['figure'] = 'safety_distillation_release4.png'
        save_json(summary, output / 'comparison.json')
    except ImportError:
        summary['figure'] = None
        save_json(summary, output / 'comparison.json')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=str(_bootstrap.ROOT / 'reports/v2_5/release4'))
    p.add_argument('--output', default=str(_bootstrap.ROOT / 'reports/v2_5'))
    args = p.parse_args()
    print(json.dumps(build(Path(args.root), Path(args.output)), indent=2))
