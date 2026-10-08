"""Build the v2.8 capability-specialization validation/release11 report.

Candidate selection is validation7-only.  release11 is a fresh holdout opened
only after the candidate model and selection JSON already existed.  Because the
explicit freeze-record file was not written until after release execution, the
report calls this a chronology/hash audit rather than pretending a pre-release
freeze manifest existed.
"""
from __future__ import annotations
import argparse
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
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def _episode_map(report: dict) -> dict[tuple[str, int], dict]:
    rows = report.get('episodes', [])
    result = {(str(row['suite_stage']), int(row['seed'])): row for row in rows}
    if not rows or len(result) != len(rows):
        raise ValueError('Expected one unique row per capability stage/seed')
    return result


def _paired(base: dict, candidate: dict, good) -> dict:
    if set(base) != set(candidate):
        raise ValueError('release11 reports are not paired on identical stage/seed scenarios')
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
        'improved': improved, 'regressed': regressed, 'both_good': both_good,
        'both_bad': both_bad, 'exact_mcnemar_p_two_sided': exact_mcnemar_p(improved, regressed),
    }


def _stage_summary(report: dict) -> dict:
    output = {}
    for stage in report['stages']:
        m = stage['metrics']
        output[stage['spec']['name']] = {
            'success_rate': m['success_rate'], 'collision_rate': m['collision_rate'],
            'out_of_road_rate': m['out_of_road_rate'], 'timeout_rate': m['timeout_rate'],
            'catastrophic_rate': m['collision_rate'] + m['out_of_road_rate'],
            'mean_route_completion': m['mean_route_completion'],
        }
    return output


def _metric_subset(m: dict) -> dict:
    return {key: m.get(key) for key in (
        'success_rate','collision_rate','out_of_road_rate','timeout_rate','mean_route_completion',
        'avg_speed_mps','lane_abs_error_m','stall_time_fraction','jerk_mean_abs_mps3',
    )}


def build(root: Path, selection_path: Path, model_path: Path, output: Path) -> dict:
    root, selection_path, model_path, output = map(Path, (root, selection_path, model_path, output))
    base_path = root / 'release11_base/suite_report.json'
    candidate_path = root / 'release11_pureexpert/suite_report.json'
    base = json.loads(base_path.read_text(encoding='utf-8'))
    candidate = json.loads(candidate_path.read_text(encoding='utf-8'))
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    if selection.get('selection_domain') != 'validation7_seeds_22000_22199_only':
        raise ValueError('v2.8 final candidate must be selected only on validation7')
    promotion = selection.get('promotion', {})
    if selection.get('selected') != 'pureexpert' or not promotion.get('promoted'):
        raise ValueError('Expected pureexpert to pass the pre-release validation7 promotion gate')
    signatures = []
    for report in (base, candidate):
        signature = tuple((s['name'], s['metadrive_map'], float(s['traffic_density']), float(s['target_speed']),
                           int(s['max_steps']), tuple(s['seeds'])) for s in report['suite'])
        signatures.append(signature)
    if signatures[0] != signatures[1]:
        raise ValueError('release11 base/candidate do not share an identical capability suite')
    seeds = [seed for stage in signatures[0] for seed in stage[-1]]
    if seeds != list(range(23000, 23025)):
        raise ValueError('Expected frozen release11 slice 23000-23024')

    first_requests = [root/'release11_base/curve_tracking/stage_request.json',
                      root/'release11_pureexpert/curve_tracking/stage_request.json']
    selection_mtime = selection_path.stat().st_mtime_ns
    model_mtime = model_path.stat().st_mtime_ns
    release_start_mtime = min(path.stat().st_mtime_ns for path in first_requests)
    chronology_ok = model_mtime < release_start_mtime and selection_mtime < release_start_mtime
    if not chronology_ok:
        raise ValueError('Selection/model chronology does not predate release11 execution')

    base_rows, candidate_rows = _episode_map(base), _episode_map(candidate)
    paired_success = _paired(base_rows, candidate_rows, lambda row: row.get('success'))
    paired_noncat = _paired(base_rows, candidate_rows,
                            lambda row: not bool(row.get('collision')) and not bool(row.get('out_of_road')))
    bm, cm = base['metrics'], candidate['metrics']
    base_stage, candidate_stage = _stage_summary(base), _stage_summary(candidate)
    stage_delta = {stage: {
        'success_change_pp': 100*(candidate_stage[stage]['success_rate']-base_stage[stage]['success_rate']),
        'catastrophic_change_pp': 100*(candidate_stage[stage]['catastrophic_rate']-base_stage[stage]['catastrophic_rate']),
        'route_completion_change_pp': 100*(candidate_stage[stage]['mean_route_completion']-base_stage[stage]['mean_route_completion']),
    } for stage in base_stage}

    summary = {
        'version': '2.8.0',
        'method': 'pure-expert full-action MetaDrive IDM behavior cloning on capability curriculum',
        'training': {
            'model': str(model_path.relative_to(_bootstrap.ROOT)),
            'model_sha256': sha256(model_path),
            'source_model': 'models/metadrive_ppo_state_expert_distilled_v27.pt',
            'teacher': 'MetaDrive IDMPolicy, training-only privileged full-action teacher',
            'expert_execution_fraction': 1.0,
            'transitions': 9000,
            'unique_training_seeds': 30,
            'capability_train_slice': '13500-13529',
            'deployment_uses_teacher': False,
        },
        'selection': {
            'domain': selection['selection_domain'], 'scenarios': 15,
            'selected': selection['selected'], 'promotion': promotion,
            'selection_sha256': sha256(selection_path),
            'release11_used_for_selection': False,
        },
        'chronology_audit': {
            'model_mtime_ns': model_mtime, 'selection_mtime_ns': selection_mtime,
            'first_release_request_mtime_ns': release_start_mtime,
            'model_and_selection_predate_release11': chronology_ok,
            'formal_freeze_manifest_created_before_release': False,
            'interpretation': 'Hashes and mtimes verify model/selection existed before release11 requests; an explicit freeze JSON was not created until post-run documentation, so no stronger claim is made.',
        },
        'release11': {
            'slice': '23000-23024', 'scenarios': 25, 'consumed': True,
            'base': _metric_subset(bm), 'candidate': _metric_subset(cm),
            'success_change_pp': 100*(cm['success_rate']-bm['success_rate']),
            'collision_change_pp': 100*(cm['collision_rate']-bm['collision_rate']),
            'offroad_change_pp': 100*(cm['out_of_road_rate']-bm['out_of_road_rate']),
            'catastrophic_change_pp': 100*((cm['collision_rate']+cm['out_of_road_rate'])-(bm['collision_rate']+bm['out_of_road_rate'])),
            'route_completion_change_pp': 100*(cm['mean_route_completion']-bm['mean_route_completion']),
            'paired_success': paired_success, 'paired_noncatastrophic': paired_noncat,
            'base_by_stage': base_stage, 'candidate_by_stage': candidate_stage,
            'stage_deltas': stage_delta,
        },
        'interpretation': (
            'release11 confirms a large aggregate capability improvement but not universal stage improvement. '
            'The candidate improves aggregate success/collision/catastrophic rates and route completion, while '
            'curve tracking remains 100% off-road and ramp-merge success regresses on this finite holdout.'
        ),
        'future_tuning_rule': 'release11 is consumed. Any v2.9 tuning must use a new validation/release domain.',
        'claim_boundary': 'Finite MetaDrive simulation evidence only; not a safety certificate, real-road risk estimate, or real-vehicle controller.',
    }
    output.mkdir(parents=True, exist_ok=True)
    save_json(summary, output/'comparison.json')
    save_json(summary['chronology_audit'] | {
        'model_sha256': summary['training']['model_sha256'],
        'selection_sha256': summary['selection']['selection_sha256'],
        'status': 'post_hoc_chronology_and_hash_audit_not_a_pre_release_freeze_manifest',
    }, output/'FREEZE_AUDIT.json')

    lines = [
        '# v2.8 capability-specialization report', '',
        'v2.8 uses MetaDrive `IDMPolicy` as a training-only privileged full-action teacher over five capability stages. '
        'The final candidate uses 100% expert execution during data collection, then deploys only the neural actor.', '',
        '## Validation7 selection', '',
        '| Policy | Success | Collision | Off-road | Catastrophic | Route completion |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in selection['scores']:
        lines.append(f"| {row['name']} | {row['success_rate']:.1%} | {row['collision_rate']:.1%} | {row['out_of_road_rate']:.1%} | {row['catastrophic_rate']:.1%} | {row['mean_route_completion']:.1%} |")
    lines += ['', f"Promotion gate: **{promotion['promoted']}**. release11 was not used for candidate selection.", '',
              '## Fresh release11 aggregate result (23000-23024)', '',
              '| Policy | Success | Collision | Off-road | Catastrophic | Route completion | Avg speed |',
              '|---|---:|---:|---:|---:|---:|---:|',
              f"| v2.7 source | {bm['success_rate']:.1%} | {bm['collision_rate']:.1%} | {bm['out_of_road_rate']:.1%} | {(bm['collision_rate']+bm['out_of_road_rate']):.1%} | {bm['mean_route_completion']:.1%} | {bm['avg_speed_mps']:.2f} m/s |",
              f"| v2.8 capability BC | {cm['success_rate']:.1%} | {cm['collision_rate']:.1%} | {cm['out_of_road_rate']:.1%} | {(cm['collision_rate']+cm['out_of_road_rate']):.1%} | {cm['mean_route_completion']:.1%} | {cm['avg_speed_mps']:.2f} m/s |", '',
              f"Aggregate success changes {100*(cm['success_rate']-bm['success_rate']):+.1f}pp, collision {100*(cm['collision_rate']-bm['collision_rate']):+.1f}pp, off-road {100*(cm['out_of_road_rate']-bm['out_of_road_rate']):+.1f}pp, and collision-or-off-road {summary['release11']['catastrophic_change_pp']:+.1f}pp. Paired success is {paired_success['improved']} improved / {paired_success['regressed']} regressed, exact McNemar p={paired_success['exact_mcnemar_p_two_sided']:.4f}.", '',
              '## Per-capability holdout behavior', '',
              '| Stage | Source success | v2.8 success | Source catastrophic | v2.8 catastrophic | Route completion Δ |',
              '|---|---:|---:|---:|---:|---:|']
    for stage in base_stage:
        b, c, d = base_stage[stage], candidate_stage[stage], stage_delta[stage]
        lines.append(f"| {stage} | {b['success_rate']:.1%} | {c['success_rate']:.1%} | {b['catastrophic_rate']:.1%} | {c['catastrophic_rate']:.1%} | {d['route_completion_change_pp']:+.1f}pp |")
    lines += ['',
              'The aggregate result is materially better, but it is **not uniform**: curve tracking remains 100% off-road and ramp-merge success drops on release11. Bottleneck, intersection and dense-progress outcomes improve. This is a research capability gain, not a safety pass.', '',
              '## Leakage / chronology boundary', '',
              '- validation7 selected and promoted the candidate before release11 requests were created.',
              '- A dedicated pre-release `FROZEN_MODEL.json` was not written; `FREEZE_AUDIT.json` therefore records only a post-hoc hash/mtime chronology audit and does not pretend otherwise.',
              '- release11 is now consumed and cannot be reused for future tuning.',
              '- The privileged IDM teacher is never used by the raw deployed student.',
              '- Simulation artifacts must not control a real vehicle.', '']
    (output/'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    try:
        import matplotlib.pyplot as plt
        stages = list(base_stage)
        x = range(len(stages)); width = .36
        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar([i-width/2 for i in x], [base_stage[s]['success_rate'] for s in stages], width, label='v2.7 source success')
        ax.bar([i+width/2 for i in x], [candidate_stage[s]['success_rate'] for s in stages], width, label='v2.8 capability success')
        ax.set_xticks(list(x), [s.replace('_','\n') for s in stages]); ax.set_ylim(0,1.05); ax.set_ylabel('Success rate')
        ax.set_title('MetaDrive release11 capability success by stage'); ax.legend(); fig.tight_layout()
        fig.savefig(output/'release11_capability_success.png', dpi=160); plt.close(fig)
        summary['figure'] = 'release11_capability_success.png'; save_json(summary, output/'comparison.json')
    except ImportError:
        pass
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default=str(_bootstrap.ROOT/'reports/v2_8'))
    p.add_argument('--selection', default=str(_bootstrap.ROOT/'reports/v2_8/selection_validation7.json'))
    p.add_argument('--model', default=str(_bootstrap.ROOT/'experiments/v2_8/capability_pureexpert_v4/model.pt'))
    p.add_argument('--output', default=str(_bootstrap.ROOT/'reports/v2_8'))
    a = p.parse_args(); print(json.dumps(build(Path(a.root),Path(a.selection),Path(a.model),Path(a.output)), indent=2))
