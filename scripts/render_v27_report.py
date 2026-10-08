"""Render v2.7 privileged-expert validation and frozen release6 diagnostics.

The candidate is selected and promoted exclusively on validation2.  The script
verifies the frozen model record predates release6 use, then summarizes release6
without allowing release outcomes to alter candidate selection.
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


def episode_map(report: dict) -> dict[tuple[str, int], dict]:
    rows = report.get('episodes', [])
    out = {(str(r['suite_stage']), int(r['seed'])): r for r in rows}
    if len(out) != len(rows):
        raise ValueError('Expected one episode for each unique stage/seed pair')
    return out


def paired(base: dict, candidate: dict, good_fn) -> dict:
    if set(base) != set(candidate):
        raise ValueError('Paired reports have different stage/seed scenarios')
    improved = regressed = both_good = both_bad = 0
    for key in sorted(base):
        b, c = bool(good_fn(base[key])), bool(good_fn(candidate[key]))
        if not b and c: improved += 1
        elif b and not c: regressed += 1
        elif b and c: both_good += 1
        else: both_bad += 1
    return {'improved': improved, 'regressed': regressed, 'both_good': both_good,
            'both_bad': both_bad, 'exact_mcnemar_p_two_sided': exact_mcnemar_p(improved, regressed)}


def subset(m: dict, supervised: bool = False) -> dict:
    keys = ['success_rate','collision_rate','out_of_road_rate','timeout_rate','mean_route_completion',
            'avg_speed_mps','lane_abs_error_m','jerk_mean_abs_mps3']
    if supervised:
        keys += ['safety_intervention_time_fraction','safety_mean_longitudinal_correction',
                 'safety_peak_longitudinal_correction','safety_mean_severity']
    return {k:m.get(k) for k in keys}


def catastrophic(rows: dict) -> float:
    return sum(bool(r.get('collision')) or bool(r.get('out_of_road')) for r in rows.values()) / len(rows)


def build(root: Path, selection_path: Path, freeze_path: Path, output: Path) -> dict:
    names = ['release6_base_raw','release6_selected_raw','release6_base_supervised','release6_selected_supervised']
    reports = {n: json.loads((root/n/'suite_report.json').read_text()) for n in names}
    selection = json.loads(selection_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    if selection.get('selection_domain') != 'validation2_seeds_11000_11199_only':
        raise ValueError('v2.7 selection must be validation2-only')
    if not selection.get('promotion',{}).get('promoted'):
        raise ValueError('Frozen v2.7 release candidate must have passed the pre-release promotion gate')
    if not freeze.get('frozen_before_release6') or freeze.get('release6_used_for_selection'):
        raise ValueError('Missing release6 freeze boundary')
    if freeze.get('selection_sha256') != sha256(selection_path):
        raise ValueError('Selection report changed after the model freeze record')
    selected_path = _bootstrap.ROOT / freeze['model_path']
    if freeze.get('model_sha256') != sha256(selected_path):
        raise ValueError('Frozen selected model hash mismatch')

    signatures=[]
    for report in reports.values():
        sig=[]
        for s in report.get('suite',[]):
            sig.append((s['name'],s['metadrive_map'],float(s['traffic_density']),float(s['target_speed']),int(s['max_steps']),tuple(s['seeds'])))
        signatures.append(tuple(sig))
    if len(set(signatures)) != 1:
        raise ValueError('release6 safety reports do not share one frozen suite')
    release_keys=[seed for stage in signatures[0] for seed in stage[-1]]
    if release_keys != list(range(12000,12040)):
        raise ValueError('Expected release6 first frozen slice 12000-12039')

    maps={n:episode_map(r) for n,r in reports.items()}
    success_pairs=paired(maps['release6_base_raw'],maps['release6_selected_raw'],lambda r:r.get('success'))
    noncat_pairs=paired(maps['release6_base_raw'],maps['release6_selected_raw'],
                        lambda r:not bool(r.get('collision')) and not bool(r.get('out_of_road')))
    base_raw=reports['release6_base_raw']['metrics']; sel_raw=reports['release6_selected_raw']['metrics']
    base_sup=reports['release6_base_supervised']['metrics']; sel_sup=reports['release6_selected_supervised']['metrics']
    base_cat=catastrophic(maps['release6_base_raw']); sel_cat=catastrophic(maps['release6_selected_raw'])
    intervention_delta=[]
    for key in sorted(maps['release6_base_supervised']):
        intervention_delta.append(float(maps['release6_selected_supervised'][key].get('safety_intervention_time_fraction',0))-
                                  float(maps['release6_base_supervised'][key].get('safety_intervention_time_fraction',0)))

    path={}
    for name in ['release6_path_base_supervised','release6_path_selected_supervised']:
        p=root/name/'suite_report.json'
        if p.is_file(): path[name]=json.loads(p.read_text())
    path_summary=None
    if len(path)==2:
        path_summary={
            'note':'Secondary post-freeze diagnostic using release6 seed identities under different path-suite scenario definitions; not an independent test set.',
            'base':subset(path['release6_path_base_supervised']['metrics'],True),
            'selected':subset(path['release6_path_selected_supervised']['metrics'],True),
        }

    summary={
      'version':'2.7.0',
      'method':'training-only privileged MetaDrive IDM lateral expert distillation',
      'selection':{
        'selected':selection['selected'],'selection_domain':selection['selection_domain'],
        'promotion':selection['promotion'],'freeze_record':str(freeze_path.relative_to(_bootstrap.ROOT)),
        'release6_not_used_for_selection':True,
      },
      'training_evidence':{
        'source_model':freeze['source_model'],'source_rl_world_steps':freeze['source_rl_world_steps'],
        'selected_expert_training_seeds':freeze['expert_training_seeds'],
        'selected_expert_transitions':freeze['expert_transitions'],
        'selected_model_sha256':freeze['model_sha256'],
        'teacher_deployed':False,
      },
      'release6':{
        'slice':'12000-12039','scenarios':40,'consumed':True,
        'base_raw':subset(base_raw),'selected_raw':subset(sel_raw),
        'base_catastrophic_rate':base_cat,'selected_catastrophic_rate':sel_cat,
        'success_change_pp':100*(sel_raw['success_rate']-base_raw['success_rate']),
        'collision_change_pp':100*(sel_raw['collision_rate']-base_raw['collision_rate']),
        'offroad_change_pp':100*(sel_raw['out_of_road_rate']-base_raw['out_of_road_rate']),
        'catastrophic_change_pp':100*(sel_cat-base_cat),
        'paired_success':success_pairs,'paired_noncatastrophic':noncat_pairs,
        'base_supervised':subset(base_sup,True),'selected_supervised':subset(sel_sup,True),
        'supervisor_intervention_relative_change':(sel_sup['safety_intervention_time_fraction']/base_sup['safety_intervention_time_fraction']-1 if base_sup['safety_intervention_time_fraction'] else None),
        'scenarios_with_lower_supervisor_intervention':sum(x<0 for x in intervention_delta),
        'scenarios_with_higher_supervisor_intervention':sum(x>0 for x in intervention_delta),
      },
      'path_release6_secondary_diagnostic':path_summary,
      'negative_experiments':{
        'anchored_distillation':'Action/feature anchoring bounded source-policy drift but did not outperform the unanchored balanced candidate on validation2 path success.',
        'large_unconstrained_dagger':'A larger encoder+actor imitation attempt reduced training loss but degraded path completion, consistent with representation drift.',
      },
      'release6_interpretation':'Validation2 promotion was not confirmed as a robust raw-policy safety improvement on release6: success changed only slightly and collision increased while off-road decreased. The lateral expert shows a directional path-following benefit in the secondary path diagnostic, but failure rates remain unacceptable.',
      'future_tuning_rule':'release6 is now consumed; future candidate tuning/selection must use new training/validation domains and a new untouched release split.',
      'claim_boundary':'Privileged expert data are training-only. The frozen 40-scenario release is research evidence, not a safety certificate or real-road risk estimate. Project is prohibited for real-vehicle control.'
    }
    output.mkdir(parents=True,exist_ok=True); save_json(summary,output/'comparison.json')
    p=selection['promotion']
    lines=['# v2.7 release6 privileged-expert distillation report','',
      'v2.7 targets the lateral/path bottleneck discovered in v2.6. A MetaDrive built-in IDM policy is used only as a privileged training-time steering teacher; deployment remains the neural state policy.','',
      '## Validation2 selection before release6','',
      f"The fixed validation2 suite used 20 scenarios (11000-11019). `{selection['selected']}` ranked first and passed the predeclared engineering promotion gate before release6 was opened.",'',
      '| Candidate | Success | Collision | Off-road | Catastrophic | Route completion |','|---|---:|---:|---:|---:|---:|']
    for row in selection['scores']:
        lines.append(f"| {row['name']} | {row['success_rate']:.1%} | {row['collision_rate']:.1%} | {row['out_of_road_rate']:.1%} | {row['catastrophic_rate']:.1%} | {row['mean_route_completion']:.1%} |")
    lines += ['',f"Promotion: **{p['promoted']}**. The material gain was exactly one of 20 validation scenarios (+5pp success), so it is a weak engineering gate rather than statistical evidence.",'',
      '## Release6 frozen safety suite (12000-12039)','',
      '| System | Success | Collision | Off-road | Timeout | Route completion | Supervisor intervention |','|---|---:|---:|---:|---:|---:|---:|',
      f"| v2.5 source raw | {base_raw['success_rate']:.1%} | {base_raw['collision_rate']:.1%} | {base_raw['out_of_road_rate']:.1%} | {base_raw['timeout_rate']:.1%} | {base_raw['mean_route_completion']:.1%} | 0% |",
      f"| v2.7 expert-distilled raw | {sel_raw['success_rate']:.1%} | {sel_raw['collision_rate']:.1%} | {sel_raw['out_of_road_rate']:.1%} | {sel_raw['timeout_rate']:.1%} | {sel_raw['mean_route_completion']:.1%} | 0% |",
      f"| v2.5 source + supervisor | {base_sup['success_rate']:.1%} | {base_sup['collision_rate']:.1%} | {base_sup['out_of_road_rate']:.1%} | {base_sup['timeout_rate']:.1%} | {base_sup['mean_route_completion']:.1%} | {base_sup['safety_intervention_time_fraction']:.1%} |",
      f"| v2.7 expert-distilled + supervisor | {sel_sup['success_rate']:.1%} | {sel_sup['collision_rate']:.1%} | {sel_sup['out_of_road_rate']:.1%} | {sel_sup['timeout_rate']:.1%} | {sel_sup['mean_route_completion']:.1%} | {sel_sup['safety_intervention_time_fraction']:.1%} |",'',
      f"Raw success changed by {100*(sel_raw['success_rate']-base_raw['success_rate']):+.1f}pp, collision {100*(sel_raw['collision_rate']-base_raw['collision_rate']):+.1f}pp, off-road {100*(sel_raw['out_of_road_rate']-base_raw['out_of_road_rate']):+.1f}pp, and collision-or-off-road {100*(sel_cat-base_cat):+.1f}pp. Paired success: {success_pairs['improved']} improved / {success_pairs['regressed']} regressed, exact McNemar p={success_pairs['exact_mcnemar_p_two_sided']:.4f}.",'',
      f"With the same longitudinal supervisor, intervention time changed from {base_sup['safety_intervention_time_fraction']:.1%} to {sel_sup['safety_intervention_time_fraction']:.1%}. This reduces longitudinal rule dependence, but system success did not improve and selected-system collision was not zero.",'']
    if path_summary:
      pb=path_summary['base'];ps=path_summary['selected']
      lines += ['## Secondary path diagnostic','',
       f"On 20 curved/mixed path-suite scenarios using release6 seed identities, supervised success changed {pb['success_rate']:.1%}→{ps['success_rate']:.1%}, off-road {pb['out_of_road_rate']:.1%}→{ps['out_of_road_rate']:.1%}, route completion {pb['mean_route_completion']:.1%}→{ps['mean_route_completion']:.1%}, and mean lane error {pb['lane_abs_error_m']:.3f}→{ps['lane_abs_error_m']:.3f} m. This is a directional lateral-control signal, but 80% off-road remains unacceptable and this diagnostic is not an independent release split.",'']
    lines += ['## Negative results and boundary','',
      '- Source-action/feature anchored expert distillation reduced imitation drift but did not beat the balanced candidate on validation2 path success.',
      '- A larger unconstrained encoder+actor imitation attempt showed representation drift and worse path completion.',
      '- release6 is now consumed and cannot be reused for future tuning.',
      '- The privileged IDM teacher is not available at deployment and no result is a safety certification.',
      '- Project artifacts are for simulation/research only and must not control a real vehicle.','']
    (output/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    try:
      import matplotlib.pyplot as plt
      labels=['Base raw','Expert raw','Base+sup','Expert+sup']; x=range(4);w=.24
      success=[base_raw['success_rate'],sel_raw['success_rate'],base_sup['success_rate'],sel_sup['success_rate']]
      collision=[base_raw['collision_rate'],sel_raw['collision_rate'],base_sup['collision_rate'],sel_sup['collision_rate']]
      off=[base_raw['out_of_road_rate'],sel_raw['out_of_road_rate'],base_sup['out_of_road_rate'],sel_sup['out_of_road_rate']]
      fig,ax=plt.subplots(figsize=(9,5)); ax.bar([i-w for i in x],success,w,label='Success');ax.bar(list(x),collision,w,label='Collision');ax.bar([i+w for i in x],off,w,label='Off-road');ax.set_xticks(list(x),labels,rotation=10);ax.set_ylim(0,1.05);ax.set_ylabel('Rate');ax.set_title('MetaDrive release6 frozen comparison');ax.legend();fig.tight_layout();fig.savefig(output/'release6_comparison.png',dpi=160);plt.close(fig)
      summary['figure']='release6_comparison.png';save_json(summary,output/'comparison.json')
    except ImportError: pass
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(_bootstrap.ROOT/'reports/v2_7'));p.add_argument('--selection',default=str(_bootstrap.ROOT/'reports/v2_7/selection_validation2.json'));p.add_argument('--freeze',default=str(_bootstrap.ROOT/'reports/v2_7/FROZEN_MODEL.json'));p.add_argument('--output',default=str(_bootstrap.ROOT/'reports/v2_7'));a=p.parse_args();print(json.dumps(build(Path(a.root),Path(a.selection),Path(a.freeze),Path(a.output)),indent=2))
