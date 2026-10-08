"""Select one raw-policy candidate from identical MetaDrive validation suites."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import _bootstrap
from src.evaluation.model_selection import promotion_decision, select_safety_candidate
from src.scenarios.generator import SPLITS
from src.utils.logger import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', action='append', required=True,
                   help='NAME=path/to/suite_report.json; repeat at least twice')
    p.add_argument('--baseline', default=None,
                   help='Optional candidate name used for the predeclared promotion gate')
    p.add_argument('--min-success-gain', type=float, default=0.05)
    p.add_argument('--min-catastrophic-reduction', type=float, default=0.10)
    p.add_argument('--max-offroad-increase', type=float, default=0.05)
    p.add_argument('--max-collision-increase', type=float, default=0.0)
    p.add_argument('--selection-split', choices=['validation', 'validation2'], default='validation')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    reports = {}
    for item in args.candidate:
        if '=' not in item:
            p.error('--candidate must be NAME=PATH')
        name, raw = item.split('=', 1)
        if not name or name in reports:
            p.error('candidate names must be non-empty and unique')
        reports[name] = json.loads(Path(raw).read_text(encoding='utf-8'))
    result = select_safety_candidate(reports, allowed_seed_range=SPLITS[args.selection_split], selection_domain=args.selection_split)
    if args.baseline:
        result['promotion'] = promotion_decision(
            result, args.baseline,
            min_success_gain=args.min_success_gain,
            min_catastrophic_reduction=args.min_catastrophic_reduction,
            max_offroad_increase=args.max_offroad_increase,
            max_collision_increase=args.max_collision_increase,
        )
    save_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
