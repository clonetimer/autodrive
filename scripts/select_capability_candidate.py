"""Select a capability-specialized candidate on fresh validation split only."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import _bootstrap
from src.evaluation.capability_selection import select_capability_candidate, capability_promotion_decision
from src.scenarios.generator import SPLITS
from src.utils.logger import save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',action='append',required=True,help='NAME=path/to/suite_report.json')
    p.add_argument('--baseline',required=True)
    p.add_argument('--selection-split',choices=['validation4','validation5','validation6','validation7'],default='validation7')
    p.add_argument('--min-success-gain',type=float,default=.05)
    p.add_argument('--min-catastrophic-reduction',type=float,default=.10)
    p.add_argument('--min-timeout-reduction',type=float,default=.10)
    p.add_argument('--max-collision-increase',type=float,default=.05)
    p.add_argument('--max-offroad-increase',type=float,default=.05)
    p.add_argument('--max-stage-catastrophic-increase',type=float,default=.20)
    p.add_argument('--output',required=True)
    args=p.parse_args();reports={}
    for item in args.candidate:
        if '=' not in item:p.error('--candidate must be NAME=PATH')
        name,path=item.split('=',1)
        if not name or name in reports:p.error('candidate names must be unique')
        reports[name]=json.loads(Path(path).read_text(encoding='utf-8'))
    result=select_capability_candidate(reports,allowed_seed_range=SPLITS[args.selection_split],selection_domain=args.selection_split)
    result['promotion']=capability_promotion_decision(result,args.baseline,
        min_success_gain=args.min_success_gain,min_catastrophic_reduction=args.min_catastrophic_reduction,
        min_timeout_reduction=args.min_timeout_reduction,max_collision_increase=args.max_collision_increase,
        max_offroad_increase=args.max_offroad_increase,max_stage_catastrophic_increase=args.max_stage_catastrophic_increase)
    save_json(result,args.output);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
