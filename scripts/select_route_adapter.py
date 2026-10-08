"""Apply the predeclared raw-policy route-adapter promotion gate."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import _bootstrap
from src.evaluation.route_adapter_selection import RouteAdapterGate, select_route_adapter
from src.utils.logger import save_json, sha256


def load_parts(paths):
    episodes=[]
    for path in paths:
        report=json.loads(Path(path).read_text())
        episodes.extend(report['episodes'])
    return episodes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', nargs='+', required=True)
    p.add_argument('--candidate', nargs='+', required=True)
    p.add_argument('--base-model', required=True); p.add_argument('--adapter', required=True)
    p.add_argument('--output', required=True)
    args=p.parse_args()
    result=select_route_adapter(load_parts(args.baseline),load_parts(args.candidate),RouteAdapterGate())
    result['protocol']={'selection_only':'validation11','runtime_rules':'none',
        'release15_opened_before_selection':False,'base_model':str(Path(args.base_model).resolve()),
        'base_model_sha256':sha256(args.base_model),'adapter':str(Path(args.adapter).resolve()),
        'adapter_sha256':sha256(args.adapter),
        'baseline_reports':[{ 'path':str(Path(x).resolve()),'sha256':sha256(x)} for x in args.baseline],
        'candidate_reports':[{ 'path':str(Path(x).resolve()),'sha256':sha256(x)} for x in args.candidate]}
    save_json(result,args.output); print(json.dumps(result,indent=2))
    return 0 if result['promoted'] else 3

if __name__=='__main__': raise SystemExit(main())
