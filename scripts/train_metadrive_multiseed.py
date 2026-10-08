"""Run independent real-MetaDrive training seeds and aggregate provenance.

This is intentionally a reproducibility runner rather than an ensemble trainer.
Each seed starts from fresh random policy weights and a disjoint simulator seed.
The resulting models are never averaged silently; downstream evaluation must
report per-seed outcomes before making algorithm-level claims.
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import _bootstrap
from src.utils.logger import save_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',action='append',type=int,required=True)
    p.add_argument('--steps',type=int,default=2048);p.add_argument('--timeout',type=float,default=120.)
    args=p.parse_args()
    seeds=args.seed
    if len(set(seeds))!=len(seeds) or len(seeds)<2:p.error('provide at least two unique --seed values')
    if args.steps<=0 or args.timeout<=0:p.error('steps/timeout must be positive')
    root=Path(args.output).resolve()
    if root.exists() and any(root.iterdir()):raise FileExistsError('Choose an empty output directory')
    root.mkdir(parents=True,exist_ok=True);runs=[]
    for seed in seeds:
        out=root/f'seed_{seed}';out.mkdir()
        cmd=[sys.executable,str(_bootstrap.ROOT/'scripts/train.py'),'--config',str(Path(args.config).resolve()),
             '--output',str(out),'--steps',str(args.steps),'--seed',str(seed)]
        started=time.monotonic()
        with (out/'process.log').open('w',encoding='utf-8') as log:
            try:
                completed=subprocess.run(cmd,cwd=_bootstrap.ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},timeout=args.timeout)
                rc=completed.returncode;timed_out=False
            except subprocess.TimeoutExpired:
                rc=None;timed_out=True
        model=out/'model.pt';result_path=out/'result.json'
        complete=(rc==0 and model.is_file() and result_path.is_file())
        row={'seed':seed,'steps_requested':args.steps,'elapsed_seconds':time.monotonic()-started,
             'return_code':rc,'timed_out':timed_out,'complete':complete,'directory':str(out.relative_to(root))}
        if complete:
            row['model_sha256']=sha256(model)
            row['result']=json.loads(result_path.read_text(encoding='utf-8'))
        runs.append(row);save_json({'status':'running','runs':runs},root/'multiseed.json')
        if not complete:raise RuntimeError(f'MetaDrive seed {seed} did not complete; inspect {out / "process.log"}')
    result={'status':'multiseed_training_completed_not_algorithm_claim','config':str(Path(args.config).resolve()),
            'steps_per_seed':args.steps,'seeds':seeds,'runs':runs,
            'claim_boundary':'Independent seeds quantify training variability; no model averaging or cherry-picking is implied.'}
    save_json(result,root/'multiseed.json');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
