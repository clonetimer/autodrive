"""Run the frozen v4.2 formal training comparison on authority_train only.

Formal treatments:
  G0  : always-authorized mixed-support longitudinal residual.
  GPL : structural neighbour-presence gate x learned intervention gate.

The development seed (12400) is intentionally excluded.  This launcher has no
validation16 or release19 evaluation path; holdout plans are created only after
all six training artifacts pass pre-open audit.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import _bootstrap
from src.utils.logger import save_json, sha256

VARIANTS={
 'G0': {'presence_gate':False,'learned_gate':False,'description':'always-authorized mixed-support longitudinal residual'},
 'GPL':{'presence_gate':True,'learned_gate':True,'description':'presence x learned intervention authority gate'},
}
FORMAL_SEEDS=(13400,14400,15400)
DEV_SEED=12400

def _csv_ints(text):
 out=[int(x.strip()) for x in text.split(',') if x.strip()]
 if not out or len(set(out))!=len(out): raise argparse.ArgumentTypeError('training seeds must be non-empty and unique')
 if DEV_SEED in out: raise argparse.ArgumentTypeError(f'development seed {DEV_SEED} is reserved and cannot be used for formal repeats')
 return out

def _command_plan(args):
 plan=[]
 for variant in ('G0','GPL'):
  v=VARIANTS[variant]
  for seed in args.training_seeds:
   name=f'{variant}_seed{seed}';trial=Path(args.output).resolve()/name
   cmd=[sys.executable,str(_bootstrap.ROOT/'scripts/train_gated_longitudinal_adapter.py'),
    '--base-model',str(Path(args.base_model).resolve()),'--lateral-adapter',str(Path(args.lateral_adapter).resolve()),
    '--output',str(trial/'train'),'--split','authority_train','--iterations',str(args.iterations),'--seeds-per-stage',str(args.seeds_per_stage),
    '--mix-betas',args.mix_betas,'--epochs',str(args.epochs),'--batch-size',str(args.batch_size),'--learning-rate',str(args.learning_rate),
    '--hidden-size',str(args.hidden_size),'--history-len',str(args.history_len),'--neighbor-heads',str(args.neighbor_heads),'--use-neighbors',
    '--max-longitudinal-correction',str(args.max_longitudinal_correction),
    '--presence-gate' if v['presence_gate'] else '--no-presence-gate',
    '--learned-gate' if v['learned_gate'] else '--no-learned-gate',
    '--gate-loss-weight',str(args.gate_loss_weight),'--gate-prior-probability',str(args.gate_prior_probability),
    '--collection-mode','mixed','--training-seed',str(seed),'--max-stage-steps',str(args.max_stage_steps),
    '--stage-timeout',str(args.stage_timeout),'--ttc-threshold',str(args.ttc_threshold),'--risk-threshold',str(args.risk_threshold),
    '--torch-threads',str(args.torch_threads)]
   plan.append({'name':name,'variant':variant,'training_seed':seed,**v,'train_output':str(trial/'train'),
                'adapter':str(trial/'train/gated_longitudinal_temporal_adapter.pt'),'train_command':cmd})
 return plan

def _fingerprint(m):
 payload={k:v for k,v in m.items() if k not in {'status','job_status','plan_fingerprint_sha256'}}
 return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def _complete(j): return (Path(j['train_output'])/'result.json').is_file() and Path(j['adapter']).is_file()
def _exec_cmd(j,resume):
 c=list(j['train_command'])
 if resume and (Path(j['train_output'])/'run_config.json').is_file(): c.append('--resume')
 return c

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--base-model',required=True);p.add_argument('--lateral-adapter',required=True);p.add_argument('--output',required=True)
 p.add_argument('--training-seeds',type=_csv_ints,default=list(FORMAL_SEEDS));p.add_argument('--iterations',type=int,default=3);p.add_argument('--seeds-per-stage',type=int,default=1)
 p.add_argument('--mix-betas',default='1.0,0.5,0.2');p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=512)
 p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--hidden-size',type=int,default=128);p.add_argument('--history-len',type=int,default=6)
 p.add_argument('--neighbor-heads',type=int,default=4);p.add_argument('--max-longitudinal-correction',type=float,default=1.5)
 p.add_argument('--gate-loss-weight',type=float,default=.25);p.add_argument('--gate-prior-probability',type=float,default=.05)
 p.add_argument('--max-stage-steps',type=int,default=420);p.add_argument('--stage-timeout',type=float,default=150.);p.add_argument('--ttc-threshold',type=float,default=1.8)
 p.add_argument('--risk-threshold',type=float,default=.68);p.add_argument('--torch-threads',type=int,default=2);p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true')
 args=p.parse_args();base=Path(args.base_model).resolve();lat=Path(args.lateral_adapter).resolve()
 if not base.is_file():p.error(f'base model does not exist: {base}')
 if not lat.is_file():p.error(f'lateral adapter does not exist: {lat}')
 root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True);jobs=_command_plan(args)
 manifest={'schema':'v42_formal_authority_training_v1','status':'planned' if args.dry_run else 'running','development_selection':'GPL',
  'formal_baseline':'G0','development_seed_excluded':DEV_SEED,'base_model':str(base),'base_model_sha256':sha256(base),'lateral_adapter':str(lat),'lateral_adapter_sha256':sha256(lat),
  'lateral_control_contract':'frozen_v32_exact_steering_passthrough','training_split':'authority_train','selection_split_reserved':'validation16','final_release_reserved':'release19',
  'validation16_accessed':False,'release19_accessed':False,'collection_mode':'mixed','training_seeds':args.training_seeds,'jobs':jobs}
 manifest['plan_fingerprint_sha256']=_fingerprint(manifest);path=root/'v42_formal_training_manifest.json';prior={}
 if path.is_file():
  old=json.loads(path.read_text());has_data=any(Path(j['train_output']).exists() for j in jobs)
  if args.resume:
   if old.get('plan_fingerprint_sha256')!=manifest['plan_fingerprint_sha256']:raise RuntimeError('v4.2 formal training plan mismatch')
   prior=old.get('job_status',{})
  elif not args.dry_run and has_data:raise RuntimeError('Formal output contains trial data; use --resume')
 manifest['job_status']=dict(prior);save_json(manifest,path)
 if args.dry_run:print(json.dumps(manifest,indent=2));return
 for j in jobs:
  n=j['name']
  if args.resume and _complete(j):print(f'[v4.2-formal] skip completed {n}',flush=True)
  else:
   c=_exec_cmd(j,args.resume);manifest['job_status'][n]='training_resume' if '--resume' in c else 'training';save_json(manifest,path)
   print(f'[v4.2-formal] train {n}',flush=True);subprocess.run(c,cwd=_bootstrap.ROOT,check=True)
  manifest['job_status'][n]='train_complete';save_json(manifest,path)
 manifest['status']='training_complete';save_json(manifest,path);print(json.dumps({'status':'training_complete','jobs':len(jobs),'validation16_accessed':False,'release19_accessed':False},indent=2))
if __name__=='__main__':main()
