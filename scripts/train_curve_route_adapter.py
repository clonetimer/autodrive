"""Train a curve/ramp-specialized neural residual from a fixed RouteAssist teacher.

The teacher is used only while collecting fresh MetaDrive DAgger data. The
published adapter receives policy-visible state and the frozen base action; no
pure-pursuit controller is called at inference.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch import nn

import _bootstrap
from src.agents.trainer import load_checkpoint
from src.end_to_end.agent import EndToEndAgent
from src.evaluation.capability_suite import standard_capability_suite
from src.multi_agent.marl_env import make_parallel_env
from src.navigation.learned_route_adapter import (
    CurveRouteAdapter, CurveRouteAdapterConfig, LearnedRouteAdapterAgent,
    policy_visible_curve_score, save_curve_route_adapter, load_route_adapter,
)
from src.navigation.route_assist import RouteAssist
from src.scenarios.generator import split_seed_block
from src.utils.logger import save_json, sha256

STAGES = ('curve_tracking', 'ramp_merge')


def _specs(split: str, iteration: int, seeds_per_stage: int):
    template = {s.name: s for s in standard_capability_suite(split, seeds_per_stage)}
    block = split_seed_block(split, iteration * len(STAGES) * seeds_per_stage,
                             len(STAGES) * seeds_per_stage)
    out = []
    for i, name in enumerate(STAGES):
        seeds = tuple(block[i * seeds_per_stage:(i + 1) * seeds_per_stage])
        out.append(replace(template[name], seeds=seeds))
    return tuple(out)


def _collect_worker(args):
    from src.evaluation.safety_suite import SafetySuiteSpec
    payload = json.loads(args.worker_spec_json); payload['seeds'] = tuple(payload['seeds'])
    spec = SafetySuiteSpec(**payload)
    base = EndToEndAgent(args.base_model)
    _, cfg, _ = load_checkpoint(args.base_model)
    env_cfg = spec.env(cfg.env)
    env_cfg.max_steps = min(env_cfg.max_steps, args.max_stage_steps)
    env_cfg.route_assist = False
    env_cfg.safety_supervisor = True
    env_cfg.safety_ttc_threshold_s = args.ttc_threshold
    env_cfg.safety_risk_threshold = args.risk_threshold
    env = make_parallel_env(env_cfg)
    learner = LearnedRouteAdapterAgent(args.base_model, args.adapter) if args.adapter else None
    teacher = RouteAssist(args.teacher_blend, mode='fixed')
    rows = {k: [] for k in ('state','base_action','teacher_steering','teacher_correction','executed_steering','seed')}
    episodes=[]
    try:
        for seed in spec.seeds:
            obs,_=env.reset(seed=seed)
            if learner: learner.reset()
            success=collision=offroad=False; steps=0
            while env.agents:
                state=np.asarray(obs['agent0'],dtype=np.float32)
                base_action=base.predict(state)
                learner_action=learner.predict(state) if learner else base_action.copy()
                decision=teacher.decide(base_action,state)
                executed=learner_action.copy()
                executed[0]=np.clip(args.mix_beta*decision.applied[0]+(1-args.mix_beta)*learner_action[0],-1,1)
                rows['state'].append(state.copy()); rows['base_action'].append(base_action.copy())
                rows['teacher_steering'].append(float(decision.applied[0]))
                rows['teacher_correction'].append(float(decision.applied[0]-base_action[0]))
                rows['executed_steering'].append(float(executed[0])); rows['seed'].append(int(seed))
                nxt,rew,term,trunc,info=env.step({'agent0':executed}); inf=info['agent0']; steps+=1
                collision |= bool(inf['collision']); offroad |= bool(inf['out_of_road'])
                success |= bool(inf['success']) and bool(term['agent0'])
                obs={n:nxt[n] for n in env.agents}
            episodes.append({'seed':int(seed),'steps':steps,'success':bool(success and not collision and not offroad),
                             'collision':collision,'out_of_road':offroad})
        arrays={k:np.asarray(v,dtype=np.int64 if k=='seed' else np.float32) for k,v in rows.items()}
        out=Path(args.worker_output); out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,**arrays)
        save_json({'status':'complete','stage':spec.name,'samples':len(arrays['state']),
                   'mix_beta':args.mix_beta,'teacher_blend':args.teacher_blend,'episodes':episodes},out.with_suffix('.json'))
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    finally:
        env.close()


def _concat(paths):
    parts=[np.load(p) for p in paths]; keys=parts[0].files
    out={k:np.concatenate([p[k] for p in parts]) for k in keys}
    for p in parts:p.close()
    return out


def _split_indices(seeds):
    unique=np.unique(seeds)
    if len(unique)<2: raise RuntimeError('Need at least two scenario seeds')
    hold=set(unique[2::3].tolist() or unique[-1:].tolist())
    valid=np.asarray([i for i,x in enumerate(seeds) if int(x) in hold],dtype=np.int64)
    train=np.asarray([i for i,x in enumerate(seeds) if int(x) not in hold],dtype=np.int64)
    return train,valid,sorted(hold)


def _fit(model, arrays, epochs, lr, batch_size, seed):
    train,valid,hold=_split_indices(arrays['seed'])
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    rng=np.random.default_rng(seed); torch.manual_seed(seed); model.train()
    for _ in range(epochs):
        perm=rng.permutation(train)
        for start in range(0,len(perm),batch_size):
            idx=perm[start:start+batch_size]
            st=torch.as_tensor(arrays['state'][idx],dtype=torch.float32)
            ba=torch.as_tensor(arrays['base_action'][idx],dtype=torch.float32)
            target=torch.as_tensor(arrays['teacher_steering'][idx],dtype=torch.float32)
            pred,corr=model(st,ba)
            curve=policy_visible_curve_score(st)
            lane=st[:,3].abs().clamp(0,1)
            weight=1.0+5.0*curve+2.0*lane
            loss=(weight*(pred-target).square()).mean()+0.01*corr.square().mean()
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
    model.eval()
    def metrics(idx):
        with torch.no_grad():
            st=torch.as_tensor(arrays['state'][idx],dtype=torch.float32); ba=torch.as_tensor(arrays['base_action'][idx],dtype=torch.float32)
            target=torch.as_tensor(arrays['teacher_steering'][idx],dtype=torch.float32); pred,corr=model(st,ba)
            return {'samples':int(len(idx)),'steering_mse':float((pred-target).square().mean()),
                    'steering_mae':float((pred-target).abs().mean()),'mean_abs_correction':float(corr.abs().mean())}
    return {'train':metrics(train),'diagnostic_holdout':metrics(valid),'holdout_seeds':hold}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model',required=True); p.add_argument('--output',required=True)
    p.add_argument('--split',default='curve_train'); p.add_argument('--iterations',type=int,default=3)
    p.add_argument('--seeds-per-stage',type=int,default=2); p.add_argument('--mix-betas',default='1.0,0.5,0.2')
    p.add_argument('--epochs',type=int,default=30); p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--learning-rate',type=float,default=6e-4); p.add_argument('--hidden-size',type=int,default=128)
    p.add_argument('--max-correction',type=float,default=.18); p.add_argument('--teacher-blend',type=float,default=.5)
    p.add_argument('--max-stage-steps',type=int,default=420); p.add_argument('--stage-timeout',type=float,default=120)
    p.add_argument('--ttc-threshold',type=float,default=1.8); p.add_argument('--risk-threshold',type=float,default=.68)
    p.add_argument('--resume', action='store_true', help='resume fingerprint-compatible completed DAgger stages')
    p.add_argument('--worker-spec-json',help=argparse.SUPPRESS); p.add_argument('--worker-output',help=argparse.SUPPRESS)
    p.add_argument('--adapter',help=argparse.SUPPRESS); p.add_argument('--mix-beta',type=float,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker_spec_json: return _collect_worker(args)
    betas=[float(x) for x in args.mix_betas.split(',')];
    if len(betas)!=args.iterations: p.error('mix-betas count must equal iterations')
    _,cfg,_=load_checkpoint(args.base_model); state_dim=20+7*cfg.env.neighbors
    root=Path(args.output).resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError('Choose an empty output directory or use --resume')
    root.mkdir(parents=True,exist_ok=True)
    completed=[]
    if args.resume and (root/'iterations.json').is_file():
        completed=json.loads((root/'iterations.json').read_text())
        if not isinstance(completed,list): raise RuntimeError('Invalid iterations.json')
    model=CurveRouteAdapter(CurveRouteAdapterConfig(state_dim,args.hidden_size,args.max_correction))
    parts=[]; iterations=[]; adapter_path=None; started=time.time()
    for it in range(args.iterations):
        idir=root/f'iteration_{it}'; idir.mkdir(exist_ok=True)
        collection=[]
        specs=_specs(args.split,it,args.seeds_per_stage)
        # A completed iteration is immutable under resume: reuse its stage data
        # and reload the saved adapter so the next iteration starts from the
        # exact historical learner rather than a newly initialized network.
        if it < len(completed):
            report=completed[it]
            if int(report.get('iteration',-1)) != it:
                raise RuntimeError('iterations.json ordering mismatch')
            for spec in specs:
                npz=idir/f'{spec.name}.npz'; meta=npz.with_suffix('.json')
                if not npz.is_file() or not meta.is_file():
                    raise RuntimeError(f'Completed iteration {it} is missing {spec.name} data')
                summary=json.loads(meta.read_text())
                if summary.get('status')!='complete' or summary.get('stage')!=spec.name:
                    raise RuntimeError(f'Completed stage metadata mismatch: {meta}')
                parts.append(npz); collection.append(summary)
            adapter_path=Path(report['artifact']['path'])
            if not adapter_path.is_file():
                adapter_path=root/f'adapter_iteration_{it}.pt'
            model,_=load_route_adapter(adapter_path,base_model=args.base_model,device='cpu')
            if not isinstance(model,CurveRouteAdapter):
                raise RuntimeError('Resume artifact is not a curve_route_adapter_v2')
            iterations.append(report)
            continue
        for spec in specs:
            npz=idir/f'{spec.name}.npz'; meta=npz.with_suffix('.json'); log=npz.with_suffix('.log')
            reuse=False
            if args.resume and npz.is_file() and meta.is_file():
                try:
                    summary=json.loads(meta.read_text())
                    reuse=(summary.get('status')=='complete' and summary.get('stage')==spec.name
                           and list(summary.get('episodes',[{}])[0:0])==[])
                except (OSError,json.JSONDecodeError): reuse=False
            if not reuse:
                npz.unlink(missing_ok=True); meta.unlink(missing_ok=True)
                cmd=[sys.executable,str(Path(__file__).resolve()),'--base-model',str(Path(args.base_model).resolve()),
                     '--output',str(root),'--worker-spec-json',json.dumps(spec.to_dict()),'--worker-output',str(npz),
                     '--mix-beta',str(betas[it]),'--teacher-blend',str(args.teacher_blend),'--max-stage-steps',str(args.max_stage_steps),
                     '--ttc-threshold',str(args.ttc_threshold),'--risk-threshold',str(args.risk_threshold)]
                if adapter_path: cmd += ['--adapter',str(adapter_path)]
                with log.open('w') as f:
                    proc=subprocess.run(cmd,cwd=_bootstrap.ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=args.stage_timeout,
                                        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
                if proc.returncode or not npz.is_file(): raise RuntimeError(f'collection failed: {log}')
                summary=json.loads(meta.read_text())
            parts.append(npz); collection.append(summary)
        arrays=_concat(parts); fit=_fit(model,arrays,args.epochs,args.learning_rate,args.batch_size,3200+it)
        adapter_path=root/f'adapter_iteration_{it}.pt'
        training={'iteration':it,'mix_beta':betas[it],'fit':fit,'collection':collection,'samples_accumulated':len(arrays['state'])}
        artifact=save_curve_route_adapter(adapter_path,model,base_model=args.base_model,training=training,dataset=arrays)
        report={**training,'artifact':artifact}; iterations.append(report); save_json(iterations,root/'iterations.json')
    arrays=_concat(parts); final=root/'curve_route_adapter.pt'
    training={'algorithm':'online_dagger_curve_route_adapter_v2','iterations':iterations,'samples_total':len(arrays['state']),
              'unique_training_seeds':sorted(set(map(int,arrays['seed']))),'split':args.split,'stages':list(STAGES),
              'teacher':'fixed_policy_visible_pure_pursuit','teacher_blend':args.teacher_blend,
              'base_model_sha256':sha256(args.base_model),'elapsed_seconds':time.time()-started}
    artifact=save_curve_route_adapter(final,model,base_model=args.base_model,training=training,dataset=arrays)
    np.savez_compressed(root/'dataset.npz',**arrays)
    result={**training,'artifact':artifact,'final_fit':iterations[-1]['fit'],'status':'curve_route_adapter_trained_not_safety_certified'}
    save_json(result,root/'result.json'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
