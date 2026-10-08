"""Train v4.1 longitudinal-only residual on top of frozen v3.2 lateral control.

The frozen policy stack is:
    v2.8 PPO base -> v3.2 learned curve-route steering adapter
Only the longitudinal command is learned here.  A stateful counterfactual
SafetySupervisor provides longitudinal labels from pre-step simulator snapshots.
``protected`` keeps the environment supervisor on for every DAgger iteration;
``mixed`` uses it only for iteration 0 and collects later learner rollouts raw.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch import nn
import _bootstrap
from src.agents.trainer import load_checkpoint
from src.evaluation.capability_suite import standard_capability_suite
from src.multi_agent.marl_env import make_parallel_env
from src.navigation.learned_route_adapter import LearnedRouteAdapterAgent
from src.navigation.longitudinal_residual_adapter import (
    FactorizedDrivingAgent, LongitudinalResidualConfig, LongitudinalTemporalAdapter,
    load_longitudinal_adapter, save_longitudinal_adapter)
from src.safety.supervisor import SafetySupervisor
from src.scenarios.generator import split_seed_block
from src.utils.logger import save_json, sha256

STAGES=('curve_tracking','ramp_merge','bottleneck_yield','intersection_yield','dense_progress')


def _specs(split:str, iteration:int, seeds_per_stage:int):
    template={s.name:s for s in standard_capability_suite(split,max(1,seeds_per_stage))}
    count=len(STAGES)*seeds_per_stage
    block=split_seed_block(split,iteration*count,count)
    return tuple(replace(template[name],seeds=tuple(block[i*seeds_per_stage:(i+1)*seeds_per_stage]))
                 for i,name in enumerate(STAGES))


def _teacher_snapshot(env):
    if hasattr(env,'_last_snapshots'): return env._last_snapshots['agent0']
    if hasattr(env,'snapshot'): return env.snapshot('agent0')
    raise RuntimeError('Environment does not expose the snapshot contract required by v4.1')


def _environment_safety_for_iteration(mode:str, iteration:int)->bool:
    if mode=='protected': return True
    if mode=='mixed': return iteration==0
    if mode=='raw': return False
    raise ValueError(f'Unknown collection mode {mode}')


def _collect_worker(args):
    from src.evaluation.safety_suite import SafetySuiteSpec
    payload=json.loads(args.worker_spec_json); payload['seeds']=tuple(payload['seeds']); spec=SafetySuiteSpec(**payload)
    _,cfg,_=load_checkpoint(args.base_model)
    env_cfg=spec.env(cfg.env); env_cfg.max_steps=min(env_cfg.max_steps,args.max_stage_steps)
    env_cfg.route_assist=False; env_cfg.safety_supervisor=bool(args.environment_safety)
    env_cfg.safety_ttc_threshold_s=args.ttc_threshold; env_cfg.safety_risk_threshold=args.risk_threshold
    env=make_parallel_env(env_cfg)
    lateral=LearnedRouteAdapterAgent(args.base_model,args.lateral_adapter)
    learner=(FactorizedDrivingAgent(args.base_model,args.lateral_adapter,args.adapter) if args.adapter else None)
    float_keys=('state','lateral_action','learner_action','teacher_longitudinal','teacher_longitudinal_correction',
                'executed_action','applied_action','reward','counterfactual_safety_severity','environment_safety_severity')
    bool_keys=('counterfactual_safety_intervention','environment_safety_intervention','collision','out_of_road','success')
    rows={k:[] for k in (*float_keys,*bool_keys,'seed','step')}; episodes=[]
    try:
        for seed in spec.seeds:
            obs,_=env.reset(seed=seed); lateral.reset()
            if learner: learner.reset()
            teacher=SafetySupervisor(ttc_threshold_s=args.ttc_threshold,risk_threshold=args.risk_threshold)
            success=collision=offroad=False; steps=0
            while env.agents:
                state=np.asarray(obs['agent0'],dtype=np.float32)
                lateral_action=np.asarray(lateral.predict(state),dtype=np.float32)
                learner_action=np.asarray(learner.predict(state) if learner else lateral_action.copy(),dtype=np.float32)
                if not np.isclose(learner_action[0],lateral_action[0],rtol=0,atol=1e-7):
                    raise RuntimeError('v4.1 learner changed frozen v3.2 steering')
                snap=_teacher_snapshot(env); decision=teacher.decide(lateral_action,snap)
                teacher_long=float(np.asarray(decision.applied,dtype=np.float32)[1])
                executed=learner_action.copy()
                executed[0]=lateral_action[0]
                executed[1]=np.clip(args.mix_beta*teacher_long+(1.0-args.mix_beta)*learner_action[1],-1.0,1.0)
                nxt,rew,term,trunc,info=env.step({'agent0':executed}); inf=info['agent0']
                rows['state'].append(state.copy()); rows['lateral_action'].append(lateral_action.copy())
                rows['learner_action'].append(learner_action.copy()); rows['teacher_longitudinal'].append(teacher_long)
                rows['teacher_longitudinal_correction'].append(teacher_long-float(lateral_action[1]))
                rows['executed_action'].append(executed.copy()); rows['applied_action'].append(np.asarray(inf['applied_action'],dtype=np.float32).copy())
                rows['reward'].append(float(rew['agent0'])); rows['counterfactual_safety_intervention'].append(bool(decision.intervened))
                rows['counterfactual_safety_severity'].append(float(decision.severity)); rows['environment_safety_intervention'].append(bool(inf.get('safety_intervention',False)))
                rows['environment_safety_severity'].append(float(inf.get('safety_severity',0.0))); rows['collision'].append(bool(inf['collision']))
                rows['out_of_road'].append(bool(inf['out_of_road'])); rows['success'].append(bool(inf['success'])); rows['seed'].append(int(seed)); rows['step'].append(int(steps))
                steps+=1; collision|=bool(inf['collision']); offroad|=bool(inf['out_of_road']); success|=bool(inf['success']) and bool(term['agent0'])
                obs={n:nxt[n] for n in env.agents}
            episodes.append({'seed':int(seed),'steps':steps,'success':bool(success and not collision and not offroad),'collision':collision,'out_of_road':offroad})
        arrays={}
        for k,v in rows.items(): arrays[k]=np.asarray(v,dtype=np.int64 if k in {'seed','step'} else np.bool_ if k in bool_keys else np.float32)
        out=Path(args.worker_output); out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,**arrays)
        save_json({'status':'complete','stage':spec.name,'samples':len(arrays['state']),'mix_beta':args.mix_beta,
                   'environment_safety':bool(args.environment_safety),
                   'counterfactual_safety_interventions':int(arrays['counterfactual_safety_intervention'].sum()),
                   'environment_safety_interventions':int(arrays['environment_safety_intervention'].sum()),'episodes':episodes},out.with_suffix('.json'))
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
    finally: env.close()


def _concat(paths):
    parts=[np.load(p) for p in paths]; keys=tuple(parts[0].files)
    for part in parts[1:]:
        if tuple(part.files)!=keys: raise RuntimeError('DAgger dataset schemas do not match')
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


def _history_tensor(states,seeds,steps,history_len):
    states=np.asarray(states,dtype=np.float32); seeds=np.asarray(seeds); steps=np.asarray(steps)
    result=np.empty((len(states),history_len,states.shape[1]),dtype=np.float32)
    for seed in np.unique(seeds):
        idx=np.flatnonzero(seeds==seed); order=idx[np.argsort(steps[idx],kind='stable')]
        if len(np.unique(steps[order]))!=len(order): raise RuntimeError(f'Duplicate step in seed {int(seed)}')
        for pos,target in enumerate(order):
            source=order[max(0,pos-history_len+1):pos+1]; missing=history_len-len(source)
            if missing: source=np.concatenate((np.repeat(source[:1],missing),source))
            result[target]=states[source]
    return result


def _fit(model,arrays,epochs,lr,batch_size,seed):
    train,valid,hold=_split_indices(arrays['seed']); histories=_history_tensor(arrays['state'],arrays['seed'],arrays['step'],model.config.history_len)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4); rng=np.random.default_rng(seed); torch.manual_seed(seed); model.train()
    for _ in range(epochs):
        perm=rng.permutation(train)
        for start in range(0,len(perm),batch_size):
            idx=perm[start:start+batch_size]
            hist=torch.as_tensor(histories[idx],dtype=torch.float32); lateral=torch.as_tensor(arrays['lateral_action'][idx],dtype=torch.float32)
            target=torch.as_tensor(arrays['teacher_longitudinal'][idx],dtype=torch.float32)
            severity=torch.as_tensor(arrays['counterfactual_safety_severity'][idx],dtype=torch.float32)
            intervention=torch.as_tensor(arrays['counterfactual_safety_intervention'][idx].astype(np.float32),dtype=torch.float32)
            pred,delta,_=model(hist,lateral); weight=.35+4.0*severity.clamp(0,1)+2.0*intervention
            imitation=(weight*(pred[:,1]-target).square()).mean(); residual=.005*delta.square().mean(); loss=imitation+residual
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
    model.eval()
    def metrics(idx):
        with torch.no_grad():
            hist=torch.as_tensor(histories[idx],dtype=torch.float32); lateral=torch.as_tensor(arrays['lateral_action'][idx],dtype=torch.float32)
            target=torch.as_tensor(arrays['teacher_longitudinal'][idx],dtype=torch.float32); pred,delta,attention=model(hist,lateral)
            inter=arrays['counterfactual_safety_intervention'][idx].astype(bool)
            r={'samples':int(len(idx)),'longitudinal_mse':float((pred[:,1]-target).square().mean()),'longitudinal_mae':float((pred[:,1]-target).abs().mean()),
               'mean_abs_longitudinal_correction':float(delta.abs().mean()),'counterfactual_intervention_rate':float(np.mean(inter)),
               'neighbor_attention_mass':float(attention[:,:-1].sum(dim=1).mean()),'steering_passthrough_max_abs_error':float((pred[:,0]-lateral[:,0]).abs().max())}
            if np.any(inter):
                mask=torch.as_tensor(inter,dtype=torch.bool); r['intervention_longitudinal_mse']=float((pred[mask,1]-target[mask]).square().mean())
            else:r['intervention_longitudinal_mse']=None
            return r
    return {'train':metrics(train),'diagnostic_holdout':metrics(valid),'holdout_seeds':hold}


def _run_config(args,betas,schedule):
    c={'schema':'v41_longitudinal_training_run_v1','base_model_sha256':sha256(args.base_model),'lateral_adapter_sha256':sha256(args.lateral_adapter),
       'split':args.split,'stages':list(STAGES),'iterations':args.iterations,'seeds_per_stage':args.seeds_per_stage,'mix_betas':betas,
       'environment_safety_schedule':schedule,'epochs':args.epochs,'batch_size':args.batch_size,'learning_rate':args.learning_rate,'hidden_size':args.hidden_size,
       'history_len':args.history_len,'neighbor_heads':args.neighbor_heads,'use_neighbors':bool(args.use_neighbors),'max_longitudinal_correction':args.max_longitudinal_correction,
       'collection_mode':args.collection_mode,'max_stage_steps':args.max_stage_steps,'ttc_threshold':args.ttc_threshold,'risk_threshold':args.risk_threshold,
       'training_seed':args.training_seed,'torch_threads':args.torch_threads}
    fp={k:v for k,v in c.items() if k!='torch_threads'}; c['algorithm_fingerprint_sha256']=hashlib.sha256(json.dumps(fp,sort_keys=True,separators=(',',':')).encode()).hexdigest(); return c


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-model',required=True); p.add_argument('--lateral-adapter',required=True); p.add_argument('--output',required=True)
    p.add_argument('--split',default='longitudinal_train'); p.add_argument('--iterations',type=int,default=3); p.add_argument('--seeds-per-stage',type=int,default=1)
    p.add_argument('--mix-betas',default='1.0,0.5,0.2'); p.add_argument('--epochs',type=int,default=40); p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--learning-rate',type=float,default=3e-4); p.add_argument('--hidden-size',type=int,default=128); p.add_argument('--history-len',type=int,default=6)
    p.add_argument('--neighbor-heads',type=int,default=4); p.add_argument('--use-neighbors',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--max-longitudinal-correction',type=float,default=1.5); p.add_argument('--collection-mode',choices=['protected','mixed','raw'],default='mixed')
    p.add_argument('--training-seed',type=int,default=9400); p.add_argument('--max-stage-steps',type=int,default=420); p.add_argument('--stage-timeout',type=float,default=150.)
    p.add_argument('--ttc-threshold',type=float,default=1.8); p.add_argument('--risk-threshold',type=float,default=.68); p.add_argument('--torch-threads',type=int,default=2)
    p.add_argument('--resume',action='store_true'); p.add_argument('--worker-spec-json',help=argparse.SUPPRESS); p.add_argument('--worker-output',help=argparse.SUPPRESS)
    p.add_argument('--adapter',help=argparse.SUPPRESS); p.add_argument('--mix-beta',type=float,help=argparse.SUPPRESS); p.add_argument('--environment-safety',action=argparse.BooleanOptionalAction,default=False,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker_spec_json:return _collect_worker(args)
    if args.iterations<=0 or args.seeds_per_stage<=0 or args.epochs<=0 or args.torch_threads<=0:p.error('positive iterations/seeds/epochs/threads required')
    betas=[float(x) for x in args.mix_betas.split(',')]
    if len(betas)!=args.iterations or any(x<0 or x>1 for x in betas):p.error('one beta in [0,1] per iteration required')
    torch.set_num_threads(args.torch_threads)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    _,cfg,_=load_checkpoint(args.base_model); state_dim=20+7*cfg.env.neighbors; torch.manual_seed(args.training_seed)
    model=LongitudinalTemporalAdapter(LongitudinalResidualConfig(state_dim=state_dim,hidden_size=args.hidden_size,history_len=args.history_len,neighbor_heads=args.neighbor_heads,use_neighbors=args.use_neighbors,max_longitudinal_correction=args.max_longitudinal_correction))
    root=Path(args.output).resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:raise FileExistsError('Choose empty output or use --resume')
    root.mkdir(parents=True,exist_ok=True); schedule=[_environment_safety_for_iteration(args.collection_mode,i) for i in range(args.iterations)]
    rc=_run_config(args,betas,schedule); rcp=root/'run_config.json'
    if args.resume:
        if not rcp.is_file():raise RuntimeError('Resume requires original run_config.json')
        old=json.loads(rcp.read_text())
        if old.get('algorithm_fingerprint_sha256')!=rc['algorithm_fingerprint_sha256']:raise RuntimeError('Resume algorithm configuration mismatch')
        rc['runtime_resume_from_torch_threads']=old.get('torch_threads'); save_json(rc,rcp)
    else:save_json(rc,rcp)
    completed=json.loads((root/'iterations.json').read_text()) if args.resume and (root/'iterations.json').is_file() else []
    if not isinstance(completed,list):raise RuntimeError('Invalid iterations.json')
    parts=[]; iterations=[]; adapter_path=None; started=time.time()
    for it in range(args.iterations):
        idir=root/f'iteration_{it}'; idir.mkdir(exist_ok=True); specs=_specs(args.split,it,args.seeds_per_stage); collection=[]; envsafe=schedule[it]
        if it<len(completed):
            report=completed[it]
            if int(report.get('iteration',-1))!=it:raise RuntimeError('iterations ordering mismatch')
            for spec in specs:
                npz=idir/f'{spec.name}.npz'; meta=npz.with_suffix('.json')
                if not npz.is_file() or not meta.is_file():raise RuntimeError('completed iteration missing data')
                summary=json.loads(meta.read_text())
                if summary.get('status')!='complete' or bool(summary.get('environment_safety'))!=envsafe:raise RuntimeError('completed stage metadata mismatch')
                parts.append(npz); collection.append(summary)
            adapter_path=Path(report['artifact']['path']);
            if not adapter_path.is_file():adapter_path=root/f'adapter_iteration_{it}.pt'
            model,_=load_longitudinal_adapter(adapter_path,base_model=args.base_model,lateral_adapter=args.lateral_adapter)
            iterations.append(report); continue
        for spec in specs:
            npz=idir/f'{spec.name}.npz'; meta=npz.with_suffix('.json'); log=npz.with_suffix('.log'); reuse=False
            if args.resume and npz.is_file() and meta.is_file():
                try:
                    sm=json.loads(meta.read_text()); reuse=sm.get('status')=='complete' and sm.get('stage')==spec.name and bool(sm.get('environment_safety'))==envsafe
                except Exception:reuse=False
            if not reuse:
                npz.unlink(missing_ok=True); meta.unlink(missing_ok=True)
                cmd=[sys.executable,str(Path(__file__).resolve()),'--base-model',str(Path(args.base_model).resolve()),'--lateral-adapter',str(Path(args.lateral_adapter).resolve()),'--output',str(root),
                     '--worker-spec-json',json.dumps(spec.to_dict()),'--worker-output',str(npz),'--mix-beta',str(betas[it]),'--max-stage-steps',str(args.max_stage_steps),
                     '--ttc-threshold',str(args.ttc_threshold),'--risk-threshold',str(args.risk_threshold),'--environment-safety' if envsafe else '--no-environment-safety']
                if adapter_path:cmd+=['--adapter',str(adapter_path)]
                with log.open('w') as f:
                    proc=subprocess.run(cmd,cwd=_bootstrap.ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=args.stage_timeout,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
                if proc.returncode or not npz.is_file():raise RuntimeError(f'collection failed: {log}')
            sm=json.loads(meta.read_text()); parts.append(npz); collection.append(sm)
        arrays=_concat(parts); fit=_fit(model,arrays,args.epochs,args.learning_rate,args.batch_size,args.training_seed+it)
        adapter_path=root/f'adapter_iteration_{it}.pt'; training={'iteration':it,'mix_beta':betas[it],'environment_safety':envsafe,'fit':fit,'collection':collection,'samples_accumulated':len(arrays['state'])}
        artifact=save_longitudinal_adapter(adapter_path,model,base_model=args.base_model,lateral_adapter=args.lateral_adapter,training=training,dataset=arrays)
        report={**training,'artifact':artifact}; iterations.append(report); save_json(iterations,root/'iterations.json')
    arrays=_concat(parts); final=root/'longitudinal_temporal_adapter.pt'; training={'algorithm':'factorized_longitudinal_dagger_v41','iterations':iterations,'samples_total':len(arrays['state']),'split':args.split,'stages':list(STAGES),'collection_mode':args.collection_mode,'environment_safety_schedule':schedule,'history_len':args.history_len,'use_neighbors':bool(args.use_neighbors),'base_model_sha256':sha256(args.base_model),'lateral_adapter_sha256':sha256(args.lateral_adapter),'training_seed':args.training_seed,'algorithm_fingerprint_sha256':rc['algorithm_fingerprint_sha256'],'torch_threads':args.torch_threads,'elapsed_seconds':time.time()-started}
    artifact=save_longitudinal_adapter(final,model,base_model=args.base_model,lateral_adapter=args.lateral_adapter,training=training,dataset=arrays)
    result={**training,'artifact':artifact,'final_fit':iterations[-1]['fit'],'status':'factorized_longitudinal_adapter_trained_not_safety_certified'}; save_json(result,root/'result.json')
    print(json.dumps(result,indent=2)); sys.stdout.flush(); sys.stderr.flush(); os._exit(0)

if __name__=='__main__':main()
