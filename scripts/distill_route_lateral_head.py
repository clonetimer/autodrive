"""Fit a learned steering head from real MetaDrive route-assisted trajectories.

Collection executes the transparent pure-pursuit route assist plus the existing
longitudinal SafetySupervisor so training states remain on-route.  Targets are
pure-pursuit steering computed from the *policy-visible state only*.  The final
checkpoint disables both runtime assists: deployment uses the learned steering
row and the untouched source longitudinal actor.
"""
from __future__ import annotations
import argparse, copy, json
from pathlib import Path
import numpy as np
import _bootstrap
from src.agents.trainer import load_checkpoint, save_checkpoint
from src.envs.metadrive_wrapper import MetaDriveParallelEnv
from src.navigation.route_assist import pure_pursuit_steering
from src.navigation.lateral_distillation import fit_lateral_head_ridge
from src.safety.curriculum import load_safety_curriculum
from src.scenarios.generator import split_seed_block
from src.utils.logger import save_json, runtime_manifest, sha256


def collect(model, base, stages, *, split: str, seed_offset: int, seeds_per_stage: int, max_steps: int):
    rows=[]
    for stage_index, stage in enumerate(stages):
        cfg=stage.apply(base);cfg.route_assist=True;cfg.route_assist_blend=1.0;cfg.safety_supervisor=True
        seeds=split_seed_block(split,seed_offset+stage_index*seeds_per_stage,seeds_per_stage)
        env=MetaDriveParallelEnv(cfg)
        try:
            for seed in seeds:
                obs,_=env.reset(seed=seed);steps=0
                while env.agents and steps < max_steps:
                    state=np.asarray(obs['agent0'],np.float32)
                    target=float(pure_pursuit_steering(state))
                    proposed=np.asarray(model.predict(state,deterministic=True),np.float32)
                    nxt,_,terms,truncs,infos=env.step({'agent0':proposed})
                    info=infos['agent0']
                    rows.append({'state':state.copy(),'target_steering':target,'source_steering':float(proposed[0]),
                                 'source_longitudinal':float(proposed[1]),'seed':int(seed),'stage':stage.name,
                                 'route_completion':float(info['route_completion']),'success':bool(info['success']),
                                 'collision':bool(info['collision']),'out_of_road':bool(info['out_of_road']),
                                 'safety_intervention':bool(info.get('safety_intervention',False))})
                    steps+=1
                    if terms['agent0'] or truncs['agent0']: break
                    obs={'agent0':nxt['agent0']}
        finally:
            env.close()
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--curriculum-config',default=str(_bootstrap.ROOT/'configs/v29_route_curriculum.yaml'))
    p.add_argument('--training-split',default='route_train');p.add_argument('--seed-offset',type=int,default=400)
    p.add_argument('--seeds-per-stage',type=int,default=3);p.add_argument('--max-steps-per-seed',type=int,default=450)
    p.add_argument('--ridge',type=float,default=1e-2)
    args=p.parse_args()
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError('Choose an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    model,cfg,payload=load_checkpoint(args.source)
    if cfg.env.backend!='metadrive' or cfg.env.observation!='state' or cfg.env.num_agents!=1:
        raise ValueError('Route-head distillation requires single-agent MetaDrive state checkpoint')
    base=copy.deepcopy(cfg.env);base.route_assist=False;base.safety_supervisor=False
    stages=load_safety_curriculum(args.curriculum_config)
    rows=collect(model,base,stages,split=args.training_split,seed_offset=args.seed_offset,
                 seeds_per_stage=args.seeds_per_stage,max_steps=args.max_steps_per_seed)
    states=np.stack([r['state'] for r in rows]).astype(np.float32)
    targets=np.asarray([r['target_steering'] for r in rows],np.float32)
    result=fit_lateral_head_ridge(model,states,targets,ridge=args.ridge,device=cfg.device)
    deploy=copy.deepcopy(cfg);deploy.env.route_assist=False;deploy.env.safety_supervisor=False
    np.savez_compressed(out/'route_lateral_dataset.npz',state=states,target_steering=targets,
                        seed=np.asarray([r['seed'] for r in rows]),stage=np.asarray([r['stage'] for r in rows],dtype='U32'))
    provenance={'method':'policy_visible_pure_pursuit_lateral_head_ridge','source_model':str(Path(args.source).resolve()),
        'source_sha256':sha256(args.source),'training_split':args.training_split,'seed_offset':args.seed_offset,
        'seeds_per_stage':args.seeds_per_stage,'samples':len(rows),'unique_seeds':len({r['seed'] for r in rows}),
        'stages':sorted({r['stage'] for r in rows}),'collection_system':'route_assist + longitudinal SafetySupervisor',
        'fit':result.to_dict(),'encoder_trained':False,'longitudinal_actor_trained':False,
        'deployment_route_assist':False,'deployment_safety_supervisor':False,
        'claim_boundary':'Steering supervision is behavior cloning from a policy-visible geometric teacher, not RL.'}
    save_checkpoint(out/'model.pt',model,deploy,optimizer=None,steps=int(payload.get('steps',0)),extra={'route_lateral_distillation':provenance})
    provenance['model_sha256']=sha256(out/'model.pt');save_json(runtime_manifest(),out/'runtime.json');save_json(provenance,out/'result.json')
    print(json.dumps(provenance,indent=2))

if __name__=='__main__':main()
