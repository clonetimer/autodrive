"""Distil full-action MetaDrive IDM capability targets on fresh training domains.

Unlike v2.7 lateral-only expert distillation, this collector teaches both
steering and longitudinal progress.  The teacher is privileged and training-
only.  Raw deployment uses only the normal project state observation and never
instantiates IDMPolicy.  Collection uses blended DAgger execution so the
student sees states near its own distribution while avoiding obviously stalled
or off-route trajectories.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import _bootstrap
from src.agents.trainer import load_checkpoint, save_checkpoint
from src.envs.metadrive_wrapper import MetaDriveParallelEnv
from src.experts.metadrive_idm import MetaDriveIDMTeacher
from src.experts.distillation import distill_anchored_expert
from src.experts.componentwise import distill_bounded_components
from src.safety.curriculum import load_safety_curriculum
from src.safety.distillation import distill_safety_actor
from src.scenarios.generator import SPLITS, split_seed_block
from src.utils.config import load_config
from src.utils.logger import runtime_manifest, save_json, sha256


def capability_seed_block(start: int, count: int, split: str = 'capability_train') -> list[int]:
    """Backward-compatible wrapper around the canonical split helper."""
    return split_seed_block(split, start, count)


def collect(policy, env_config, seeds: list[int], max_steps_per_seed: int, stage_name: str,
            *, steering_blend: float=.7, longitudinal_blend: float=.7) -> list[dict]:
    if not (0 < steering_blend <= 1 and 0 < longitudinal_blend <= 1):
        raise ValueError('Capability DAgger blend weights must lie in (0,1]')
    env = MetaDriveParallelEnv(env_config)
    rows=[]
    try:
        for seed in seeds:
            observations,_=env.reset(seed=seed)
            teacher=MetaDriveIDMTeacher(env,'agent0',seed=seed,target_speed_mps=env_config.target_speed)
            steps=0
            while env.agents and steps < max_steps_per_seed:
                state=np.asarray(observations['agent0'],np.float32)
                proposed=np.asarray(policy.predict(state,deterministic=True),np.float32)
                expert=teacher.decide()
                executed=proposed.copy()
                executed[0]=np.clip((1-steering_blend)*proposed[0]+steering_blend*expert.steering,-1,1)
                executed[1]=np.clip((1-longitudinal_blend)*proposed[1]+longitudinal_blend*expert.longitudinal,-1,1)
                next_obs,_,terms,truncs,infos=env.step({'agent0':executed})
                info=infos['agent0']
                delta=np.abs(expert.action-proposed)
                severity=float(np.clip(max(float(delta[0]),float(delta[1])),0,1))
                rows.append({
                    'state':state.copy(),'proposed_action':proposed.copy(),'target_action':expert.action.copy(),
                    'executed_action':executed.copy(),'intervened':bool(delta.max()>.03),'severity':severity,
                    'steering_correction':float(delta[0]),'longitudinal_correction':float(delta[1]),
                    'seed':int(seed),'step':int(info['step']),'stage':stage_name,
                    'collision':bool(info['collision']),'out_of_road':bool(info['out_of_road']),
                    'success':bool(info['success']),'truncated':bool(truncs['agent0']),
                    'route_completion':float(info['route_completion']),'speed':float(info['speed']),
                    'progress_m':float(info.get('progress_m',0.0)),
                })
                steps+=1
                if terms['agent0'] or truncs['agent0']:
                    break
                observations={'agent0':next_obs['agent0']}
        if not rows:
            raise RuntimeError('No capability transitions were collected')
        return rows
    finally:
        env.close()


def arrays(rows):
    return {
        'state':np.stack([r['state'] for r in rows]).astype(np.float32),
        'target_action':np.stack([r['target_action'] for r in rows]).astype(np.float32),
        'proposed_action':np.stack([r['proposed_action'] for r in rows]).astype(np.float32),
        'intervened':np.asarray([r['intervened'] for r in rows],bool),
        'severity':np.asarray([r['severity'] for r in rows],np.float32),
        'seed':np.asarray([r['seed'] for r in rows],np.int32),
        'step':np.asarray([r['step'] for r in rows],np.int32),
        'stage':np.asarray([r['stage'] for r in rows],dtype='U32'),
    }


def summarize(rows):
    terminal=[r for r in rows if r['success'] or r['collision'] or r['out_of_road'] or r['truncated']]
    return {
        'samples':len(rows),'unique_seeds':len({r['seed'] for r in rows}),
        'stages':sorted({r['stage'] for r in rows}),
        'intervention_fraction':float(np.mean([r['intervened'] for r in rows])),
        'mean_steering_correction':float(np.mean([r['steering_correction'] for r in rows])),
        'mean_longitudinal_correction':float(np.mean([r['longitudinal_correction'] for r in rows])),
        'mean_progress_m_per_transition':float(np.mean([r['progress_m'] for r in rows])),
        'terminal_episodes':len(terminal),'terminal_successes':sum(r['success'] for r in terminal),
        'terminal_collisions':sum(r['collision'] for r in terminal),
        'terminal_offroad':sum(r['out_of_road'] for r in terminal),
        'terminal_timeouts':sum(r['truncated'] and not r['success'] for r in terminal),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True); p.add_argument('--output',required=True)
    p.add_argument('--collection-config')
    p.add_argument('--curriculum-config',default=str(_bootstrap.ROOT/'configs/v28_capability_curriculum.yaml'))
    p.add_argument('--seed-offset',type=int,default=0);p.add_argument('--seeds-per-stage',type=int,default=3)
    p.add_argument('--training-split',default='capability_train',choices=sorted(SPLITS))
    p.add_argument('--state-schema-version',type=int,choices=[3,4])
    p.add_argument('--max-steps-per-seed',type=int,default=260);p.add_argument('--epochs-per-stage',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--intervention-weight',type=float,default=2.0)
    p.add_argument('--steering-weight',type=float,default=1.0,help='MSE weight for steering relative to longitudinal action')
    p.add_argument('--steering-blend',type=float,default=.7);p.add_argument('--longitudinal-blend',type=float,default=.7)
    p.add_argument('--mode',choices=['actor_only','balanced','anchored','componentwise','componentwise_anchored'],default='actor_only')
    p.add_argument('--reference-action-coef',type=float,default=1.5)
    p.add_argument('--feature-anchor-coef',type=float,default=.08)
    p.add_argument('--steering-reference-coef',type=float,default=4.0)
    p.add_argument('--longitudinal-reference-coef',type=float,default=1.0)
    p.add_argument('--steering-margin',type=float,default=.12)
    p.add_argument('--longitudinal-margin',type=float,default=.25)
    args=p.parse_args()
    if min(args.seeds_per_stage,args.max_steps_per_seed,args.epochs_per_stage,args.batch_size)<=0 or args.seed_offset<0:
        p.error('count arguments must be positive and seed-offset nonnegative')
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError('Choose an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    model,source_config,payload=load_checkpoint(args.source)
    if source_config.env.backend!='metadrive' or source_config.env.observation!='state' or source_config.env.num_agents!=1:
        raise ValueError('Capability distillation requires single-agent MetaDrive state PPO')
    reference=copy.deepcopy(model)
    base=copy.deepcopy(load_config(args.collection_config).env if args.collection_config else source_config.env)
    base.safety_supervisor=False
    if args.state_schema_version is not None:
        base.state_schema_version=args.state_schema_version
    deploy_config=copy.deepcopy(source_config);deploy_config.env=copy.deepcopy(base);deploy_config.env.safety_supervisor=False
    stages=load_safety_curriculum(args.curriculum_config)
    save_json(runtime_manifest(),out/'runtime.json')
    all_rows=[];evidence=[]
    for index,stage in enumerate(stages):
        env_config=stage.apply(base);env_config.safety_supervisor=False
        seeds=capability_seed_block(args.seed_offset+index*args.seeds_per_stage,args.seeds_per_stage,args.training_split)
        rows=collect(model,env_config,seeds,args.max_steps_per_seed,stage.name,
                     steering_blend=args.steering_blend,longitudinal_blend=args.longitudinal_blend)
        all_rows.extend(rows);data=arrays(all_rows)
        if args.mode=='anchored':
            result=distill_anchored_expert(model,reference,data['state'],data['target_action'],data['intervened'],data['severity'],
                epochs=args.epochs_per_stage,batch_size=args.batch_size,learning_rate=args.learning_rate,
                intervention_weight=args.intervention_weight,longitudinal_teacher_weight=1.0,
                reference_action_coef=args.reference_action_coef,feature_anchor_coef=args.feature_anchor_coef,
                device=source_config.device,seed=source_config.seed+index+2800)
            distillation_evidence=result.to_dict()
        elif args.mode=='componentwise':
            steering_result=distill_safety_actor(model,data['state'],data['target_action'],data['intervened'],data['severity'],
                epochs=args.epochs_per_stage,batch_size=args.batch_size,learning_rate=args.learning_rate,
                intervention_weight=args.intervention_weight,steering_weight=args.steering_weight,train_encoder=False,steering_only=True,
                device=source_config.device,seed=source_config.seed+index+2800)
            longitudinal_result=distill_safety_actor(model,data['state'],data['target_action'],data['intervened'],data['severity'],
                epochs=args.epochs_per_stage,batch_size=args.batch_size,learning_rate=args.learning_rate,
                intervention_weight=args.intervention_weight,steering_weight=args.steering_weight,train_encoder=False,longitudinal_only=True,
                device=source_config.device,seed=source_config.seed+index+3800)
            distillation_evidence={'steering':steering_result.to_dict(),'longitudinal':longitudinal_result.to_dict(),
                'invariant':'shared encoder is frozen in both passes; each actor row is updated independently'}
        elif args.mode=='componentwise_anchored':
            bounded=distill_bounded_components(model,reference,data['state'],data['target_action'],data['intervened'],data['severity'],
                epochs=args.epochs_per_stage,batch_size=args.batch_size,learning_rate=args.learning_rate,
                intervention_weight=args.intervention_weight,steering_reference_coef=args.steering_reference_coef,
                longitudinal_reference_coef=args.longitudinal_reference_coef,steering_margin=args.steering_margin,
                longitudinal_margin=args.longitudinal_margin,device=source_config.device,seed=source_config.seed+index+4800)
            distillation_evidence=bounded.to_dict()
        else:
            result=distill_safety_actor(model,data['state'],data['target_action'],data['intervened'],data['severity'],
                epochs=args.epochs_per_stage,batch_size=args.batch_size,learning_rate=args.learning_rate,
                intervention_weight=args.intervention_weight,steering_weight=args.steering_weight,
                train_encoder=(args.mode=='balanced'),device=source_config.device,seed=source_config.seed+index+2800)
            distillation_evidence=result.to_dict()
        stage_evidence={'index':index,'stage':stage.to_dict(),'seeds':seeds,
                        'new_collection':summarize(rows),'aggregate_collection':summarize(all_rows),
                        'distillation':distillation_evidence}
        evidence.append(stage_evidence)
        save_checkpoint(out/f'stage_{index+1}.pt',model,deploy_config,optimizer=None,steps=int(payload.get('steps',0)),
                        extra={'capability_distillation':stage_evidence})
    data=arrays(all_rows);np.savez_compressed(out/'capability_dataset.npz',**data)
    provenance={'method':'full_action_metadrive_idm_capability_dagger','source_model':str(Path(args.source).resolve()),
        'source_sha256':sha256(args.source),'source_rl_world_steps':int(payload.get('steps',0)),'mode':args.mode,
        'steering_blend':args.steering_blend,'longitudinal_blend':args.longitudinal_blend,
        'training_split':args.training_split,'capability_train_seed_offset':args.seed_offset,
        'state_schema_version':deploy_config.env.state_schema_version,'steering_weight':args.steering_weight,
        'curriculum':[s.to_dict() for s in stages],
        'stages':evidence,'aggregate_collection':summarize(all_rows),'critic_trained':False,'log_std_trained':False,
        'deployment_uses_expert':False,'deployment_safety_supervisor_enabled':False,
        'claim_boundary':'IDM is a privileged training teacher only; release evaluation uses the raw learned actor.'}
    final=out/'model.pt';save_checkpoint(final,model,deploy_config,optimizer=None,steps=int(payload.get('steps',0)),extra={'capability_distillation':provenance})
    provenance['model_sha256']=sha256(final);save_json(provenance,out/'result.json');print(json.dumps(provenance,indent=2))

if __name__=='__main__':main()
