"""Offline anchored re-fit over one or more real-MetaDrive expert datasets."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import _bootstrap
from src.agents.trainer import load_checkpoint, save_checkpoint
from src.experts.distillation import distill_anchored_expert
from src.utils.logger import save_json, sha256, runtime_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--dataset',action='append',required=True)
    p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--intervention-weight',type=float,default=3.0)
    p.add_argument('--longitudinal-teacher-weight',type=float,default=.25)
    p.add_argument('--reference-action-coef',type=float,default=1.0);p.add_argument('--feature-anchor-coef',type=float,default=.05)
    p.add_argument('--seed',type=int,default=2701);a=p.parse_args()
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError('Choose an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    model,cfg,payload=load_checkpoint(a.source); reference,ref_cfg,_=load_checkpoint(a.source)
    arrays=[]; sources=[]
    for raw in a.dataset:
        path=Path(raw).resolve(); data=np.load(path,allow_pickle=False)
        needed={'state','target_action','intervened','severity'}
        if not needed.issubset(data.files): raise ValueError(f'{path} missing {sorted(needed-set(data.files))}')
        arrays.append({k:data[k] for k in needed});sources.append({'path':str(path),'sha256':sha256(path),'samples':int(len(data['state']))})
    states=np.concatenate([x['state'] for x in arrays]);targets=np.concatenate([x['target_action'] for x in arrays])
    intervened=np.concatenate([x['intervened'] for x in arrays]);severity=np.concatenate([x['severity'] for x in arrays])
    result=distill_anchored_expert(model,reference,states,targets,intervened,severity,epochs=a.epochs,batch_size=a.batch_size,
        learning_rate=a.learning_rate,intervention_weight=a.intervention_weight,longitudinal_teacher_weight=a.longitudinal_teacher_weight,
        reference_action_coef=a.reference_action_coef,feature_anchor_coef=a.feature_anchor_coef,device=cfg.device,seed=a.seed)
    final=out/'model.pt';save_checkpoint(final,model,cfg,optimizer=None,steps=int(payload.get('steps',0)),extra={'anchored_expert_distillation':result.to_dict()})
    evidence={'method':'anchored_privileged_expert_distillation','source':str(Path(a.source).resolve()),'source_sha256':sha256(a.source),
              'datasets':sources,'samples':int(len(states)),'result':result.to_dict(),'model_sha256':sha256(final),'runtime':runtime_manifest(),
              'deployment_uses_expert':False,'claim_boundary':'privileged teacher data only; frozen source action/feature anchoring is an optimization constraint, not a safety guarantee'}
    save_json(evidence,out/'result.json');print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
