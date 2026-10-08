"""Reference-bounded component-wise expert distillation.

The v2.8 development screen showed that unconstrained full-action imitation can
trade collisions for off-road failures. This optimizer keeps the shared state
encoder frozen and fits steering/longitudinal actor rows independently while a
soft margin penalizes drift from a frozen source policy. The expert remains
training-only; deployment uses the learned actor alone.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import copy
import numpy as np
import torch
from torch import nn
from src.agents.distillation import actor_actions


@dataclass(frozen=True)
class BoundedComponentResult:
    samples: int
    intervention_samples: int
    epochs: int
    batches: int
    steering_initial_teacher_mse: float
    steering_final_teacher_mse: float
    longitudinal_initial_teacher_mse: float
    longitudinal_final_teacher_mse: float
    steering_reference_mse: float
    longitudinal_reference_mse: float
    steering_changed_l1: float
    longitudinal_changed_l1: float
    encoder_changed_l1: float
    steering_reference_coef: float
    longitudinal_reference_coef: float
    steering_margin: float
    longitudinal_margin: float

    def to_dict(self):
        return asdict(self)


def _validate(states, targets, intervened, severity):
    states=np.asarray(states,np.float32);targets=np.asarray(targets,np.float32)
    intervened=np.asarray(intervened,bool);severity=np.asarray(severity,np.float32)
    n=len(states)
    if n==0 or states.ndim!=2 or targets.shape!=(n,2) or intervened.shape!=(n,) or severity.shape!=(n,):
        raise ValueError('componentwise expert arrays must align as state[N,D], target[N,2], masks[N]')
    if not np.isfinite(states).all() or not np.isfinite(targets).all() or not np.isfinite(severity).all():
        raise ValueError('componentwise expert arrays must be finite')
    if np.any((severity<0)|(severity>1)):raise ValueError('severity must be in [0,1]')
    return states,targets,intervened,severity


def distill_bounded_components(model, reference_model, states, targets, intervened, severity, *,
                               epochs:int=12,batch_size:int=256,learning_rate:float=2e-4,
                               intervention_weight:float=2.5,steering_reference_coef:float=4.0,
                               longitudinal_reference_coef:float=1.0,steering_margin:float=.12,
                               longitudinal_margin:float=.25,device:str='cpu',seed:int=0)->BoundedComponentResult:
    states,targets,intervened,severity=_validate(states,targets,intervened,severity)
    if min(epochs,batch_size)<=0 or learning_rate<=0 or intervention_weight<0:
        raise ValueError('invalid componentwise optimizer settings')
    if min(steering_reference_coef,longitudinal_reference_coef,steering_margin,longitudinal_margin)<0:
        raise ValueError('reference coefficients/margins must be nonnegative')
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=model.to(device);reference_model=reference_model.to(device);reference_model.eval()
    for p in reference_model.parameters():p.requires_grad_(False)
    before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n.startswith('encoder.') or n.startswith('actor.')}
    # Strict invariant: feature representation is frozen in this adaptation.
    for p in model.encoder.parameters():p.requires_grad_(False)
    state_t=torch.as_tensor(states,device=device)
    target_t=torch.as_tensor(targets,device=device)
    with torch.no_grad():ref_all=actor_actions(reference_model,state_t).detach()
    mask=np.asarray(intervened,np.float32);weights_np=1.0+intervention_weight*mask*(.5+.5*severity)

    def teacher_mse(component:int):
        with torch.no_grad():pred=actor_actions(model,state_t)[:,component]
        return float(nn.functional.mse_loss(pred,target_t[:,component]).item())

    initial=[teacher_mse(0),teacher_mse(1)];batches=0
    configs=[(0,steering_reference_coef,steering_margin),(1,longitudinal_reference_coef,longitudinal_margin)]
    for component,ref_coef,margin in configs:
        optimizer=torch.optim.Adam(model.actor.parameters(),lr=learning_rate)
        model.train()
        for _ in range(epochs):
            order=rng.permutation(len(states))
            for start in range(0,len(states),batch_size):
                idx=order[start:start+batch_size]
                obs=torch.as_tensor(states[idx],device=device);tgt=torch.as_tensor(targets[idx,component],device=device)
                w=torch.as_tensor(weights_np[idx],device=device)
                with torch.no_grad():ref=actor_actions(reference_model,obs)[:,component]
                pred=actor_actions(model,obs)[:,component]
                teacher=((pred-tgt).square()*w).sum()/w.sum().clamp_min(1e-8)
                drift=(pred-ref).abs()
                reference_penalty=torch.relu(drift-margin).square().mean()
                loss=teacher+ref_coef*reference_penalty
                if not torch.isfinite(loss):raise FloatingPointError('non-finite bounded component loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                other=1-component
                if model.actor.weight.grad is not None:model.actor.weight.grad[other].zero_()
                if model.actor.bias.grad is not None:model.actor.bias.grad[other].zero_()
                nn.utils.clip_grad_norm_(model.actor.parameters(),5.0);optimizer.step();batches+=1
    model.eval()
    for p in model.encoder.parameters():p.requires_grad_(True)
    final=[teacher_mse(0),teacher_mse(1)]
    with torch.no_grad():pred=actor_actions(model,state_t)
    ref_mse=[float(nn.functional.mse_loss(pred[:,i],ref_all[:,i]).item()) for i in (0,1)]
    enc=steer=longitudinal=0.0
    for name,p in model.named_parameters():
        if name not in before:continue
        delta=(p.detach().cpu()-before[name]).abs()
        if name.startswith('encoder.'):enc+=float(delta.sum())
        elif name=='actor.weight':steer+=float(delta[0].sum());longitudinal+=float(delta[1].sum())
        elif name=='actor.bias':steer+=float(delta[0]);longitudinal+=float(delta[1])
    return BoundedComponentResult(len(states),int(intervened.sum()),epochs,batches,initial[0],final[0],initial[1],final[1],
        ref_mse[0],ref_mse[1],steer,longitudinal,enc,float(steering_reference_coef),float(longitudinal_reference_coef),
        float(steering_margin),float(longitudinal_margin))
