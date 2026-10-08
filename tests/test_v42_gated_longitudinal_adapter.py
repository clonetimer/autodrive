from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from src.navigation.gated_longitudinal_adapter import (
    GatedLongitudinalConfig,GatedLongitudinalTemporalAdapter,
    save_gated_longitudinal_adapter,load_gated_longitudinal_adapter)

def hist(batch=2,visible=True):
 x=torch.zeros(batch,6,62);x[...,12:20]=torch.tensor([.1,.01,.2,.02,.4,.03,.7,.05])
 if visible:
  for n in range(6): x[...,20+7*n:20+7*n+7]=torch.tensor([.1,.0,0.,0.,0.,1.,1.])
 return x

def test_v42_identity_and_exact_steering():
 m=GatedLongitudinalTemporalAdapter(GatedLongitudinalConfig(62,hidden_size=64))
 base=torch.tensor([[.2,.4],[-.3,-.2]])
 out,raw,eff,auth,prob,_,presence,_=m(hist(),base)
 assert torch.equal(out[:,0],base[:,0]); assert torch.allclose(out[:,1],base[:,1],atol=1e-7)
 assert torch.allclose(raw,torch.zeros_like(raw));assert torch.allclose(eff,torch.zeros_like(eff))
 assert torch.all((auth>=0)&(auth<=1));assert torch.all(presence==1)
 assert torch.allclose(prob,torch.full_like(prob,.05),atol=1e-6)

def test_v42_presence_gate_structurally_blocks_nonzero_residual():
 m=GatedLongitudinalTemporalAdapter(GatedLongitudinalConfig(62,hidden_size=64,presence_gate=True,learned_gate=False))
 with torch.no_grad(): m.residual_head.bias.fill_(.8)
 base=torch.tensor([[.123,.2],[-.456,.1]])
 out,raw,eff,auth,_,_,presence,_=m(hist(2,visible=False),base)
 assert torch.equal(out[:,0],base[:,0]);assert torch.any(raw.abs()>0)
 assert torch.equal(presence,torch.zeros_like(presence));assert torch.equal(auth,torch.zeros_like(auth));assert torch.equal(eff,torch.zeros_like(eff));assert torch.equal(out[:,1],base[:,1])

def test_v42_g0_remains_always_authorized_without_visible_neighbours():
 m=GatedLongitudinalTemporalAdapter(GatedLongitudinalConfig(62,hidden_size=64,presence_gate=False,learned_gate=False))
 with torch.no_grad(): m.residual_head.bias.fill_(.4)
 base=torch.tensor([[.1,.0]])
 out,raw,eff,auth,_,_,presence,_=m(hist(1,visible=False),base)
 assert torch.equal(auth,torch.ones_like(auth));assert torch.equal(presence,torch.ones_like(presence));assert torch.allclose(eff,raw);assert out[0,1]!=base[0,1]

def test_v42_artifact_binds_lateral_sha(tmp_path):
 b=tmp_path/'b';b.write_bytes(b'b');l=tmp_path/'l';l.write_bytes(b'l');w=tmp_path/'w';w.write_bytes(b'w')
 m=GatedLongitudinalTemporalAdapter(GatedLongitudinalConfig(62,hidden_size=64));p=tmp_path/'g.pt'
 save_gated_longitudinal_adapter(p,m,base_model=b,lateral_adapter=l,training={'unit':True})
 loaded,payload=load_gated_longitudinal_adapter(p,base_model=b,lateral_adapter=l);assert payload['format']=='gated_longitudinal_temporal_adapter_v42';assert isinstance(loaded,GatedLongitudinalTemporalAdapter)
 with pytest.raises(ValueError,match='different lateral adapter'): load_gated_longitudinal_adapter(p,base_model=b,lateral_adapter=w)

def test_v42_fit_preserves_steering_and_zero_presence_authority():
 scripts=str(Path(__file__).resolve().parents[1]/'scripts');sys.path.insert(0,scripts) if scripts not in sys.path else None
 from scripts.train_gated_longitudinal_adapter import _fit
 rng=np.random.default_rng(2);n=60;states=np.zeros((n,62),np.float32);states[:,12:20]=rng.normal(size=(n,8)).astype(np.float32)
 # No visible neighbours anywhere: presence-gated effective correction must remain zero.
 lateral=rng.uniform(-.5,.5,size=(n,2)).astype(np.float32);teacher=lateral[:,1].copy();inter=np.zeros(n,bool);inter[::10]=True;teacher[::10]=np.clip(teacher[::10]-.3,-1,1)
 arrays={'state':states,'lateral_action':lateral,'teacher_longitudinal':teacher.astype(np.float32),'counterfactual_safety_severity':inter.astype(np.float32)*.8,'counterfactual_safety_intervention':inter,'seed':np.repeat([1,2,3],20),'step':np.tile(np.arange(20),3)}
 m=GatedLongitudinalTemporalAdapter(GatedLongitudinalConfig(62,hidden_size=32,neighbor_heads=4,presence_gate=True,learned_gate=True))
 fit=_fit(m,arrays,2,3e-4,32,7,.25);d=fit['diagnostic_holdout'];assert d['steering_passthrough_max_abs_error']==0.;assert d['zero_presence_mean_abs_effective_correction']==0.
