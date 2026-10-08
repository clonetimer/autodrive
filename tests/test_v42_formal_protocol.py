from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys,pytest

def load():
 root=Path(__file__).resolve().parents[1];scripts=root/'scripts';sys.path.insert(0,str(scripts)) if str(scripts) not in sys.path else None
 sp=importlib.util.spec_from_file_location('v42formal',scripts/'run_v42_formal_authority.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def args(tmp): return SimpleNamespace(training_seeds=[13400,14400,15400],output=str(tmp/'o'),base_model=str(tmp/'b'),lateral_adapter=str(tmp/'l'),iterations=3,seeds_per_stage=1,mix_betas='1.0,0.5,0.2',epochs=40,batch_size=512,learning_rate=3e-4,hidden_size=128,history_len=6,neighbor_heads=4,max_longitudinal_correction=1.5,gate_loss_weight=.25,gate_prior_probability=.05,max_stage_steps=420,stage_timeout=150.,ttc_threshold=1.8,risk_threshold=.68,torch_threads=2,resume=False)

def test_formal_plan_only_g0_vs_gpl_and_fresh_init_seeds(tmp_path):
 m=load();jobs=m._command_plan(args(tmp_path));assert len(jobs)==6
 assert {j['variant'] for j in jobs}=={'G0','GPL'};assert {j['training_seed'] for j in jobs}=={13400,14400,15400}
 assert all(j['training_seed']!=m.DEV_SEED for j in jobs)
 for j in jobs:
  c=j['train_command'];assert c[c.index('--split')+1]=='authority_train';assert c[c.index('--collection-mode')+1]=='mixed'
  assert all('validation16' not in x and 'release19' not in x for x in c)
  if j['variant']=='GPL':assert '--presence-gate' in c and '--learned-gate' in c
  else:assert '--no-presence-gate' in c and '--no-learned-gate' in c

def test_formal_parser_rejects_development_seed():
 m=load()
 with pytest.raises(Exception):m._csv_ints('12400,13400')

def test_formal_resume_only_for_started_trial(tmp_path):
 m=load();j=m._command_plan(args(tmp_path))[0];assert '--resume' not in m._exec_cmd(j,True)
 p=Path(j['train_output']);p.mkdir(parents=True);(p/'run_config.json').write_text('{}');assert m._exec_cmd(j,True)[-1]=='--resume'
