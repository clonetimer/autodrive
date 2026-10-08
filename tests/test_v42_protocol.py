from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys
from src.scenarios.generator import SPLITS,split_seeds

def load():
 root=Path(__file__).resolve().parents[1];scripts=root/'scripts';sys.path.insert(0,str(scripts)) if str(scripts) not in sys.path else None
 sp=importlib.util.spec_from_file_location('v42',scripts/'run_v42_authority_ablation.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def args(tmp): return SimpleNamespace(variants=['G0','GP','GL','GPL'],training_seeds=[12400],output=str(tmp/'o'),base_model=str(tmp/'b'),lateral_adapter=str(tmp/'l'),iterations=3,seeds_per_stage=1,mix_betas='1.0,0.5,0.2',epochs=40,batch_size=512,learning_rate=3e-4,hidden_size=128,history_len=6,neighbor_heads=4,use_neighbors=True,max_longitudinal_correction=1.5,gate_loss_weight=.25,gate_prior_probability=.05,max_stage_steps=420,stage_timeout=150.,ttc_threshold=1.8,risk_threshold=.68,torch_threads=2,resume=False)

def test_v42_four_authority_variants_are_factorial_and_training_only(tmp_path):
 m=load();jobs=m._command_plan(args(tmp_path));assert len(jobs)==4;mp={j['variant']:j for j in jobs}
 assert (mp['G0']['presence_gate'],mp['G0']['learned_gate'])==(False,False)
 assert (mp['GP']['presence_gate'],mp['GP']['learned_gate'])==(True,False)
 assert (mp['GL']['presence_gate'],mp['GL']['learned_gate'])==(False,True)
 assert (mp['GPL']['presence_gate'],mp['GPL']['learned_gate'])==(True,True)
 for j in jobs:
  c=j['train_command'];assert c[c.index('--split')+1]=='authority_train';assert '--collection-mode' in c and c[c.index('--collection-mode')+1]=='mixed';assert all('validation16' not in x and 'release19' not in x for x in c)

def test_v42_domains_fresh_and_release18_historical():
 def ds(k): lo,hi=SPLITS[k];return set(range(lo,hi+1))
 new=ds('authority_train')|ds('validation16')|ds('release19');old=ds('validation15')|ds('release18')
 assert new.isdisjoint(old);assert split_seeds('authority_train',1)==[47000];assert split_seeds('validation16',1)==[48000];assert split_seeds('release19',1)==[49000]

def test_v42_resume_only_injected_for_started_job(tmp_path):
 m=load();a=args(tmp_path);a.resume=True;j=m._command_plan(a)[0];assert '--resume' not in m._execution_command(j,True);p=Path(j['train_output']);p.mkdir(parents=True);(p/'run_config.json').write_text('{}');assert m._execution_command(j,True)[-1]=='--resume'
