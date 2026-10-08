import copy
import numpy as np
from src.evaluation.capability_suite import standard_capability_suite
from src.evaluation.capability_selection import select_capability_candidate, capability_promotion_decision
from src.evaluation.metrics import EpisodeMetrics
from src.scenarios.generator import SPLITS, split_seeds


def _row(seed, stage, *, success=False, collision=False, offroad=False, timeout=False, completion=.5, stall=.2):
    return {'seed':seed,'suite_stage':stage,'success':success,'collision':collision,'out_of_road':offroad,
            'truncated':timeout,'route_completion':completion,'duration_s':10.0,'stall_time_fraction':stall}


def test_v28_domains_are_fresh_and_disjoint():
    for name in ('capability_train','validation3','release7','validation4','release8','validation5','release9','validation6','release10','validation7','release11'):
        lo,hi=SPLITS[name]
        values=set(split_seeds(name,hi-lo+1))
        assert values==set(range(lo,hi+1))
        for other,(olo,ohi) in SPLITS.items():
            if other!=name:
                assert values.isdisjoint(range(olo,ohi+1))


def test_capability_suite_contains_distinct_merge_yield_curve_and_dense_stages():
    suite=standard_capability_suite('validation4',2)
    assert [s.name for s in suite]==['curve_tracking','ramp_merge','bottleneck_yield','intersection_yield','dense_progress']
    assert [s.metadrive_map for s in suite]==['SCS','SrS','SyS','SXT','SSS']
    assert len({seed for s in suite for seed in s.seeds})==10
    assert suite[0].seeds==(16000,16001)
    assert suite[-1].seeds==(16008,16009)


def test_episode_metrics_records_deadlock_and_signed_progress():
    ep=EpisodeMetrics(1,'agent0')
    base={'dt':.1,'speed':.1,'lane_offset':0.,'lane_width':3.6,'jerk':0.,'collision':False,'out_of_road':False,
          'success':False,'route_completion':.2,'lane_changes':0,'progress_m':.04,'risk':{},'lateral_acceleration':0.,
          'lateral_jerk':0.,'steering_rate':0.}
    for _ in range(5):ep.update(0.,base,False,False,.1)
    moving=dict(base,speed=4.,progress_m=.5,route_completion=.4)
    for _ in range(5):ep.update(0.,moving,False,False,.1)
    r=ep.result()
    assert abs(r['stall_time_fraction']-.5)<1e-9
    assert r['forward_progress_mps']>0
    assert r['signed_progress_mps']>0


def test_capability_selection_uses_validation_domain_only_and_stage_regression_gate():
    suite=[s.to_dict() for s in standard_capability_suite('validation4',1)]
    stages=[s['name'] for s in suite]
    base_rows=[];candidate_rows=[]
    for i,(seed,stage) in enumerate(zip(range(16000,16005),stages)):
        base_rows.append(_row(seed,stage,success=(i==0),collision=(i in (1,2)),timeout=(i in (3,4)),completion=.45,stall=.35))
        candidate_rows.append(_row(seed,stage,success=(i in (0,4)),collision=(i==1),timeout=(i in (2,3)),completion=.58,stall=.18))
    reports={'base':{'suite':suite,'episodes':base_rows},'candidate':{'suite':copy.deepcopy(suite),'episodes':candidate_rows}}
    result=select_capability_candidate(reports,allowed_seed_range=SPLITS['validation4'],selection_domain='validation4')
    assert result['selected']=='candidate'
    decision=capability_promotion_decision(result,'base',max_stage_catastrophic_increase=1.0)
    assert decision['promoted'] is True
    bad=copy.deepcopy(reports)
    bad['candidate']['episodes'][0]['out_of_road']=True
    selected=select_capability_candidate(bad,allowed_seed_range=SPLITS['validation4'],selection_domain='validation4')
    # If candidate remains selected, strict stage gate must reject the local regression.
    if selected['selected']=='candidate':
        assert capability_promotion_decision(selected,'base',max_stage_catastrophic_increase=.1)['promoted'] is False


def test_capability_suite_fails_if_release_seed_is_used_for_validation_selection():
    suite=[s.to_dict() for s in standard_capability_suite('release8',1)]
    rows=[_row(spec['seeds'][0],spec['name'],success=True,completion=1.,stall=0.) for spec in suite]
    report={'suite':suite,'episodes':rows}
    import pytest
    with pytest.raises(ValueError):
        select_capability_candidate({'a':report,'b':copy.deepcopy(report)},allowed_seed_range=SPLITS['validation4'])


def test_capability_maps_reset_in_real_metadrive_when_available():
    import importlib.util, pytest
    if importlib.util.find_spec('metadrive') is None:
        pytest.skip('real MetaDrive not importable')
    from src.utils.config import EnvConfig
    from src.envs.metadrive_wrapper import MetaDriveParallelEnv
    for spec in standard_capability_suite('capability_train',1):
        cfg=spec.env(EnvConfig(backend='metadrive',observation='state',num_agents=1,max_steps=50))
        env=MetaDriveParallelEnv(cfg)
        try:
            obs,_=env.reset(seed=spec.seeds[0])
            assert obs['agent0'].shape[0]>0
            _,_,_,_,info=env.step({'agent0':np.array([0.,0.],np.float32)})
            assert info['agent0']['backend']=='metadrive'
        finally:
            env.close()


def test_bounded_component_distillation_freezes_encoder_and_limits_reference_drift():
    import copy, torch
    from src.agents.policy import DrivingPolicy
    from src.agents.distillation import actor_actions
    from src.experts.componentwise import distill_bounded_components
    torch.manual_seed(280)
    rng=np.random.default_rng(280)
    model=DrivingPolicy('state',neighbors=1,hidden_size=24)
    reference=copy.deepcopy(model)
    states=rng.normal(size=(160,27)).astype(np.float32)
    with torch.no_grad():
        ref=actor_actions(reference,torch.as_tensor(states)).cpu().numpy()
    targets=ref.copy();targets[:,0]=np.clip(targets[:,0]+.35*np.tanh(states[:,3]),-1,1);targets[:,1]=np.clip(targets[:,1]+.25,-1,1)
    intervened=np.ones(160,bool);severity=np.full(160,.7,np.float32)
    enc_before=copy.deepcopy(model.encoder.state_dict())
    result=distill_bounded_components(model,reference,states,targets,intervened,severity,epochs=8,batch_size=40,
        learning_rate=4e-4,steering_reference_coef=5.,longitudinal_reference_coef=2.,steering_margin=.1,longitudinal_margin=.15,seed=280)
    assert result.encoder_changed_l1==0
    assert result.steering_changed_l1>0 and result.longitudinal_changed_l1>0
    assert result.steering_final_teacher_mse<result.steering_initial_teacher_mse
    assert result.longitudinal_final_teacher_mse<result.longitudinal_initial_teacher_mse
    for name,value in enc_before.items():torch.testing.assert_close(model.encoder.state_dict()[name],value,rtol=0,atol=0)


def test_future_validation7_release11_are_reserved_and_disjoint():
    assert split_seeds('validation7', 3) == [22000, 22001, 22002]
    assert split_seeds('release11', 3) == [23000, 23001, 23002]
    assert set(range(*[SPLITS['validation7'][0], SPLITS['validation7'][1] + 1])).isdisjoint(
        range(SPLITS['release11'][0], SPLITS['release11'][1] + 1)
    )
