import pytest
from src.evaluation.route_adapter_selection import select_route_adapter


def rows(successes, stage='curve_tracking', start=1, collision=False, offroad=False):
    out=[]
    for i,s in enumerate(successes):
        out.append({'suite_stage':stage,'seed':start+i,'success':s,
                    'collision':collision if not s else False,'out_of_road':offroad if not s else False})
    return out


def test_route_adapter_gate_promotes_real_success_and_failure_reduction():
    base = rows([False,False,False], offroad=True) + rows([True,False,True], 'ramp_merge', 10, offroad=True)
    cand = rows([True,True,True]) + rows([True,True,True], 'ramp_merge', 10)
    result=select_route_adapter(base,cand)
    assert result['promoted']
    assert result['candidate_stages']['curve_tracking']['success_rate']==1.0


def test_route_adapter_gate_rejects_collision_to_offroad_trade():
    base = rows([False,False,False], collision=True) + rows([True,True,True], 'ramp_merge', 10)
    cand = rows([True,True,True]) + rows([False,False,False], 'ramp_merge', 10, offroad=True)
    result=select_route_adapter(base,cand)
    assert not result['promoted']
    assert not result['checks']['stage_success_non_regression']


def test_route_adapter_gate_requires_exact_scenario_pairing():
    with pytest.raises(ValueError):
        select_route_adapter(rows([False,False]), rows([True,True], start=2))
