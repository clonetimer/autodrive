import torch
from src.navigation.learned_route_adapter import LearnedRouteAdapter, RouteAdapterConfig, policy_visible_curve_score


def test_curve_score_uses_preview_without_privileged_geometry():
    straight = torch.zeros(1, 62)
    straight[:, 12:20] = torch.tensor([[.10, 0, .20, 0, .40, 0, .70, 0]])
    curve = straight.clone()
    curve[:, 13] = .12
    curve[:, 15] = .25
    curve[:, 17] = .40
    curve[:, 19] = .55
    assert float(policy_visible_curve_score(curve)) > float(policy_visible_curve_score(straight))



def test_curve_score_is_bounded():
    state = torch.zeros(4, 62)
    state[:, 13:20:2] = 5.0
    score = policy_visible_curve_score(state)
    assert torch.all((score >= 0) & (score <= 1))


def test_focused_two_seed_adapter_split_is_supported():
    import sys
    from pathlib import Path
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from scripts import train_learned_route_adapter as module
    assert callable(module._train_adapter)


def test_curve_route_adapter_direct_residual_and_identity_initialization():
    from src.navigation.learned_route_adapter import CurveRouteAdapter, CurveRouteAdapterConfig
    model = CurveRouteAdapter(CurveRouteAdapterConfig(62, hidden_size=64, max_correction=.18))
    state = torch.zeros(3, 62)
    action = torch.tensor([[.2,.1],[-.3,.4],[0.,0.]])
    steering, correction = model(state, action)
    assert torch.allclose(correction, torch.zeros_like(correction), atol=1e-7)
    assert torch.allclose(steering, action[:,0], atol=1e-7)


def test_curve_route_adapter_artifact_loads_through_common_agent_loader(tmp_path):
    from src.navigation.learned_route_adapter import CurveRouteAdapter, CurveRouteAdapterConfig, save_curve_route_adapter, load_route_adapter
    base = tmp_path / 'base.bin'; base.write_bytes(b'base-model')
    model = CurveRouteAdapter(CurveRouteAdapterConfig(62, hidden_size=64))
    path = tmp_path / 'adapter.pt'
    save_curve_route_adapter(path, model, base_model=base, training={'unit': True})
    loaded, payload = load_route_adapter(path, base_model=base)
    assert payload['format'] == 'curve_route_adapter_v2'
    assert type(loaded).__name__ == 'CurveRouteAdapter'
