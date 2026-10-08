import numpy as np
import torch
from src.agents.policy import DrivingPolicy
from src.navigation.lateral_distillation import fit_lateral_head_ridge


def test_ridge_lateral_fit_preserves_encoder_and_longitudinal_row():
    torch.manual_seed(3)
    model=DrivingPolicy('state',neighbors=1,hidden_size=16)
    states=np.random.default_rng(4).normal(size=(256,27)).astype(np.float32)
    with torch.no_grad():
        features=model.encoder(torch.as_tensor(states)).numpy()
    latent=.25*features[:,0]-.12*features[:,1]+.03
    targets=np.tanh(latent).astype(np.float32)
    before_enc={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('encoder.')}
    before_long_w=model.actor.weight[1].detach().clone();before_long_b=model.actor.bias[1].detach().clone()
    result=fit_lateral_head_ridge(model,states,targets,ridge=1e-6)
    assert result.final_action_mse < result.initial_action_mse
    assert result.final_action_mse < 1e-6
    assert result.encoder_changed_l1 == 0
    assert result.longitudinal_actor_changed_l1 == 0
    for n,p in model.named_parameters():
        if n in before_enc: torch.testing.assert_close(p,before_enc[n])
    torch.testing.assert_close(model.actor.weight[1],before_long_w);torch.testing.assert_close(model.actor.bias[1],before_long_b)
