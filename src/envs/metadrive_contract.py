"""Pure configuration contract for the real official-observation CPU reference."""
from __future__ import annotations


def reference_config(*, start_seed: int, num_scenarios: int, horizon: int,
                     road_map: str = 'S', traffic_density: float = .1) -> dict:
    if min(start_seed, num_scenarios, horizon) < 0 or num_scenarios == 0 or horizon == 0:
        raise ValueError('Seed must be nonnegative, scenario count and horizon positive')
    if not 0 <= traffic_density <= 1 or not road_map:
        raise ValueError('Invalid road map or traffic density')
    return {'use_render': False, 'image_observation': False,
            'start_seed': start_seed, 'num_scenarios': num_scenarios,
            'horizon': horizon, 'map': road_map, 'traffic_density': traffic_density,
            'truncate_as_terminate': False, 'log_level': 40,
            'decision_repeat': 5, 'physics_world_step_size': .02}


def reference_plan(steps=32768, train_start=0, train_count=100,
                   eval_start=1000, eval_count=20, horizon=400,
                   road_map='S', traffic_density=.1) -> dict:
    if steps < 1:
        raise ValueError('Training steps must be positive')
    if max(train_start, eval_start) < min(train_start + train_count, eval_start + eval_count):
        raise ValueError('Training and evaluation scenario domains overlap')
    args = {'horizon': horizon, 'road_map': road_map, 'traffic_density': traffic_density}
    return {'backend': 'metadrive', 'observation': 'official_state', 'reward': 'official_default',
            'device': 'cpu', 'requested_training_steps': steps,
            'train_config': reference_config(start_seed=train_start, num_scenarios=train_count, **args),
            'evaluation_config': reference_config(start_seed=eval_start, num_scenarios=eval_count, **args),
            'evaluation_seeds': list(range(eval_start, eval_start + eval_count)),
            'checkpoint_selection': 'last_checkpoint_no_evaluation_selection',
            'model_action': '[steering, signed_throttle_brake]',
            'api': 'Gymnasium 5-tuple; reset on terminated OR truncated',
            'dt_seconds': .1, 'real_steps_executed': 0}
