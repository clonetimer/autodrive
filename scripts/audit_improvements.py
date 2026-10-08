"""Reproduce the reward/observation defects and their opt-in fixes with real steps."""
from dataclasses import replace
import argparse
from pathlib import Path
import _bootstrap
import numpy as np
from src.envs.native_world import NativeWorld
from src.utils.config import EnvConfig
from src.utils.logger import save_json, sha256


def audit():
    results = {'reward': {}, 'observation': {}, 'scope': 'controlled regression evidence, not trained driving safety'}
    for version in (2, 3):
        rows = []
        for speed in (12, 24):
            world = NativeWorld(EnvConfig(traffic_vehicles=0, target_speed=12, initial_speed=speed,
                                          reward_version=version))
            try:
                world.reset(seed=0)
                ego = world.vehicles['agent0']
                # Compensate the same 0.015*v drag term used in vehicle dynamics.
                observation, reward, term, trunc, info = world.step({'agent0': [0, .015 * speed / 3]})
                rows.append({'initial_speed_mps': speed, 'actual_speed_mps': info['agent0']['speed'],
                             'reward': reward['agent0'], 'terms': info['agent0']['reward_terms']})
            finally:
                world.close()
        results['reward'][f'v{version}'] = {'rows': rows, 'target_preferred_to_double_speed': rows[0]['reward'] > rows[1]['reward']}
    for mode in ('bev', 'bev_ego'):
        world = NativeWorld(EnvConfig(observation=mode, traffic_vehicles=0))
        try:
            world.reset(seed=0)
            world.vehicles['agent0'].speed = 8
            slow = world.observe('agent0')
            world.vehicles['agent0'].speed = 24
            fast = world.observe('agent0')
            results['observation'][mode] = {key: {'equal': bool(np.array_equal(slow[key], fast[key])),
                'changed_elements': int(np.count_nonzero(slow[key] != fast[key]))} for key in slow}
        finally:
            world.close()
    results['checks'] = {'legacy_defect_reproduced': not results['reward']['v2']['target_preferred_to_double_speed'],
                        'new_reward_check_passed': results['reward']['v3']['target_preferred_to_double_speed'],
                        'legacy_geometry_alias_retained': results['observation']['bev']['bev']['equal'],
                        'ego_motion_alias_resolved': not results['observation']['bev_ego']['motion']['equal']}
    results['source_sha256'] = {name: sha256(_bootstrap.ROOT / name) for name in
        ['src/envs/reward.py', 'src/envs/native_world.py', 'src/envs/observation.py', 'src/perception/history.py']}
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='reports/v2_1/improvement_audit.json')
    args = parser.parse_args()
    result = audit()
    save_json(result, args.output)
    print(result['checks'])
    if not all(result['checks'].values()):
        raise SystemExit(1)
