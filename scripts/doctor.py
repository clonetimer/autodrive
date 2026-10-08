"""Record real environment prerequisites; --smoke does actual MetaDrive steps."""
import argparse
from pathlib import Path
import sys
import _bootstrap
from src.envs.metadrive_wrapper import prerequisite_status, MetaDriveParallelEnv
from src.utils.config import EnvConfig
from src.utils.logger import save_json, runtime_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', default='reports/metadrive_doctor.json')
    args = parser.parse_args()
    report = {'runtime': runtime_manifest(), 'metadrive': prerequisite_status(args.source),
              'real_steps_executed': 0, 'smoke_test': 'not_requested'}
    exit_code = 0
    if args.smoke:
        env = None
        try:
            env = MetaDriveParallelEnv(EnvConfig(backend='metadrive', max_steps=40))
            obs, _ = env.reset(seed=0)
            for _ in range(40):
                env.step({name: env.action_space.sample() for name in env.agents})
                report['real_steps_executed'] += 1
                if not env.agents:
                    env.reset()
            report['smoke_test'] = 'passed'
        except Exception as exc:
            report.update(smoke_test='blocked_or_failed', exception=type(exc).__name__, message=str(exc))
            exit_code = 2
        finally:
            if env is not None:
                env.close()
    save_json(report, args.output)
    print(report)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
