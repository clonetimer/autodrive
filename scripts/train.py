"""Train a real policy; dependency failure never changes the requested backend."""
import argparse
import json
from pathlib import Path
import _bootstrap
from src.agents.trainer import train
from src.utils.config import load_config
from src.utils.logger import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--resume')
    args = parser.parse_args()
    overrides = {}
    if args.steps is not None:
        overrides['total_steps'] = args.steps
    if args.seed is not None:
        overrides.update(seed=args.seed, env={'seed': args.seed})
    config = load_config(args.config, overrides)
    try:
        result = train(config, args.output, args.resume)
    except Exception as exc:
        save_json({'status': 'failed', 'exception': type(exc).__name__, 'message': str(exc),
                   'requested_backend': config.env.backend}, Path(args.output) / 'failure.json')
        raise
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
