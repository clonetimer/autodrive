"""Create a reward-transfer warm-start checkpoint with a fresh critic/optimizer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import _bootstrap
from src.agents.transfer import transfer_actor
from src.utils.config import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result = transfer_actor(args.source, cfg, args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
