# RL AutoDrive Agent v4.2.0-dev — Source Package

This is a **source-only** distribution. It intentionally excludes trained `.pt`/`.zip` weights, MetaDrive assets, wheelhouses, experiment trajectories, videos, caches, and logs.

## Included

- `src/`: agent, environments, navigation, safety, training/evaluation utilities.
- `scripts/`: training, evaluation, release/governance and v4.2 launchers.
- `tests/`: regression and protocol tests.
- `configs/`, `docs/`, `tools/`, `data/`.
- build/runtime metadata (`pyproject.toml`, requirements files, Dockerfile).
- model registries/README, but no binary weights.
- v4.2 development-selection and pre-registered validation16 protocol JSON.

## Required historical baselines for v4.2 formal training

Copy these files from the previously delivered full project/model bundle into `models/`:

- `metadrive_ppo_state_capability_distilled_v28.pt`  
  SHA-256: `e8346bd17b3cf9a78d5f987806bce19ada298841820bfa4805ced18ddef8b69c`
- `metadrive_curve_route_adapter_v32.pt`  
  SHA-256: `4230d110459468a31cad87bb3fa12fe556f3a951e66ed880e5d5ce3c44bb92a4`

MetaDrive 0.4.3 runtime/assets are external dependencies and are not redistributed in this source package.

## Current v4.2 research state

Development screening selected `GPL` (presence gate × learned intervention gate). Formal comparison is frozen to `G0 vs GPL × {13400,14400,15400}` on `authority_train`. `validation16` and `release19` remain unopened at source-package creation time.

Generate the formal train-only plan with:

```bash
python scripts/run_v42_formal_authority.py \
  --base-model models/metadrive_ppo_state_capability_distilled_v28.pt \
  --lateral-adapter models/metadrive_curve_route_adapter_v32.pt \
  --output experiments/v4_2/formal \
  --dry-run
```

Do not open `validation16` until all six formal artifacts are complete, audited and frozen. `release19` remains a final one-shot gate.
