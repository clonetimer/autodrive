# v2.8.0 Capability Specialization / Fresh Holdout Report

## Scope

v2.8 addresses the failure modes identified in v2.7: curve/path tracking, ramp merge, bottleneck yield, intersection yield and dense-traffic progress. The work remains a CPU/MetaDrive research project, not a real-vehicle controller or safety-certified stack.

The final v2.8 candidate uses MetaDrive `IDMPolicy` as a **training-only privileged full-action teacher**. During the final collection run, the expert action is executed 100% of the time, producing supervised state/action pairs. Deployment uses only the neural state actor; no IDM teacher or runtime safety supervisor is instantiated by the raw policy.

## Data domains and leakage boundary

Historical release domains are never reused for v2.8 candidate selection. Capability work uses disjoint ranges:

- `capability_train`: 13000-13999.
- `validation3` through `validation6`: consumed during failed v2.8 development screens and retained as negative evidence.
- `validation7`: 22000-22199. The final selection uses only the first 15 scenarios, 22000-22014.
- `release11`: 23000-23199. The final holdout uses only 23000-23024 and is now consumed.

The final model file existed at 10:37 UTC, selection JSON at 10:40 UTC, and release11 stage requests started at 10:41 UTC. A dedicated pre-release freeze JSON was not created at the time. `reports/v2_8/FREEZE_AUDIT.json` therefore records a post-hoc hash/mtime chronology audit and explicitly does **not** claim that a formal freeze manifest existed before release execution.

## Negative development screens retained

Several methods were run before the final validation7 candidate:

1. Anchored full-action DAgger on validation4: no success improvement.
2. Balanced DAgger on validation4: collisions changed into off-road failures.
3. Component-wise imitation on validation5: aggregate success improved, but off-road increased by 33.3pp and failed the promotion gate.
4. Reference-bounded component adaptation on validation6: no success or catastrophic improvement and lower route completion.

Because each screen consumed its validation domain, the next candidate moved to a fresh validation split rather than repeatedly tuning on the same observations.

## Final training run

`experiments/v2_8/capability_pureexpert_v4/model.pt`

- Source: `models/metadrive_ppo_state_expert_distilled_v27.pt`.
- Capability stages: curve tracking, ramp merge, bottleneck yield, intersection yield, dense progress.
- Training seeds: 13500-13529, 30 unique seeds.
- Collected transitions: 9,000.
- Expert execution during collection: 100% for steering and longitudinal action.
- Terminal collection episodes observed: 15; 14 successes, 1 collision, 0 off-road.
- Distillation: encoder + actor supervised fitting; critic and exploration log-std are not trained in this behavior-cloning phase.
- Final model SHA-256: `e8346bd17b3cf9a78d5f987806bce19ada298841820bfa4805ced18ddef8b69c`.

## Validation7 selection

15 frozen validation scenarios (3 per capability stage):

| Policy | Success | Collision | Off-road | Catastrophic | Route completion |
|---|---:|---:|---:|---:|---:|
| v2.7 source | 6.7% | 40.0% | 53.3% | 93.3% | 52.8% |
| v2.8 pure-expert BC | **26.7%** | **20.0%** | 53.3% | **73.3%** | 52.7% |

The predeclared engineering gate passed because success improved by 20pp and catastrophic failure fell by 20pp without collision/off-road bounds being violated. This gate is model promotion logic, not statistical safety evidence.

## Fresh release11 holdout

25 unseen scenarios, five per capability stage:

| Policy | Success | Collision | Off-road | Catastrophic | Route completion | Avg speed |
|---|---:|---:|---:|---:|---:|---:|
| v2.7 source | 4.0% | 32.0% | 64.0% | 96.0% | 46.9% | 8.15 m/s |
| **v2.8 capability BC** | **32.0%** | **12.0%** | **56.0%** | **68.0%** | **56.2%** | 6.59 m/s |

Aggregate changes:

- success: +28pp;
- collision: -20pp;
- off-road: -8pp;
- collision-or-off-road: -28pp;
- route completion: +9.25pp.

Paired success outcome across the same 25 stage/seed scenarios: 8 improved, 1 regressed, 0 both-success, 16 both-fail; descriptive exact McNemar p=0.0391. This finite paired result is useful research evidence but does not estimate real-road safety.

## Capability-level release11 behavior

| Stage | Source success | v2.8 success | Source catastrophic | v2.8 catastrophic | Route completion change |
|---|---:|---:|---:|---:|---:|
| Curve tracking | 0% | 0% | 100% | 100% | -11.3pp |
| Ramp merge | 20% | 0% | 80% | 100% | -24.0pp |
| Bottleneck yield | 0% | 40% | 100% | 60% | +23.5pp |
| Intersection yield | 0% | 60% | 100% | 40% | +23.2pp |
| Dense progress | 0% | 60% | 100% | 40% | +34.9pp |

The holdout therefore confirms a meaningful **aggregate capability gain**, especially for bottleneck/intersection/dense progress, but explicitly exposes two unresolved failures: curve tracking remains 100% off-road and ramp-merge success regresses. v2.8 is not a universal capability pass.

## Interpretation

The strongest v2.8 conclusion is that broad, pure-expert full-action state/action coverage generalizes better than the earlier bounded DAgger variants on the aggregate capability suite. It does **not** show that imitation learning alone solves all driving skills. The remaining curve/ramp failures suggest that route representation, lateral planning and topology-conditioned behavior need dedicated treatment in v2.9 rather than further reuse of release11.

## Next experiment rule

`release11` is permanently consumed. Future tuning must introduce a new training/validation/release domain. A sensible next direction is topology-aware route encoding plus separate curve/ramp curricula and multi-source PPO seeds, with release11 retained only as historical evidence.

## Safety boundary

All results are finite MetaDrive simulations. Privileged expert data are training-only. No policy in this repository is certified for road use, and project artifacts must not control a real vehicle.
