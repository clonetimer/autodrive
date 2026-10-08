# v2.9 Transparent Hybrid Route-Control Research Release

v2.9 does **not** promote a new learned driving checkpoint. Several schema-v4 / lateral-distillation experiments were executed and retained as negative evidence because they did not reliably solve raw closed-loop curve/ramp failures. The released system therefore makes its rule components explicit:

```text
Frozen v2.8 neural actor
      │
      ├─ steering → pure-pursuit RouteAssist (policy-visible route preview)
      └─ longitudinal → SafetySupervisor when risk requires correction
                        ↓
                  MetaDrive 0.4.3
```

The route layer is not hidden inside the neural policy, and every route/safety intervention is recorded.

## Validation8 selection (26000-26014)

| System | Success | Collision | Off-road | Route completion | Lane error | Route intervention | Safety intervention |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw v2.8 | 26.7% | 13.3% | 60.0% | 49.7% | 0.566 m | 0.0% | 0.0% |
| + route assist | 66.7% | 26.7% | 6.7% | 83.2% | 0.063 m | 99.7% | 0.0% |
| + route + safety | 93.3% | 0.0% | 6.7% | 97.3% | 0.062 m | 99.8% | 12.9% |

The transparent hybrid was frozen after validation8. `release12_freeze.json` was written **before** any release12 stage request, and all frozen source hashes still match.

## Fresh release12 (27000-27014)

| System | Success | Collision | Off-road | Catastrophic | Route completion | Lane error | Route intervention | Safety intervention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw v2.8 | 33.3% | 13.3% | 53.3% | 66.7% | 53.2% | 0.599 m | 0.0% | 0.0% |
| v2.9 hybrid | 93.3% | 0.0% | 6.7% | 6.7% | 96.7% | 0.071 m | 99.8% | 15.4% |

Paired success: **10 improved / 1 regressed**, exact McNemar p=0.0117. Aggregate success changes +60.0pp and collision-or-off-road -60.0pp.

### Per capability

| Stage | Raw success | Hybrid success | Raw catastrophic | Hybrid catastrophic | Route completion Δ |
|---|---:|---:|---:|---:|---:|
| curve_tracking | 0.0% | 100.0% | 100.0% | 0.0% | +90.2pp |
| ramp_merge | 0.0% | 100.0% | 100.0% | 0.0% | +57.7pp |
| bottleneck_yield | 33.3% | 100.0% | 66.7% | 0.0% | +48.6pp |
| intersection_yield | 66.7% | 66.7% | 33.3% | 33.3% | +3.5pp |
| dense_progress | 66.7% | 100.0% | 33.3% | 0.0% | +17.4pp |

## Interpretation boundary

- RouteAssist intervenes on **99.8%** of evaluated time steps; this is therefore a rule-dominant lateral system result, not raw-policy performance.
- SafetySupervisor intervenes on **15.4%** of evaluated time steps and only changes longitudinal control.
- The learned v2.8 checkpoint is unchanged; no failed v2.9 lateral student was added to the curated learned-model registry.
- 15 fresh simulated scenarios are finite research evidence, not a safety certificate or real-road risk estimate.
- release12 is consumed and must not be used for further tuning. Use new validation9/release13 domains.

## Learned lateral experiments retained as negative evidence

- `route_schema4_expert_small`: 3,183 real MetaDrive samples / 10 seeds. Validation8 curve remained 100% off-road; ramp only partially improved.
- `route_schema4_steer4`: steering loss weight 4.0; curve still 100% off-road in the screened validation subset.
- `route_schema4_dagger35`: online blended DAgger from the weighted model; closed-loop validation became worse and was rejected.
- `route_head_ridge`: 3,894 route-assisted transitions. Offline steering MSE improved from **0.001336 → 0.0000596** while encoder and longitudinal actor stayed exact, yet raw curve validation still failed. This directly demonstrates closed-loop distribution shift despite a strong supervised fit.
- `dev_partial_route_schema4_expert`: larger run timed out after three stages and is explicitly marked incomplete; it is excluded from release selection.

These failures are part of the release evidence. v2.9 does not create a neural success claim by hiding them.

## Engineering changes

- `state_schema_version=4` implements arc-length-forward route preview while preserving schema3 default compatibility.
- `RouteAssist` changes only steering; `SafetySupervisor` changes only longitudinal action. Real MetaDrive integration tests assert the separation.
- Evaluation stores `proposed_action`, `route_assisted_action`, and final `applied_action`, plus separate intervention metrics.
- `models/system_registry.json` records the hybrid system separately from `models/registry.json`.
- `release12_freeze.json` was created before release12 requests and freezes the exact model/config/core implementation hashes.
