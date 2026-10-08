# v3.1.0 — Learned Route Adapter / Rule-to-Policy Capability Transfer

## 1. Problem

v3.0 showed that a transparent adaptive RouteAssist could keep a frozen v2.8 actor at 80% success on release13 while using only 11.4% *mean* geometric steering blend.  However, the rule controller still produced non-zero steering corrections on about 76.5% of steps.  Mean authority alone therefore did not prove that the neural system had learned the route-following correction.

v3.1 asks a stricter question:

> Can the policy-visible pure-pursuit correction be distilled into a neural residual module and then removed completely from inference?

The base PPO checkpoint is kept frozen.  A new `LearnedRouteAdapter` sees only the same normalized state vector exposed to the policy plus the base actor action.  It predicts a raw steering residual and a learned blend.  At inference the geometric pure-pursuit controller is **not called**.

```text
policy-visible state ──→ frozen v2.8 actor ──→ base [steer,longitudinal]
        │                                      │
        └──────────────→ LearnedRouteAdapter ──┤
                                               ▼
                                  learned residual steering
                                               │
                                               ▼
                                           MetaDrive
```

An optional longitudinal `SafetySupervisor` is evaluated separately and remains a rule-based system component.

## 2. Online DAgger collection

Fresh training domain: `adapter_train = 31000–31999`.  The formal run used 15 unique seeds (`31000–31014`) across curve, ramp, bottleneck, intersection and dense-progress maps.

Three iterations execute a mixture of the current learner and the adaptive RouteAssist teacher:

| Iteration | teacher execution mix | purpose |
|---|---:|---|
| 0 | 1.0 | safe teacher-distribution bootstrap |
| 1 | 0.5 | expose learner-induced states |
| 2 | 0.2 | mostly learner-distribution DAgger |

Collection uses the longitudinal SafetySupervisor so lateral training is not dominated by avoidable rear/front collision truncations.  This supervisor does not generate the lateral target.

Formal dataset:

- 3,634 real MetaDrive transitions;
- 15 mutually distinct training seeds;
- target = adaptive RouteAssist steering generated from policy-visible state;
- no release/validation seed is present in training.

The final scenario-held diagnostic fit (still inside `adapter_train`, **not** the model-selection holdout) is:

- steering MSE: `0.0007008`;
- steering MAE: `0.01847` normalized steering units;
- final adapter SHA-256: `9c1b0e3e4f0154b86b3b320b31a452066995104898a6786643bee0e8ffba2f4b`.

## 3. Validation10 selection

`validation10 = 32000–32199`; only `32000–32014` are evaluated (3 scenarios × 5 capability stages).  Release14 had not been opened.

| Validation10 | Raw base | Learned adapter raw | Learned adapter + longitudinal supervisor | v3.0 adaptive rule reference |
|---|---:|---:|---:|---:|
| Success | 33.3% | **46.7%** | 60.0% | 93.3% |
| Collision | 13.3% | **13.3%** | **0%** | 0% |
| Off-road | 53.3% | **40.0%** | 40.0% | 6.7% |
| Route completion | 52.8% | **65.5%** | 73.2% | 92.2% |

Predeclared engineering promotion gate:

- success gain ≥ 10 percentage points;
- catastrophic (`collision + off-road`) reduction ≥ 10 points;
- no collision-rate increase;
- no off-road-rate increase.

Observed raw learned-adapter changes: success `+13.3pp`, catastrophic `-13.3pp`, collision `+0pp`, off-road `-13.3pp`.  The adapter passed before release14 was opened.

`reports/v3_1/release14_freeze.json` then froze the base checkpoint, adapter, config, selection report and core adapter/evaluator/metrics code hashes.

## 4. Fresh Release14

`release14 = 33000–33199`; the formal frozen evaluation uses `33000–33014` exactly once for this release.

| Release14 | Raw base | **Learned adapter raw** | Learned adapter + supervisor | v3.0 adaptive rule reference |
|---|---:|---:|---:|---:|
| Success | 33.3% | **60.0%** | 66.7% | 80.0% |
| Collision | 26.7% | **6.7%** | **0%** | 0% |
| Off-road | 40.0% | **33.3%** | 33.3% | **20.0%** |
| Collision OR off-road | 66.7% | **40.0%** | 33.3% | **20.0%** |
| Route completion | 58.3% | **68.4%** | 72.3% | 87.2% |
| Lane abs. error | 0.421 m | **0.352 m** | 0.338 m | **0.310 m** |

Paired raw base → learned-adapter success outcomes:

- 4 failure → success;
- 0 success → failure;
- 5 both success;
- 6 both failure;
- exact two-sided McNemar `p = 0.125`.

This is directionally encouraging but **not statistically conclusive** at this sample size.

The raw adapter uses no geometric route controller at runtime.  Its learned residual is active on 98.0% of evaluated time and has mean predicted blend 14.8%, but both quantities describe a neural module, not rule authority.

### Capability breakdown

The transfer is not uniform:

- `ramp_merge`: 33.3% → **100%** success;
- `dense_progress`: 0% → **66.7%** success;
- `bottleneck_yield`: remains 66.7%;
- `intersection_yield`: remains 66.7%;
- `curve_tracking`: remains **0% success / 100% off-road**.

Curve tracking therefore remains the most important lateral weakness.  v3.1 must not be described as “solving route following.”

## 5. What became more intelligent

Compared with v3.0, route correction is no longer computed online by pure pursuit for the v3.1 raw adapter evaluation.  The system learned a state-conditioned residual/gate mapping from DAgger data and executes that neural mapping directly.

This is a genuine rule-to-policy capability transfer, but it is still imitation learning from a hand-designed teacher.  It does not demonstrate discovery of a novel planner, and the underlying source PPO remains mostly a single-training-seed artifact.

## 6. Claim boundary

Allowed:

> Distilled policy-visible adaptive route corrections into a neural residual steering adapter using three-round online DAgger.  On a fresh 15-scenario MetaDrive holdout, the rule-free-at-runtime learned adapter improved raw success from 33.3% to 60.0% and reduced collision rate from 26.7% to 6.7%, while curve tracking remained unsolved.

Not allowed:

- “RL is 60% safe.”
- “The adapter solves curves.”
- “p=0.125 proves a statistically significant improvement.”
- “The supervised 66.7% result is fully learned”; it still uses longitudinal SafetySupervisor.
- “Release14 estimates real-road failure probability.”

Release14 is now consumed.  Further tuning must use `validation11/release15` or later fresh domains.

## 7. Engineering verification

Final source-tree regression is partitioned because Panda3D can intermittently block interpreter teardown after multiple environment constructions in one long pytest process. The two logical partitions are disjoint:

- non-MetaDrive: **283 passed, 1 skipped, 7 deselected**;
- real MetaDrive: **7 passed, 1 Gymnasium warning**;
- logical total: **290 passed, 1 skipped, 0 failed**.

Combined `src/` line coverage after appending the real-MetaDrive partition is **88%**. The wheel is built as `rl_autodrive_agent-3.1.0-py3-none-any.whl` and was installed into a fresh target directory outside the repository before importing `LearnedRouteAdapter`. CLI smoke is also executed from a temporary working directory.

`videos/v3_1_learned_adapter_seed31050.gif` is a persistent training-domain diagnostic rollout; it is deliberately not a release14 cherry-picked demonstration.

