# v3.0 Adaptive Shared-Control MetaDrive Research Release

v3.0 keeps the frozen v2.8 neural checkpoint and changes only the **transparent system handoff**. Instead of replacing steering with pure pursuit on every step, adaptive RouteAssist uses only policy-visible curvature/heading/lane-error features to choose a time-varying blend capped at 0.5.

## Validation9 selection (29000-29014)

| System | Success | Collision | Off-road | Mean route blend | Learned steering authority |
|---|---:|---:|---:|---:|---:|
| fixed v2.9 semantics | 86.7% | 0.0% | 13.3% | 100.0% | 0.0% |
| **adaptive v3.0** | **93.3%** | 0.0% | **6.7%** | **13.7%** | **86.3%** |

`release13_freeze.json` was written before any release13 stage request and all frozen hashes still match.

## Fresh release13 (30000-30014)

| System | Success | Collision | Off-road | Route completion | Mean route blend | Learned steering authority |
|---|---:|---:|---:|---:|---:|---:|
| raw v2.8 actor | 26.7% | 6.7% | 66.7% | 47.6% | 0.0% | 100.0% |
| fixed route+safety | 80.0% | 0.0% | 20.0% | 92.9% | 100.0% | 0.0% |
| **adaptive route+safety** | **80.0%** | **0.0%** | **20.0%** | 84.5% | **11.4%** | **88.6%** |

Fixed→adaptive paired success: **1 improved / 1 regressed / 11 both-success / 2 both-fail**, exact McNemar p=1.

Aggregate fixed and adaptive success are equal on release13, but adaptive reduces mean geometric steering blend from 100% to **11.4%**, giving the learned actor **88.6% mean steering authority**. This is a system-authority result, not evidence that the raw actor learned the missing lateral skill.

### Per capability

| Stage | Raw success | Fixed success | Adaptive success | Adaptive route blend |
|---|---:|---:|---:|---:|
| curve_tracking | 0.0% | 66.7% | 33.3% | 18.7% |
| ramp_merge | 0.0% | 100.0% | 100.0% | 10.1% |
| bottleneck_yield | 33.3% | 100.0% | 100.0% | 10.4% |
| intersection_yield | 66.7% | 33.3% | 66.7% | 11.3% |
| dense_progress | 33.3% | 100.0% | 100.0% | 11.7% |

## Interpretation boundary

- The neural checkpoint is unchanged from v2.8; no v3.0 learned model is promoted.
- Adaptive RouteAssist contributes steering on many steps, although its mean blend is low; this is still hybrid shared control.
- SafetySupervisor remains enabled for the two hybrid systems and can modify longitudinal control.
- release13 contains only 15 MetaDrive scenarios and is not a safety certificate or real-road risk estimate.
- validation9 and release13 are now consumed; future tuning must move to fresh domains.
