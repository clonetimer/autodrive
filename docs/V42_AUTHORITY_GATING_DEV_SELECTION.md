# v4.2 Authority Gating — Development Selection

Status: **development architecture selected; validation16 and release19 unopened**.

The v4.1 longitudinal-only branch improved interaction stress but regressed standard capability because the longitudinal residual could exert large authority in states with no visible neighbours. v4.2 therefore factorizes longitudinal authority into a structural neighbour-presence gate and an optional learned intervention-probability gate.

Development screening reused **validation15 only because that split had already been consumed by v4.1**. It is not formal evidence for v4.2. All four variants used development initialization seed 12400:

| Variant | Presence gate | Learned gate | Standard | Interaction |
|---|---:|---:|---:|---:|
| G0 | no | no | 14/15 success, 0 catastrophic | 9/12 success, 3 catastrophic |
| GP | yes | no | 14/15 success, 0 catastrophic | 9/12 success, 3 catastrophic |
| GL | no | yes | 14/15 success, 1 collision | 6/12 success, 6 catastrophic |
| **GPL** | **yes** | **yes** | **15/15 success, 0 catastrophic** | **7/12 success, 5 catastrophic** |

The development rule was deliberately lexicographic: first recover zero-regression standard capability, then retain non-trivial interaction improvement. **GPL is the only gated treatment that passed the first gate**, so it is the selected formal architecture. G0 remains the fresh-split always-authorized baseline.

Formal training is restricted to `G0 vs GPL × {13400, 14400, 15400}` on `authority_train`. Development seed 12400 is excluded. The training launcher intentionally has no validation16 or release19 execution path.

Fresh formal domains remain reserved:

- `validation16`: formal standard-capability + interaction selection.
- `release19`: final one-shot release gate only after validation16 selection is frozen.
