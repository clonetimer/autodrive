# v4.2 Formal Validation16 Protocol

This protocol is frozen **before validation16 is opened**. Formal candidates are G0 and GPL, each trained with initialization seeds 13400, 14400, and 15400. Development seed 12400 is excluded. The historical incumbent is the frozen v3.2 system: v2.8 base actor plus v3.2 learned lateral adapter, with RouteAssist and SafetySupervisor disabled at runtime.

Selection is lexicographic. First, a candidate must pass the standard capability regression gate: at least 44/45 successes across the three training seeds, at most 1/45 collision-or-off-road, no catastrophic curve episode, and canonical seed 13400 must achieve at least 14/15 successes with zero catastrophic failures. Only candidates that pass this gate are allowed to claim interaction improvement.

Second, on the interaction stress suite, the candidate must have no more catastrophic failures and no fewer successes than the incumbent reference, with at least one of those two counts strictly improved. Continuous metrics such as route completion, correction magnitude, jerk, and authority are descriptive/tie-break metrics only after the binary gates pass.

If GPL passes both gates and is not inferior to G0 under the same lexicographic ordering, the precommitted **GPL seed 13400** artifact is the only release19 candidate. There is no best-seed selection. If no candidate passes both gates, release19 remains unopened and v3.2 remains incumbent.
