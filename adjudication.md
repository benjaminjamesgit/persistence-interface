# Adjudication: P0-P4

Verdicts computed by the LOCKED rules in pre_registration.md from the locked results. Predictions are unchanged. Nulls and failures are reported as findings.

## P0 NULL (overfit catch)

**SUPPORTED.** At alpha=0 (purely additive substrate), no interface (M3@*, M4, M2) beats M1 out-of-sample beyond noise.

No interface beats M1 at any alpha=0 cell beyond noise. The held-out-config split plus regularization prevent the pipeline from manufacturing structure where the substrate is additive.

## P1 LADDER SATURATION

**SUPPORTED.** M3@k beats M3@(k-1) beyond noise only where the substrate carries genuine aggregate k-way signal (necessary conditions: 2>1 requires alpha>0; 3>2 requires alpha>0 and max_order=3).

Saturation order per cell (primary read-out):

| cell | saturation order |
|---|---|
| homogeneous/mo2/a0 | 1 |
| homogeneous/mo2/a1 | 1 |
| homogeneous/mo2/a3 | 2 |
| homogeneous/mo3/a0 | 1 |
| homogeneous/mo3/a1 | 2 |
| homogeneous/mo3/a3 | 2 |
| heterogeneous/mo2/a0 | 1 |
| heterogeneous/mo2/a1 | 1 |
| heterogeneous/mo2/a3 | 1 |
| heterogeneous/mo3/a0 | 1 |
| heterogeneous/mo3/a1 | 1 |
| heterogeneous/mo3/a3 | 1 |
| sparse/mo2/a0 | 1 |
| sparse/mo2/a1 | 1 |
| sparse/mo2/a3 | 1 |
| sparse/mo3/a0 | 1 |
| sparse/mo3/a1 | 1 |
| sparse/mo3/a3 | 1 |

## P2 QUANTITY (M3 = identity-blind power sums)

- homogeneous (power sums sufficient: M2 does not beat M3@max beyond noise; identity-blind counts suffice -- this is the sufficient-statistics positive control, the micro ceiling fails to beat additive beyond noise on 3 of the 4 cells here (it is worse than additive on the two max_order=2 cells; only mo3/a3 is a CASE-A win), and mo2/a1 is a degenerate-denominator tiebreak; see Honest interpretation #1-2): **SUPPORTED**
  - mo2/a1: LL[M1=0.4108 M3@2=0.4086 M2=0.4193]; sufficient (M3@max >= M2) = True; M3@max beats M1 = True; winner = M3@1
  - mo2/a3: LL[M1=0.4172 M3@2=0.4145 M2=0.4237]; sufficient (M3@max >= M2) = True; M3@max beats M1 = True; winner = M3@2
  - mo3/a1: LL[M1=0.4566 M3@3=0.4465 M2=0.4547]; sufficient (M3@max >= M2) = True; M3@max beats M1 = True; winner = M3@2
  - mo3/a3: LL[M1=0.4645 M3@3=0.4525 M2=0.4614]; sufficient (M3@max >= M2) = True; M3@max beats M1 = True; winner = M3@2
- heterogeneous (predicted 'partial', 0 < gain < 0.9; all cells here are CASE A / non-degenerate, none excluded): **NOT SUPPORTED**
  - mo2/a1: gain(M3@3) = -1.963, M3 beats M1 beyond noise = False [M3 WORSE than M1]
  - mo2/a3: gain(M3@3) = -1.076, M3 beats M1 beyond noise = False [M3 WORSE than M1]
  - mo3/a1: gain(M3@3) = -1.023, M3 beats M1 beyond noise = False [M3 WORSE than M1]
  - mo3/a3: gain(M3@3) = -0.527, M3 beats M1 beyond noise = False [M3 WORSE than M1]
- sparse (aggregate counts blind to specific couplings; directional rollup = M3 does not beat M1. CAVEAT: the literal 'gain -> ~0 / M3 ~ M1' was NOT met -- M3 collapses BELOW additive, gain ~ -1, M1 beats M3 beyond noise, same identity-blindness mechanism as heterogeneous): **SUPPORTED**
  - mo2/a1: gain(M3@3) = -1.140, M3 beats M1 beyond noise = False [M3 WORSE than M1]
  - mo2/a3: gain(M3@3) = -1.041, M3 beats M1 beyond noise = False [M3 WORSE than M1]
  - mo3/a1: gain(M3@3) = -1.036, M3 beats M1 beyond noise = False [M3 WORSE than M1]
  - mo3/a3: gain(M3@3) = -1.070, M3 beats M1 beyond noise = False [M3 WORSE than M1]

## P3 STRUCTURE (M4 = L1 sparse-specific)

- sparse (M4 matches the ceiling at 33-37% of micro features -- compact vs micro as a description-length fact, but Lasso over-selection ~2.5-4x the true support, NOT a clean recovery; see Honest interpretation #3): **SUPPORTED**
  - mo2/a1: gain(M4) = 1.059, nonzeros = 61.7, winner = M4
  - mo2/a3: gain(M4) = 1.073, nonzeros = 58.5, winner = M4
  - mo3/a1: gain(M4) = 1.049, nonzeros = 63.7, winner = M4
  - mo3/a3: gain(M4) = 1.063, nonzeros = 64.1, winner = M4
- homogeneous (M4 no compression advantage over M3): **SUPPORTED**
  - mo2/a1: nonzeros = 101.0 (vs M3 DL <= 3), winner = M3@1
  - mo2/a3: nonzeros = 100.0 (vs M3 DL <= 3), winner = M3@2
  - mo3/a1: nonzeros = 113.7 (vs M3 DL <= 3), winner = M3@2
  - mo3/a3: nonzeros = 116.2 (vs M3 DL <= 3), winner = M3@2
- heterogeneous (dense: M4 approaches M2 AND not compact; 'not compact' = nonzero fraction >= 0.50, a REPORTING HEURISTIC, not a locked threshold): **SUPPORTED**
  - mo2/a1: nonzeros = 107.6 (61% of micro), M4 approaches M2 = True, gain(M4) = 1.045, winner = M4
  - mo2/a3: nonzeros = 113.2 (65% of micro), M4 approaches M2 = True, gain(M4) = 1.031, winner = M4
  - mo3/a1: nonzeros = 150.8 (86% of micro), M4 approaches M2 = True, gain(M4) = 1.015, winner = M4
  - mo3/a3: nonzeros = 153.5 (88% of micro), M4 approaches M2 = True, gain(M4) = 1.008, winner = M4

## P4 ADJUDICATION (winners map)

Winner = lowest-DL interface reaching gain_captured >= 0.90 (M1 at alpha=0 iff null holds). 'Coherence is a good persistence interface' is SUPPORTED in a cell iff a compact interface wins: QUANTITY-sense if M3@k, STRUCTURE-sense if M4.

(For M4 wins the nonzero fraction is shown: M4 is only meaningfully 'compact' when it uses well under the 175 micro features; a near-175 nonzero count is a marginal/micro-ish win, cross-read with P3.)

| cell | winner | sense |
|---|---|---|
| homogeneous/mo2/a0 | M1 | additive null |
| homogeneous/mo2/a1 | M3@1 | QUANTITY (compact) |
| homogeneous/mo2/a3 | M3@2 | QUANTITY (compact) |
| homogeneous/mo3/a0 | M1 | additive null |
| homogeneous/mo3/a1 | M3@2 | QUANTITY (compact) |
| homogeneous/mo3/a3 | M3@2 | QUANTITY (compact) |
| heterogeneous/mo2/a0 | M1 | additive null |
| heterogeneous/mo2/a1 | M4 | STRUCTURE but weak compression ~micro (61% of micro features) |
| heterogeneous/mo2/a3 | M4 | STRUCTURE but weak compression ~micro (65% of micro features) |
| heterogeneous/mo3/a0 | M1 | additive null |
| heterogeneous/mo3/a1 | M4 | STRUCTURE but weak compression ~micro (86% of micro features) |
| heterogeneous/mo3/a3 | M4 | STRUCTURE but weak compression ~micro (88% of micro features) |
| sparse/mo2/a0 | M1 | additive null |
| sparse/mo2/a1 | M4 | STRUCTURE (compact, 35% of micro features) |
| sparse/mo2/a3 | M4 | STRUCTURE (compact, 33% of micro features) |
| sparse/mo3/a0 | M1 | additive null |
| sparse/mo3/a1 | M4 | STRUCTURE (compact, 36% of micro features) |
| sparse/mo3/a3 | M4 | STRUCTURE (compact, 37% of micro features) |

## Sanity checks

All passed: base rate ~0.5; M2 in-sample dominant; homogeneous sufficiency (M3@max ~ M2); P0 null holds.

## Honest interpretation (post-run close)

The locked winners and verdicts above stand. This section states what they do and do not show, with every number recomputed from `results.json`; it deflates four interpretive overstatements. (Hand-authored narrative: `run.py --report` regenerates the P0-P4 tables above but not this section or the Terminus.)

1. **The micro ceiling is not uniformly "below the floor."** M2 fails to beat additive M1 beyond noise on 3 of the 4 homogeneous alpha>0 cells, but it is literally WORSE than M1 (M1 beats M2 beyond noise) only on the two `max_order=2` cells: M2-M1 = +0.0085 (mo2/a1) and +0.0064 (mo2/a3). On `mo3/a1` the ceiling is within noise (denom M1-M2 = +0.0020, a CASE-B tie); only `mo3/a3` is a CASE-A positive-denominator cell (M2 beats M1 beyond noise, denom +0.0030). The cause is overfitting -- M2 fits 175 features and does not out-generalize the low-dimensional truth on held-out configurations.

2. **The flagship `homogeneous/mo2/a1` -> M3@1 "QUANTITY win" does not beat M1 beyond noise.** Its margin is M1 - M3@1 = 0.00025 nats = 0.12x the pre-registered eps-gate (0.002). It wins only as the lowest-description-length interface not beaten by the (degenerate) ceiling -- a CASE-B tiebreak, not a demonstrated improvement over additive. Across the four homogeneous alpha>0 cells the *winning* M3 interface beats M1 beyond noise in three (mo2/a3, mo3/a1, mo3/a3), but with a clear margin only in the two `mo3` cells (M1 - M3@2 = 2.21% and 2.56% of log-loss); `mo2/a3` passes the eps-gate only marginally (0.65%), and `mo2/a1` not at all.

3. **M4 on sparse is predictively adequate, not a clean structure recovery.** It selects 58.5-64.1 of 175 features (33-37%) -- compact relative to micro as a description-length fact -- against a true support of 15 specific couplings (SPARSE_S2=10 pairs + SPARSE_S3=5 triples), up to ~25 counting the weak order-1 terms. That is an over-selection of ~4x the 15 couplings (~2.5x against 25): the standard Lasso behaviour when the irrepresentable condition fails. M4 predictively matches the ceiling (gain ~ 1.05-1.07) while over-selecting; it does not recover the couplings.

4. **Textbook framing.** The homogeneous QUANTITY win is the sufficient-statistics positive control for an exchangeable / mean-field model (Pitman-Koopman-Darmois; de Finetti; Curie-Weiss): where the substrate is exchangeable the winning aggregate power-sum IS the model's sufficient statistic -- a passing unit test, not a finding. The sparse STRUCTURE win is L1 sparse recovery (Lasso) under sparsity. The quantity/structure distinction therefore survives as a diagnostic lens, not as a discovery. The heterogeneous/sparse "M3 fails" outcome is largely by construction: M3 is identity-blind even at order 1 (it sees only the count s1), so it collapses toward chance (log-loss ~ ln 2 = 0.69; observed M3 ~ 0.65-0.69) the moment order-1 weights are identity-specific.

## Terminus -- precise statement

A compact interface beats the additive baseline and matches the micro ceiling only in the two regimes standard statistics already owns: exchangeable substrates, where the winning aggregate IS the model's sufficient statistic (a positive control), and sparse substrates, where the winner is L1 recovery (over-selecting 2.5-4x the true support). Where signal is identity-specific, no compact interface exists. "Coherence as a better interface" therefore adds nothing beyond statistics already named; what the experiment establishes is negative and locating -- it disciplines coherence-as-quantity out of the disordered regimes -- and the quantity/structure framing is retained only as a diagnostic lens. Tested as case 2 (a better statistical interface); no case-3 claim ("more than statistics" / causal addition over micro) is made or supported.
