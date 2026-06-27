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

- homogeneous (power sums sufficient: M2 does not beat M3@max beyond noise; identity-blind counts suffice): **SUPPORTED**
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

- sparse (M4 recovers gain at small nonzero count, compact wins): **SUPPORTED**
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
