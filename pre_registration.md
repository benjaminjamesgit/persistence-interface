# Pre-registration: Persistence-Interface Toy

Compact interface vs. microscopic model for predicting persistence;
coherence-as-QUANTITY (mean-field) vs. coherence-as-STRUCTURE (sparse-specific).

- Master seed: `20260627`
- Pre-registered: 2026-06-27
- Status: LOCKED. This file is committed BEFORE any model code is written.
  Sections 1-8 (design, predictions P0-P4, adjudication rule, sanity checks)
  are not edited after the experiment is run. Nulls and failures are reported
  as findings, not patched away.

This document is the contract. `run.py` implements exactly what is written
here. `results.json` / `results.md` report every scalar. `adjudication.md`
fills in SUPPORTED / NOT SUPPORTED against P0-P4 using the locked numbers, with
no revision of any prediction or threshold after seeing results.

---

## 0. Question (scope)

For a system whose persistence depends on a set of binary constraints, is there
a compact interface that predicts persistence nearly as well as the full
microscopic model, and better than an independent-constraint baseline, at far
lower description length? Two distinct senses of "compact interface" are tested
SEPARATELY and must not be conflated:

- QUANTITY (mean-field): persistence as a function of identity-blind aggregate
  counts (how many constraints active, how many co-active pairs/triples).
  This is interface `M3`.
- STRUCTURE (sparse-specific): persistence as a function of a FEW specific
  constraint-couplings, identity-aware but sparse. This is interface `M4`.

The deliverable is a MAP, not a yes/no: locate WHERE on the substrate spectrum
(identity-irrelevant -> identity-decisive) each sense works, if either, and
where the microscopic model is irreducible.

Out of scope (must not be claimed): whether a compact macro interface is "more
than statistics" / a genuine causal addition over the micro model. This tests
only whether a compact macro interface is a BETTER STATISTICAL INTERFACE for
predicting persistence at lower description length.

---

## 1. Substrate (generative ground truth)

- `N = 10` binary constraints. Configuration `x` in `{0,1}^N`. All `2^N = 1024`
  configurations are enumerated.
- Persistence log-odds (raw):

      eta_raw(x) = sum_i a_i x_i
                 + alpha * ( sum_{i<j} b_ij x_i x_j
                             + [max_order == 3] * sum_{i<j<k} c_ijk x_i x_j x_k )

- Standardization + centering (applied identically to EVERY cell and replicate,
  with a single locked temperature `TEMP = 2.5`):
  1. Compute `eta_raw(x)` over all `2^N` configs.
  2. `eta_std(x) = TEMP * (eta_raw(x) - mean(eta_raw)) / std(eta_raw)`.
     If `std(eta_raw) == 0` (degenerate all-zero coefficients), `eta_std = 0`.
  3. `eta(x) = eta_std(x) - eta_bar`, where scalar `eta_bar` is found by
     bisection so that `mean_x sigmoid(eta(x)) = 0.5` over all configs.
- `p(x) = sigmoid(eta(x))`. Label `y ~ Bernoulli(p(x))`.

  Rationale for standardization (LOCKED, declared in advance): `TEMP` fixes the
  overall signal-to-noise of the substrate to a healthy regime (avoids
  degenerate near-deterministic saturation and logistic-separation blow-ups)
  identically across all cells. Because it is a single global affine transform
  of `eta` applied before the sigmoid, it is IDENTICAL for every interface
  within a cell and therefore cannot change the within-cell ranking of
  M1/M3/M4/M2, the gain_captured ratios, or the integrity guards. It only sets
  difficulty. It is fixed once and never tuned per cell. An affine transform of
  eta preserves the sufficient-statistic structure (Section, P2): if eta_raw is
  a function of (s1,s2,s3) then so is eta.

### Coupling pattern (structure type)

- `homogeneous`: `a_i = a` const, `b_ij = b` const, `c_ijk = c` const, with
  LOCKED constants `a = b = c = 1.0`. Pure symmetry: the aggregate power sums
  (s1,s2,s3) are SUFFICIENT statistics for eta. (Positive control for QUANTITY.)
- `heterogeneous`: `a_i, b_ij, c_ijk ~ N(0,1)` iid, all entries present (dense).
- `sparse`: `a_i ~ N(0, 0.3^2)` (weak order-1); `b_ij` and `c_ijk` all zero
  EXCEPT `s2 = N = 10` randomly chosen pairs and `s3 = N//2 = 5` randomly chosen
  triples, each set to a large magnitude `~N(0, 2.0^2)`. Signal concentrated in
  a few specific couplings. (Positive control for STRUCTURE.)

  For `heterogeneous` and `sparse`, the random coefficients (and the choice of
  which pairs/triples are nonzero, for sparse) are RE-DRAWN per replicate, so
  the reported mean +/- std reflects variation over substrate instances as well
  as over config-splits and sampling. For `homogeneous` the coefficients are
  constant; only the split and sampling vary per replicate.

### Sweep grid (18 cells)

- `structure` in {homogeneous, heterogeneous, sparse}
- `max_order` in {2, 3}
- `alpha` in {0, 1, 3}

`alpha = 0` makes the substrate purely additive (order-1 only) = the NULL.
Note (consequence, declared in advance): at `alpha = 0` the `max_order` knob has
no effect on the substrate (the higher-order terms are multiplied by 0), so the
two `alpha=0` cells of a given structure are statistically identical substrates
differing only by RNG stream. They are still run separately for completeness and
serve as an internal consistency check.

---

## 2. Interfaces (all logistic regression; identical fit/eval protocol)

Feature maps over `x` (all interaction features are products of the relevant
`x_i`; all feature matrices are standardized with a `StandardScaler` fit on the
TRAIN samples only, then applied to TEST):

- `M1` ADDITIVE: features `{x_i}` (order-1, identity-aware). Description length
  = `N = 10`. The independent-constraint / energy-entropy baseline. Fit with
  weak L2 (large C, see Section 7).
- `M3` SYMMETRIC LADDER (identity-blind power sums):
  `s1 = sum_i x_i`, `s2 = sum_{i<j} x_i x_j`, `s3 = sum_{i<j<k} x_i x_j x_k`.
  Fit NESTED as three separate models:
  `M3@1 = {s1}` (DL 1), `M3@2 = {s1,s2}` (DL 2), `M3@3 = {s1,s2,s3}` (DL 3).
  Weak L2. [QUANTITY.] M3 stays strictly identity-blind: power sums only,
  never any `x_i`-specific feature.
- `M4` SPARSE-SPECIFIC: L1-regularized logistic over the FULL feature set
  `{x_i} U {x_i x_j} U {x_i x_j x_k}` (orders 1,2,3; `N + C(N,2) + C(N,3) = 175`
  features). The L1 strength is selected by cross-validation on the TRAIN
  samples only. Description length = number of NONZERO coefficients (intercept
  excluded). [STRUCTURE.] M4 must DISCOVER its couplings via L1; it is NEVER
  given the true coupling pattern.
- `M2` FULL MICRO: the full feature set (orders 1,2,3; 175 features) with light,
  fixed L2. Description length = `175`. This is the ceiling.

The interface feature sets are FIXED and do not depend on the substrate's
`max_order`. M2/M4 always include order-3 features and M3 always fits @1/@2/@3,
regardless of whether the substrate carries 3-way signal. This is what makes the
ladder-saturation test (P1) meaningful: the interface always has the capacity
for order-3; we observe whether using it actually pays.

---

## 3. Protocol

- Held-out CONFIGURATIONS. Per replicate, the `2^N = 1024` configs are randomly
  split into TRAIN-configs (70% = 717) and TEST-configs (30% = 307). Samples
  `(x, y ~ Bernoulli(p(x)))` are drawn with `x` taken only from the relevant
  pool: TRAIN samples from TRAIN-configs, TEST samples from TEST-configs. Total
  `M = 20000` samples: `M_train = 14000` (70%) from train-configs, `M_test =
  6000` (30%) from test-configs. Configs within a pool are drawn uniformly with
  replacement. Models are fit on train-config samples and evaluated on
  test-config samples. Because test configs are never seen in training,
  memorization cannot help; this tests generalization of persistence STRUCTURE.
- Metrics (out-of-sample, on test-config samples): log-loss (PRIMARY), AUC
  (secondary). Lower log-loss / higher AUC is better.
- Replicates: `R = 10` master-seed replicates. Report mean +/- std over
  replicates for every scalar.
- Description length: feature count for M1 / M3@k / M2; nonzero-coefficient
  count for M4 (mean +/- std over replicates).
- gain_captured(M):

      gain_captured(M) = (LL_M1 - LL_M) / (LL_M1 - LL_M2)

  = fraction of the micro ceiling's log-loss improvement over additive that M
  recovers (1.0 = matches ceiling; 0 = no better than additive). Point estimate
  uses the ratio of replicate-MEAN log-losses; the per-replicate distribution
  gives the spread. It MAY exceed 1.0 if a compact interface out-generalizes an
  overfit M2 -> reported, NOT clipped.

  DEGENERATE DENOMINATOR (generalizes the alpha=0 case). gain_captured is a ratio
  whose denominator `(LL_M1 - LL_M2)` is the micro ceiling's improvement over
  additive. When that improvement is not meaningfully positive the ratio is
  undefined / numerically unstable, so it is NOT computed (reported as null) and
  the cell is adjudicated by ABSOLUTE log-loss gaps instead. "Not meaningfully
  positive" is LOCKED as `(mean LL_M1 - mean LL_M2) <= eps_LL` (eps_LL = 0.002,
  Section 3). This covers two situations with the SAME rule:
  (a) `alpha = 0` (additive substrate): M1 and M2 both fit the true additive
      model, so the denominator is ~0 by construction. (This is the case the
      pre-registration originally named.)
  (b) any cell where the held-out-config micro model M2 fails to beat additive
      M1 (e.g. a low-dimensional symmetric substrate where the overparameterized
      M2 over-fits on train configs and does not out-generalize M1 on test
      configs). Here the "ceiling" is at or below additive, so there is no
      positive gap to take a fraction of.
  This is a soundness clarification of the metric's undefined regime (a ratio
  with non-positive denominator), decided on the structural condition above and
  NOT on which interface happens to win. It changes no prediction (P0-P4), no
  threshold (0.9, eps_LL, 2*SE), and no substrate constant.

### "Beyond noise" rule (LOCKED)

For a paired comparison of interface A vs B across the `R` replicates, let
`d_r = LL_A(r) - LL_B(r)` (same replicate -> same split/substrate/samples). Let
`mean_d` and `SE_d = std(d) / sqrt(R)`. B beats A "beyond noise" iff
`mean_d > 2 * SE_d` AND `mean_d > eps_LL`, with LOCKED `eps_LL = 0.002`
log-loss units. (Both a significance gate and a minimum-effect-size gate, so
trivially-tiny-but-consistent differences do not count as wins.)

---

## 4. Locked predictions / falsifiers (P0-P4)

These are fixed before running and are NOT edited afterward.

- P0 NULL (overfit catch). At `alpha = 0` (additive substrate), NO interface
  (M3@1/2/3, M4, M2) beats M1 out-of-sample beyond noise (Section 3 rule). If
  any compact interface beats M1 at `alpha = 0`, the pipeline overfits; that is
  a pipeline failure to be fixed (regularization / CV / sample size) and re-run
  BEFORE interpreting any other cell. Reported per the rule, not patched post
  hoc by changing P0.

- P1 LADDER SATURATION. M3's gain saturates at the substrate's true `max_order`:
  `M3@3` beats `M3@2` beyond noise only when `max_order = 3` carries genuine
  3-way signal (alpha > 0 and max_order = 3); otherwise `M3@2 ~ M3@3`.
  Similarly `M3@2` beats `M3@1` beyond noise only when there is genuine 2-way
  signal (alpha > 0). The SATURATION ORDER per cell (highest k in {1,2,3} that
  still beats k-1 beyond noise; 1 if even @2 does not beat @1) is a primary
  read-out: the order at which adding aggregate combinatorial structure stops
  paying.

- P2 QUANTITY (M3).
  - `homogeneous`: M3 captures ~all of M2's gain (gain_captured -> ~1).
    Specifically M3@(substrate max_order) ~ M2 because the power sums are
    sufficient statistics (positive control / bug-catch).
  - `heterogeneous`: M3 partial (captures only the mean-coupling component;
    0 < gain_captured < 0.9 typically, and well below ceiling).
  - `sparse`: M3 ~ M1 (gain_captured -> ~0): aggregate counts are blind to
    specific couplings.

- P3 STRUCTURE (M4).
  - `sparse`: M4 recovers most of M2's gain (gain_captured >= 0.9) at a SMALL
    nonzero-feature count -> coherence-as-structure wins where quantity fails.
  - `homogeneous`: M4 gives no compression advantage over M3 (M4 nonzero count
    exceeds M3's DL of <=3 while M3 already captures the gain).
  - `heterogeneous` (dense): M4 does NOT stay compact (many nonzeros) and
    approaches M2 -> no compact interface; persistence is micro-ish.

- P4 ADJUDICATION (locked rule).
  CASE A - denominator meaningfully positive (`mean LL_M1 - mean LL_M2 > eps_LL`;
  this is the typical alpha>0 het/sparse cell). The WINNER is the interface
  achieving `gain_captured >= 0.9` at MINIMUM description length, chosen among
  `{M1, M3@1, M3@2, M3@3, M4, M2}`. M2 has gain_captured = 1.0 by definition
  (DL 175) and so always qualifies; thus the winner is the smallest-DL interface
  that reaches 0.9. Tie-break on DL ties: prefer M3@k over M4 over M2 (identity-
  blind and simplest first); among M3@k prefer smaller k.

  CASE B - degenerate denominator (`mean LL_M1 - mean LL_M2 <= eps_LL`; alpha=0,
  and low-dimensional cells where M2 fails to beat M1, Section 3). There is no
  micro ceiling above additive to take a fraction of, so the cell is adjudicated
  by ABSOLUTE test log-loss:
  - At `alpha = 0`: winner is M1 (additive null) iff P0 holds (no interface
    beats M1 beyond noise); if any interface beats M1 beyond noise the cell's
    winner is "PIPELINE-FAIL" and the failure is reported.
  - At `alpha > 0`: let `best` be the interface with lowest mean test log-loss;
    let the TIED-FOR-BEST set be every interface that `best` does not beat beyond
    noise (Section 3 rule; includes `best`). The winner is the MINIMUM-DL
    interface in that set, with the same tie-break (M3@k < M4 < M1 < M2; smaller
    k first). (So a compact interface that captures real signal the micro model
    over-fit away wins; if nothing beats M1, M1 wins.)

  This CASE B absolute-gap rule is the generalization of the originally-stated
  alpha=0 fallback; it adjudicates the undefined-ratio regime on absolute
  log-loss and does not alter any prediction, threshold, or the 0.9 criterion of
  CASE A.

  "Coherence is a good persistence-interface" is SUPPORTED in a cell IFF a
  compact interface (M3@k or M4) wins; QUANTITY-sense if the winner is M3@k,
  STRUCTURE-sense if the winner is M4. The deliverable is the winners map across
  (structure x max_order x alpha) plus the ladder saturation orders. No cell's
  verdict may be revised post hoc.

---

## 5. Sanity checks (assert + report; never silently pass)

Computed and written to results; failures flagged loudly.

- Base rate ~0.5 per cell: mean train label and mean test label within
  `0.5 +/- 0.05` (tolerance on the sampled rate; the population mean is 0.5 by
  construction via eta_bar). Reported; flagged if outside.
- M2 >= others IN-sample: M2's in-sample log-loss is <= that of M1/M3@k/M4
  (within a small tolerance `1e-6`) since M2 is the richest model with only
  light reg. Reported; flagged if violated.
- Homogeneous sufficiency: on `homogeneous`, M3@(substrate max_order) ~ M2
  out-of-sample (|gain_captured - 1| small, or M3 within noise of M2). Reported;
  flagged if grossly violated (this is the sufficient-statistic bug-catch).
- P0 null holds: at `alpha = 0`, no interface beats M1 beyond noise. Flagged
  LOUDLY if violated (the pipeline must be fixed and re-run before interpreting
  other cells).

These checks report and flag; they do not alter P0-P4 or the adjudication.

---

## 6. Outputs

- `pre_registration.md` (this file) - committed FIRST.
- `results.json` - every scalar: per cell, per interface, log-loss/AUC mean+std,
  description length mean+std, gain_captured; environment + versions + seeds.
- `results.md` - summary table (rows = cells; columns = M1, M3@1/2/3, M4, M2;
  plus a winner column) and the ladder-saturation table.
- `adjudication.md` - for each of P0-P4: SUPPORTED / NOT SUPPORTED with the
  numbers; the winners map; honest interpretation including any null/failure.
  Predictions unchanged.

---

## 7. Engineering conventions (LOCKED)

- Master seed `20260627`. Per (cell, replicate) randomness is derived
  deterministically via `numpy.random.SeedSequence([20260627, cell_id,
  replicate])` -> independent `default_rng` streams. Result is independent of
  execution order (so parallel execution is identical to serial).
- ASCII-only code. `scikit-learn` LogisticRegression.
  - M1, M3@k: `solver='liblinear'`, `penalty='l2'`, `C = 1e6` (weak L2, near
    unregularized; few features, no separation risk).
  - M2: `solver='liblinear'`, `penalty='l2'`, `C = 1.0` (light, fixed L2 ceiling
    to avoid separation blow-ups). LOCKED, not tuned per cell.
  - M4: `LogisticRegressionCV`, `penalty='l1'`, `solver='liblinear'`,
    `scoring='neg_log_loss'`, `cv = 4` folds on TRAIN only, `Cs = 16`
    logspace points in `[1e-3, 1e2]` (LOCKED grid), `refit=True`. Description
    length = count of |coef| > 1e-8 after refit.
  - All feature matrices standardized with `StandardScaler` fit on TRAIN.
    `max_iter = 5000` for all fits.
- Pinned versions recorded in `results.json`: numpy 2.1.3, scikit-learn 1.5.2,
  scipy 1.14.1 (see `requirements.txt`).
- `python run.py` regenerates ALL outputs deterministically from the master
  seed in one command.

---

## 8. Do NOT (integrity constraints)

- Do not tune features, regularization, thresholds, sample size, TEMP,
  homogeneous constants, or seeds to make M3 or M4 win. The values above are
  locked here, before any results exist.
- Do not weaken or drop the `alpha = 0` null or the held-out-config split.
- Do not edit P0-P4, the thresholds, or the adjudication rule after seeing
  results. Report nulls and failures as findings.
- Do not give M3 identity-aware features. Do not give M4 the true coupling
  pattern; it must find its couplings via L1.
