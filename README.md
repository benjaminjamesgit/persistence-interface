# Persistence-Interface Toy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20974005.svg)](https://doi.org/10.5281/zenodo.20974005)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A small, **pre-registered** computational experiment. For a system whose
persistence depends on a set of binary constraints, does a *compact interface*
predict persistence nearly as well as the full microscopic model, and better
than an independent-constraint baseline, at far lower description length? Two
senses of "compact interface" are tested **separately** and never conflated:

- **QUANTITY** (mean-field, interface `M3`): persistence as a function of
  identity-blind aggregate counts (how many constraints active, how many
  co-active pairs/triples).
- **STRUCTURE** (sparse-specific, interface `M4`): persistence as a function of
  a few *specific* constraint-couplings, identity-aware but sparse.

The deliverable is a **map** -- where on the substrate spectrum
(identity-irrelevant -> identity-decisive) each sense works, if either, and
where the microscopic model is irreducible -- not a yes/no.

Out of scope (not claimed): whether a compact macro interface is "more than
statistics" / a genuine causal addition over the micro model. This tests only
whether it is a better *statistical* interface for prediction at lower
description length.

## Discipline

1. `pre_registration.md` was written and committed **before any model code**.
   It locks the design, the predictions P0-P4, the adjudication rule, and the
   integrity guards.
2. The experiment can fail: the `alpha=0` additive **null** (P0) and the
   **held-out-configuration** split are integrity guards and are not weakened.
   Nulls and failures are reported as findings.
3. No post-hoc edits to P0-P4, thresholds, or the adjudication rule after the
   run. (One pre-run soundness clarification to the gain_captured
   degenerate-denominator fallback is recorded transparently in git history; it
   changed no prediction, threshold, or constant.)
4. Deterministic: one master seed (`20260627`); per-(cell, replicate) seeds
   derived from it; `python run.py` regenerates every output.

## Run

```
pip install -r requirements.txt   # numpy==2.1.3 scikit-learn==1.5.2 scipy==1.14.1
python run.py                     # use python3 if 'python' is unavailable
```

Runtime is roughly 1.5 hours on a 20-core machine (parallel across replicates;
the L1 cross-validated `M4` fits on the dense heterogeneous cells dominate -- the
weakly-regularized high-`C` end of the locked grid is the cost). All randomness
is seeded; output is identical across runs and independent of execution order.

`python run.py --report` regenerates only `results.md` and `adjudication.md`
from an existing `results.json` (the reporting layer), without re-running the
experiment.

## Interfaces (all logistic regression, identical fit/eval protocol)

| interface | features | description length |
|---|---|---|
| `M1` additive (baseline) | `{x_i}` | N = 10 |
| `M3@k` symmetric ladder (QUANTITY) | power sums `s1,s2,s3` nested | k (<=3) |
| `M4` sparse-specific (STRUCTURE) | L1 over all 175 order-1/2/3 features | # nonzero coefs |
| `M2` full micro (ceiling) | all 175 order-1/2/3 features, light L2 | 175 |

`M3` stays strictly identity-blind; `M4` discovers its couplings via L1 and is
never given the true coupling pattern.

## Outputs

- `pre_registration.md` -- the locked contract (committed first).
- `results.json` -- every scalar (per cell, per interface): log-loss/AUC
  mean+/-std, description length, gain_captured, comparisons, environment/seeds.
- `results.md` -- summary tables (test log-loss, gain_captured + DL, AUC, ladder
  saturation, sanity) with a winner per cell.
- `adjudication.md` -- P0-P4 verdicts (SUPPORTED / NOT SUPPORTED) computed by the
  locked rules from the locked numbers, the winners map, and honest
  interpretation including nulls.

## Reading the winners map

A cell's verdict supports "coherence is a good persistence-interface" iff a
*compact* interface wins: **QUANTITY**-sense if `M3@k` wins, **STRUCTURE**-sense
if `M4` wins. `M1` winning means additive suffices (the null); `M2` winning means
persistence is micro-irreducible (no compact interface). `gain_captured` is the
fraction of the micro ceiling's improvement over additive that an interface
recovers; it may exceed 1.0 when a compact interface out-generalizes the overfit
micro model on held-out configurations (reported, not clipped).

## Citation

Archived on Zenodo. Cite the concept DOI (resolves to the latest version):
**[10.5281/zenodo.20974005](https://doi.org/10.5281/zenodo.20974005)**. The v0.1.0
version DOI is [10.5281/zenodo.20974006](https://doi.org/10.5281/zenodo.20974006).
Machine-readable metadata is in [`CITATION.cff`](CITATION.cff); GitHub's "Cite
this repository" button renders it.

## License

[Apache License 2.0](LICENSE).

