# persistence-interface

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20974005.svg)](https://doi.org/10.5281/zenodo.20974005)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**A pre-registered test of whether a compact macro interface can predict persistence better than its parts — and a record of how that question closed, deflationarily, into known statistics.**

persistence-interface asks whether "coherence," construed as a compact interface over a system's constraints, earns its keep at predicting persistence: better than an independent-constraint baseline, and near the full microscopic model, at far lower description length. It tests two senses separately — coherence as **quantity** (identity-blind aggregate counts; `M3`) and coherence as **structure** (a few specific couplings; `M4`) — against a micro ceiling (`M2`) and an additive floor (`M1`), across substrates spanning identity-irrelevant to identity-decisive, with predictions locked before the run.

> Where a compact interface wins, is it discovering something — or re-instantiating statistics that were already named?

**Result (terminal): the latter, and for the reason that matters, necessarily.** A compact interface wins only in the two regimes standard statistics already owns. When the substrate is exchangeable, the aggregate that wins is the model's **sufficient statistic** (Pitman-Koopman-Darmois; de Finetti; the Curie-Weiss mean field) — a positive control passing, not a finding. When the signal sits in a few couplings, the interface that wins is **L1 sparse recovery** (Lasso), and it over-selects 2.5-4x the true support rather than recovering it. Where identity carries the signal, no compact interface exists at all. "Coherence as a better interface" adds nothing beyond statistics already in the textbooks; the quantity/structure distinction survives only as a diagnostic lens.

## What was tested
A substrate of `N=10` binary constraints with persistence `y ~ Bernoulli(sigmoid(eta(x)))`, `eta` carrying interactions to order 3, swept over structure {homogeneous, heterogeneous, sparse} x max_order {2,3} x interaction-strength alpha {0,1,3}, 10 replicates, deterministic from master seed `20260627`. Four interfaces, identical fit/eval on **held-out configurations**, scored by out-of-sample log-loss: `M1` additive (`{x_i}`), `M3` the identity-blind symmetric power-sum ladder (orders 1->2->3 = coherence-as-quantity), `M4` an L1-sparse model over the full feature set (coherence-as-structure), `M2` the full micro model (ceiling).

## The arc — tested -> found
| regime | locked prediction | what was found |
|---|---|---|
| alpha=0 (all structures) | additive `M1` wins (null) | **held** — no interface beats `M1` |
| homogeneous, alpha>0 | `M3` (quantity) wins | **won — but it is the sufficient-statistics positive control.** The flagship `mo2/a1` cell does not beat `M1` beyond noise (a degenerate-denominator tiebreak, 0.12x the eps-gate); the micro ceiling is *worse* than additive on the two `max_order=2` cells. The winning `M3` beats `M1` beyond noise in three of four cells, but by a clear margin (~2.2-2.6%) only in the two `mo3` cells; `mo2/a3` clears the gate marginally (0.65%) and `mo2/a1` not at all. |
| sparse, alpha>0 | `M4` (structure) wins | **won on prediction** — but as Lasso: ~60/175 features against a true support of ~15-25, an over-selection of 2.5-4x. Predictively adequate, not a clean recovery. |
| heterogeneous (dense), alpha>0 | `M3` partial | **NOT SUPPORTED** — `M3` is identity-blind even at order 1, so it collapses to chance (~ln 2) the moment identities matter. No compact interface; micro-irreducible. |

## The result — the terminal close
Coherence-as-quantity (`M3`) is a sufficient-statistics unit test: it wins exactly where the substrate is exchangeable and collapses to chance everywhere else, by construction. Coherence-as-structure (`M4`) is Lasso: it predicts well under sparsity while over-selecting the support. Neither exceeds standard statistics. The wall is the program's, restated a third time: a claimed surplus that, once operationalized, reduces to something already named.

## What holds, what closed
**Holds.** The pre-registration discipline (locked-first, honest failures, adversarially validated); the reproducible winners map; and the **lens** — the legitimate diagnostic question, *where on the identity-irrelevant -> identity-decisive spectrum does an aggregate suffice?*
**Closed.** Coherence as a *distinctive* better-interface for persistence beyond the statistics it is built from. It is not one.

## Why the close is trustworthy
The pre-registration was committed before any model code; the single pre-run edit changed no prediction, threshold, or constant; results were committed last; the post-run commit added a caveat *against* the experiment (the literal `P2` prediction failed, and is reported as failed). An independent four-agent adversarial validation reproduced every number (max error 0.0), confirmed the by-configuration split (no leakage) and the interior L1 path, recomputed the winners with zero mismatch, and confirmed `gain>1` is the micro model overfitting held-out configs — never a surplus "beyond information."

## Scope
Tested as **case 2** — whether a compact macro interface is a *better statistical interface* at lower description length. It makes **no claim** that any interface is "more than statistics" or a causal addition over the micro model. Theory-led where it matters: the homogeneous result is a sufficient-statistics identity by construction; the heterogeneous/sparse `M3` failure is largely definitional (identity-blindness at order 1). `N=10`, 10 replicates; the margins are thin where the regime is degenerate.

## Place in the program
This is the third pass of the same blade. **Coherence-Information** closed the claim that coherence is a *measurable* beyond statistical information; **steer-push** closed the claim that a *sign* can stand below life without a beneficiary; **persistence-interface** closes the claim that coherence is a *better interface* than the statistics it is built from. One wall, three faces: in each, a claimed surplus evaporates under operationalization, and what survives is the diagnostic lens, not the quantity. The discipline that closed each — pre-register the death-cell, then occupy it when the evidence demands — is the contribution the program actually has.

## Repo map
- `pre_registration.md` — the locked design (substrate, interfaces, P0-P4, adjudication rule); append-only.
- `run.py` — apparatus + deterministic runner. `results.json` — all scalars. `results.md` — the winners table. `adjudication.md` — verdicts + honest interpretation + terminus.
- `requirements.txt` — pinned environment.

## Reproduction
`python run.py` regenerates everything deterministically from seed `20260627` (~90 min; the dense-cell L1 cross-validation dominates). `python run.py --report` regenerates the markdown tables/verdicts from `results.json` in seconds (the hand-authored "Honest interpretation" and "Terminus" sections of `adjudication.md` are the closing narrative and are not regenerated).

## Citation
Archived on Zenodo. Cite the concept DOI (resolves to the latest version): **[10.5281/zenodo.20974005](https://doi.org/10.5281/zenodo.20974005)**. Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

## License
Apache-2.0 (match the sibling repos). See [`LICENSE`](LICENSE).
