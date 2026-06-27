"""Persistence-Interface Toy: one-command runner.

Implements EXACTLY the design locked in pre_registration.md. Running

    python run.py

regenerates results.json, results.md, and adjudication.md deterministically
from the master seed. No tuning to make any interface win; the alpha=0 null and
the held-out-config split are integrity guards and are not weakened. ASCII only.
"""

import json
import platform
import sys
from itertools import combinations

import numpy as np
import scipy
import sklearn
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

import warnings

# ---------------------------------------------------------------------------
# PARAMS (single locked block; see pre_registration.md sections 1, 3, 7)
# ---------------------------------------------------------------------------
P = dict(
    MASTER_SEED=20260627,
    N=10,                       # binary constraints; 2^N configs enumerated
    TEMP=2.5,                   # locked eta standardization temperature
    HOMOG_A=1.0,                # locked homogeneous constants a=b=c
    HOMOG_B=1.0,
    HOMOG_C=1.0,
    SPARSE_A_SD=0.3,            # sparse order-1 weak
    SPARSE_COUP_SD=2.0,         # sparse pair/triple large magnitude
    SPARSE_S2=10,               # = N pairs nonzero
    SPARSE_S3=5,                # = N//2 triples nonzero
    HET_SD=1.0,                 # heterogeneous N(0,1)
    M_TRAIN=14000,
    M_TEST=6000,
    TRAIN_CONFIG_FRAC=0.70,
    R=10,                       # replicates
    C_WEAK=1e6,                 # M1, M3@k near-unregularized L2
    C_M2=1.0,                   # M2 light fixed L2 ceiling
    M4_CS=16,                   # grid points for M4 L1 CV
    M4_CS_LO=1e-3,
    M4_CS_HI=1e2,
    M4_CV=4,
    MAX_ITER=5000,
    EPS_LL=0.002,               # "beyond noise" minimum effect size (log-loss)
    GAIN_WIN=0.90,              # adjudication: gain_captured threshold
    NONZERO_TOL=1e-8,           # M4 nonzero-coefficient threshold
    N_JOBS=16,
)

STRUCTURES = ["homogeneous", "heterogeneous", "sparse"]
MAX_ORDERS = [2, 3]
ALPHAS = [0, 1, 3]

# Interface keys and their fixed description lengths (M4 is variable).
IFACES = ["M1", "M3@1", "M3@2", "M3@3", "M4", "M2"]


# ---------------------------------------------------------------------------
# Feature precomputation (depends only on N -> module-level constants)
# ---------------------------------------------------------------------------
def _build_features(N):
    n_cfg = 1 << N
    # configs as (n_cfg, N) in {0,1}; bit i of the integer index
    bits = ((np.arange(n_cfg)[:, None] >> np.arange(N)[None, :]) & 1)
    X1 = bits.astype(np.float64)                       # (n_cfg, N)
    pairs = list(combinations(range(N), 2))            # C(N,2)
    triples = list(combinations(range(N), 3))          # C(N,3)
    X2 = np.column_stack([X1[:, i] * X1[:, j] for (i, j) in pairs])
    X3 = np.column_stack([X1[:, i] * X1[:, j] * X1[:, k] for (i, j, k) in triples])
    XFULL = np.hstack([X1, X2, X3])                    # (n_cfg, 175 for N=10)
    POW = np.column_stack([X1.sum(1), X2.sum(1), X3.sum(1)])  # s1,s2,s3
    return dict(n_cfg=n_cfg, X1=X1, X2=X2, X3=X3, XFULL=XFULL, POW=POW,
                pairs=pairs, triples=triples)


FEAT = _build_features(P["N"])
DL_FIXED = {"M1": P["N"], "M3@1": 1, "M3@2": 2, "M3@3": 3,
            "M2": FEAT["XFULL"].shape[1]}


# ---------------------------------------------------------------------------
# Substrate
# ---------------------------------------------------------------------------
def draw_coefficients(structure, rng):
    """Return (a, b, c) coefficient vectors aligned to X1/X2/X3 columns."""
    N = P["N"]
    n_pair = len(FEAT["pairs"])
    n_trip = len(FEAT["triples"])
    if structure == "homogeneous":
        a = np.full(N, P["HOMOG_A"])
        b = np.full(n_pair, P["HOMOG_B"])
        c = np.full(n_trip, P["HOMOG_C"])
    elif structure == "heterogeneous":
        a = rng.normal(0.0, P["HET_SD"], N)
        b = rng.normal(0.0, P["HET_SD"], n_pair)
        c = rng.normal(0.0, P["HET_SD"], n_trip)
    elif structure == "sparse":
        a = rng.normal(0.0, P["SPARSE_A_SD"], N)
        b = np.zeros(n_pair)
        c = np.zeros(n_trip)
        pair_pos = rng.choice(n_pair, size=P["SPARSE_S2"], replace=False)
        trip_pos = rng.choice(n_trip, size=P["SPARSE_S3"], replace=False)
        b[pair_pos] = rng.normal(0.0, P["SPARSE_COUP_SD"], P["SPARSE_S2"])
        c[trip_pos] = rng.normal(0.0, P["SPARSE_COUP_SD"], P["SPARSE_S3"])
    else:
        raise ValueError(structure)
    return a, b, c


def build_p(structure, max_order, alpha, rng):
    """Generate p(x) over all configs per the locked substrate spec."""
    a, b, c = draw_coefficients(structure, rng)
    eta_raw = FEAT["X1"] @ a
    if alpha != 0:
        eta_raw = eta_raw + alpha * (FEAT["X2"] @ b)
        if max_order == 3:
            eta_raw = eta_raw + alpha * (FEAT["X3"] @ c)
    sd = eta_raw.std()
    if sd == 0:
        eta_std = np.zeros_like(eta_raw)
    else:
        eta_std = P["TEMP"] * (eta_raw - eta_raw.mean()) / sd
    # find eta_bar so mean sigmoid(eta_std - eta_bar) = 0.5 (bisection)
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if expit(eta_std - mid).mean() > 0.5:
            lo = mid          # mean too high -> increase shift
        else:
            hi = mid
    eta_bar = 0.5 * (lo + hi)
    p_all = expit(eta_std - eta_bar)
    return p_all


def sample(p_all, rng):
    """Held-out-config split, then draw train/test samples (x, y)."""
    n_cfg = FEAT["n_cfg"]
    n_tr_cfg = int(round(P["TRAIN_CONFIG_FRAC"] * n_cfg))
    perm = rng.permutation(n_cfg)
    train_cfg = perm[:n_tr_cfg]
    test_cfg = perm[n_tr_cfg:]
    train_idx = rng.choice(train_cfg, size=P["M_TRAIN"], replace=True)
    test_idx = rng.choice(test_cfg, size=P["M_TEST"], replace=True)
    y_train = (rng.random(P["M_TRAIN"]) < p_all[train_idx]).astype(int)
    y_test = (rng.random(P["M_TEST"]) < p_all[test_idx]).astype(int)
    return train_idx, test_idx, y_train, y_test


# ---------------------------------------------------------------------------
# Interfaces (logistic regression; identical fit/eval protocol)
# ---------------------------------------------------------------------------
def _fit_eval(Xtr, ytr, Xte, yte, model):
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(Xtr_s, ytr)
    proba_te = model.predict_proba(Xte_s)[:, 1]
    proba_tr = model.predict_proba(Xtr_s)[:, 1]
    ll = float(log_loss(yte, proba_te, labels=[0, 1]))
    ll_in = float(log_loss(ytr, proba_tr, labels=[0, 1]))
    try:
        auc = float(roc_auc_score(yte, proba_te))
    except ValueError:
        auc = float("nan")
    return ll, auc, ll_in, model


def _l2(C):
    return LogisticRegression(penalty="l2", C=C, solver="liblinear",
                              max_iter=P["MAX_ITER"], random_state=0)


def run_cell_replicate(cell_id, structure, max_order, alpha, rep):
    ss = np.random.SeedSequence([P["MASTER_SEED"], cell_id, rep])
    rng = np.random.default_rng(ss)

    p_all = build_p(structure, max_order, alpha, rng)
    train_idx, test_idx, y_train, y_test = sample(p_all, rng)

    rec = dict(cell_id=cell_id, structure=structure, max_order=max_order,
               alpha=alpha, rep=rep,
               base_rate_train=float(y_train.mean()),
               base_rate_test=float(y_test.mean()),
               mean_p_all=float(p_all.mean()))

    # feature matrices for the sampled rows
    X1_tr, X1_te = FEAT["X1"][train_idx], FEAT["X1"][test_idx]
    POW_tr, POW_te = FEAT["POW"][train_idx], FEAT["POW"][test_idx]
    XF_tr, XF_te = FEAT["XFULL"][train_idx], FEAT["XFULL"][test_idx]

    # M1 additive
    ll, auc, ll_in, _ = _fit_eval(X1_tr, y_train, X1_te, y_test, _l2(P["C_WEAK"]))
    rec.update({"M1_ll": ll, "M1_auc": auc, "M1_ll_in": ll_in, "M1_dl": DL_FIXED["M1"]})

    # M3 symmetric ladder, nested @1/@2/@3
    for k in (1, 2, 3):
        ll, auc, ll_in, _ = _fit_eval(POW_tr[:, :k], y_train, POW_te[:, :k],
                                      y_test, _l2(P["C_WEAK"]))
        key = "M3@%d" % k
        rec.update({key + "_ll": ll, key + "_auc": auc, key + "_ll_in": ll_in,
                    key + "_dl": DL_FIXED[key]})

    # M2 full micro (light L2 ceiling)
    ll, auc, ll_in, _ = _fit_eval(XF_tr, y_train, XF_te, y_test, _l2(P["C_M2"]))
    rec.update({"M2_ll": ll, "M2_auc": auc, "M2_ll_in": ll_in, "M2_dl": DL_FIXED["M2"]})

    # M4 sparse-specific (L1, CV strength on train only)
    Cs = np.logspace(np.log10(P["M4_CS_LO"]), np.log10(P["M4_CS_HI"]), P["M4_CS"])
    m4 = LogisticRegressionCV(Cs=Cs, cv=P["M4_CV"], penalty="l1",
                              solver="liblinear", scoring="neg_log_loss",
                              max_iter=P["MAX_ITER"], refit=True, random_state=0)
    ll, auc, ll_in, fitted = _fit_eval(XF_tr, y_train, XF_te, y_test, m4)
    nnz = int(np.sum(np.abs(fitted.coef_) > P["NONZERO_TOL"]))
    rec.update({"M4_ll": ll, "M4_auc": auc, "M4_ll_in": ll_in, "M4_dl": nnz,
                "M4_C": float(fitted.C_[0])})
    return rec


# ---------------------------------------------------------------------------
# Aggregation, statistics, adjudication
# ---------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return None


def col(records, key):
    return np.array([r[key] for r in records], dtype=float)


def mean_std(a):
    a = np.asarray(a, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=1))


def beats(worse_ll, better_ll):
    """Does `better` beat `worse` beyond noise? (lower log-loss is better)."""
    d = np.asarray(worse_ll) - np.asarray(better_ll)   # >0 means better wins
    R = len(d)
    mean_d = float(np.mean(d))
    se = float(np.std(d, ddof=1) / np.sqrt(R))
    flag = (mean_d > 2.0 * se) and (mean_d > P["EPS_LL"])
    return dict(beats=bool(flag), mean_diff=mean_d, se=se)


def gain_point(ll1_mean, llM_mean, ll2_mean):
    denom = ll1_mean - ll2_mean
    if abs(denom) < 1e-12:
        return None
    return (ll1_mean - llM_mean) / denom


def aggregate(records):
    """Group per cell, compute all reported scalars."""
    cells = {}
    for r in records:
        cells.setdefault(r["cell_id"], []).append(r)
    out = []
    for cell_id in sorted(cells):
        recs = sorted(cells[cell_id], key=lambda r: r["rep"])
        base = recs[0]
        cell = dict(cell_id=cell_id, structure=base["structure"],
                    max_order=base["max_order"], alpha=base["alpha"],
                    base_rate_train=mean_std(col(recs, "base_rate_train")),
                    base_rate_test=mean_std(col(recs, "base_rate_test")),
                    mean_p_all=mean_std(col(recs, "mean_p_all")),
                    interfaces={})
        ll1 = col(recs, "M1_ll")
        ll2 = col(recs, "M2_ll")
        ll1_mean = float(np.mean(ll1))
        ll2_mean = float(np.mean(ll2))
        denom = ll1_mean - ll2_mean
        denom_ok = bool(denom > P["EPS_LL"])  # micro ceiling above additive?
        cell["denom"] = denom
        cell["denom_meaningful"] = denom_ok
        cell["ll_reps"] = {k: col(recs, k + "_ll").tolist() for k in IFACES}
        for k in IFACES:
            llk = col(recs, k + "_ll")
            cell["interfaces"][k] = dict(
                ll=mean_std(llk),
                auc=mean_std(col(recs, k + "_auc")),
                ll_in=mean_std(col(recs, k + "_ll_in")),
                dl=mean_std(col(recs, k + "_dl")),
                gain=(gain_point(ll1_mean, float(np.mean(llk)), ll2_mean)
                      if denom_ok else None),
            )
        # paired comparisons used by adjudication
        cell["cmp"] = dict(
            M3_2_vs_1=beats(col(recs, "M3@1_ll"), col(recs, "M3@2_ll")),
            M3_3_vs_2=beats(col(recs, "M3@2_ll"), col(recs, "M3@3_ll")),
            beats_M1={k: beats(ll1, col(recs, k + "_ll"))
                      for k in ["M3@1", "M3@2", "M3@3", "M4", "M2"]},
            M2_insample_dominates=bool(np.all(
                col(recs, "M2_ll_in") <= np.minimum.reduce(
                    [col(recs, k + "_ll_in") for k in
                     ["M1", "M3@1", "M3@2", "M3@3", "M4"]]) + 1e-6)),
        )
        out.append(cell)
    return out


def saturation_order(cell):
    if cell["cmp"]["M3_3_vs_2"]["beats"]:
        return 3
    if cell["cmp"]["M3_2_vs_1"]["beats"]:
        return 2
    return 1


def beats_keys(cell, worse_key, better_key):
    """Does `better_key` beat `worse_key` beyond noise (lower log-loss)?"""
    return beats(cell["ll_reps"][worse_key], cell["ll_reps"][better_key])["beats"]


# tie-break preference (lower = preferred): compact + identity-blind + simple
PREF = {"M3@1": 0, "M3@2": 1, "M3@3": 2, "M4": 3, "M1": 4, "M2": 5}


def winner(cell):
    """Locked P4 adjudication (CASE A gain>=0.9; CASE B absolute-gap)."""
    if not cell["denom_meaningful"]:
        # CASE B: no micro ceiling above additive -> absolute log-loss
        if cell["alpha"] == 0:
            any_beat = any(v["beats"] for v in cell["cmp"]["beats_M1"].values())
            return "PIPELINE-FAIL" if any_beat else "M1"
        keys = ["M1", "M3@1", "M3@2", "M3@3", "M4", "M2"]
        means = {k: float(np.mean(cell["ll_reps"][k])) for k in keys}
        best = min(keys, key=lambda k: means[k])
        tied = [k for k in keys if not beats_keys(cell, k, best)]
        tied.sort(key=lambda k: (cell["interfaces"][k]["dl"][0], PREF[k]))
        return tied[0]
    # CASE A: gain_captured >= 0.9 at minimum description length
    quals = []
    for k in ["M3@1", "M3@2", "M3@3", "M4", "M2"]:
        g = cell["interfaces"][k]["gain"]
        dl = cell["interfaces"][k]["dl"][0]
        if g is not None and g >= P["GAIN_WIN"]:
            quals.append((dl, PREF[k], k))
    quals.sort()
    return quals[0][2] if quals else "M2"


def build_adjudication(agg):
    by = {(c["structure"], c["max_order"], c["alpha"]): c for c in agg}
    alpha0 = [c for c in agg if c["alpha"] == 0]
    pos = [c for c in agg if c["alpha"] != 0]

    # P0
    p0_violations = []
    for c in alpha0:
        for k, v in c["cmp"]["beats_M1"].items():
            if v["beats"]:
                p0_violations.append((c["structure"], c["max_order"], c["alpha"],
                                      k, v["mean_diff"]))
    P0 = dict(supported=(len(p0_violations) == 0), violations=p0_violations)

    # P1: necessary conditions + saturation map
    p1_violations = []
    sat = {}
    for c in agg:
        key = "%s/mo%d/a%d" % (c["structure"], c["max_order"], c["alpha"])
        sat[key] = saturation_order(c)
        if c["cmp"]["M3_2_vs_1"]["beats"] and not (c["alpha"] > 0):
            p1_violations.append(("M3@2>M3@1 with no 2-way signal", key))
        if c["cmp"]["M3_3_vs_2"]["beats"] and not (c["alpha"] > 0 and c["max_order"] == 3):
            p1_violations.append(("M3@3>M3@2 with no aggregate 3-way signal", key))
    P1 = dict(supported=(len(p1_violations) == 0), violations=p1_violations,
              saturation=sat)

    # P2: QUANTITY (M3)
    p2 = dict(homogeneous=[], heterogeneous=[], sparse=[])
    for c in pos:
        s = c["structure"]
        cellname = "mo%d/a%d" % (c["max_order"], c["alpha"])
        if s == "homogeneous":
            gk = "M3@%d" % c["max_order"]
            # sufficiency: power sums at least match the micro model
            # (M2 does not beat M3@max beyond noise); report if M3 beats M1 too.
            p2[s].append(dict(cell=cellname, m3=gk, gain=c["interfaces"][gk]["gain"],
                              ll_m1=c["interfaces"]["M1"]["ll"][0],
                              ll_m3=c["interfaces"][gk]["ll"][0],
                              ll_m2=c["interfaces"]["M2"]["ll"][0],
                              sufficient=(not beats_keys(c, gk, "M2")),
                              m3_beats_m1=beats_keys(c, "M1", gk),
                              winner=winner(c)))
        else:
            g = c["interfaces"]["M3@3"]["gain"]
            m3_beats_m1 = any(c["cmp"]["beats_M1"][k]["beats"]
                              for k in ["M3@1", "M3@2", "M3@3"])
            p2[s].append(dict(cell=cellname, gain_M3at3=g, m3_beats_m1=m3_beats_m1,
                              m3_worse_than_m1=beats_keys(c, "M3@3", "M1"),
                              degenerate=(not c["denom_meaningful"])))
    p2_homog_ok = all(d["sufficient"] for d in p2["homogeneous"])
    # heterogeneous prediction = "partial" (0 < gain < 0.9). Degenerate-denominator
    # (CASE B, gain undefined) cells are NOT refutations; they are excluded from the
    # rollup and reported separately, so a CASE B cell is never scored "not supported".
    het_eval = [d for d in p2["heterogeneous"] if not d["degenerate"]]
    p2_het_partial = (len(het_eval) > 0 and
                      all(0.0 < d["gain_M3at3"] < P["GAIN_WIN"] for d in het_eval))
    p2_sparse_null = all((not d["m3_beats_m1"]) for d in p2["sparse"])  # robust: uses beats, not gain
    P2 = dict(homogeneous_sufficient=p2_homog_ok, heterogeneous_partial=p2_het_partial,
              sparse_blind=p2_sparse_null, detail=p2)

    # P3: STRUCTURE (M4)
    # COMPACT_FRAC is a REPORTING HEURISTIC only (NOT a pre-registered threshold):
    # it binarizes the qualitative "M4 not compact (many nonzeros)" prediction as
    # "M4 uses a majority of the 175 micro features". The binding result is the
    # winners map + the reported nonzero counts; this fraction enters no winner().
    COMPACT_FRAC = 0.5
    p3 = dict(homogeneous=[], heterogeneous=[], sparse=[])
    for c in pos:
        s = c["structure"]
        nnz = c["interfaces"]["M4"]["dl"][0]
        p3[s].append(dict(cell="mo%d/a%d" % (c["max_order"], c["alpha"]),
                          gain_M4=c["interfaces"]["M4"]["gain"], nonzeros=nnz,
                          nonzero_frac=nnz / DL_FIXED["M2"],
                          m4_approaches_m2=(not beats_keys(c, "M4", "M2")),
                          degenerate=(not c["denom_meaningful"]),
                          winner=winner(c)))
    sp_eval = [d for d in p3["sparse"] if not d["degenerate"]]
    p3_sparse_ok = (len(sp_eval) > 0 and
                    all(d["gain_M4"] >= P["GAIN_WIN"] and d["nonzeros"] < DL_FIXED["M2"]
                        for d in sp_eval))
    p3_homog_nocompress = all((d["nonzeros"] > 3) for d in p3["homogeneous"])
    # heterogeneous "dense, approaches M2": spec-grounded approach (M4 within noise of
    # M2, locked beyond-noise rule) AND not compact (majority of micro features; heuristic).
    p3_het_dense = all((d["m4_approaches_m2"] and d["nonzero_frac"] >= COMPACT_FRAC)
                       for d in p3["heterogeneous"])
    P3 = dict(sparse_structure_wins=p3_sparse_ok, homogeneous_no_compression=p3_homog_nocompress,
              heterogeneous_dense=p3_het_dense, compact_frac_heuristic=COMPACT_FRAC, detail=p3)

    # P4: winners map
    winners = {}
    for c in agg:
        winners["%s/mo%d/a%d" % (c["structure"], c["max_order"], c["alpha"])] = winner(c)
    P4 = dict(winners=winners)

    return dict(P0=P0, P1=P1, P2=P2, P3=P3, P4=P4)


# ---------------------------------------------------------------------------
# Sanity checks (assert-style reporting; flagged, never silently passed)
# ---------------------------------------------------------------------------
def sanity(agg, adj):
    flags = []
    for c in agg:
        brt = c["base_rate_train"][0]
        bre = c["base_rate_test"][0]
        if not (0.45 <= brt <= 0.55 and 0.45 <= bre <= 0.55):
            flags.append("BASE_RATE off in %s/mo%d/a%d: train=%.3f test=%.3f"
                         % (c["structure"], c["max_order"], c["alpha"], brt, bre))
        if not c["cmp"]["M2_insample_dominates"]:
            flags.append("M2_INSAMPLE not dominant in %s/mo%d/a%d"
                         % (c["structure"], c["max_order"], c["alpha"]))
        if c["structure"] == "homogeneous" and c["alpha"] != 0:
            gk = "M3@%d" % c["max_order"]
            if beats_keys(c, gk, "M2"):
                flags.append("HOMOG_SUFFICIENCY violated in mo%d/a%d: "
                             "M2 beats %s beyond noise (power sums not sufficient)"
                             % (c["max_order"], c["alpha"], gk))
    if not adj["P0"]["supported"]:
        flags.append("P0 NULL VIOLATED (pipeline overfits): %s"
                     % (adj["P0"]["violations"],))
    return flags


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def fmt(ms, nd=4):
    return "%.*f+-%.*f" % (nd, ms[0], nd, ms[1])


def write_results_md(agg, adj, flags, path):
    L = []
    L.append("# Results: Persistence-Interface Toy\n")
    L.append("Out-of-sample TEST log-loss (lower is better), mean +- std over "
             "%d replicates. Held-out configs. See pre_registration.md.\n" % P["R"])
    L.append("Winner = lowest-description-length interface with "
             "gain_captured >= %.2f (P4); at alpha=0 winner is M1 iff the null "
             "holds.\n" % P["GAIN_WIN"])

    # log-loss table
    L.append("## Test log-loss (primary)\n")
    head = "| structure | mo | alpha | " + " | ".join(IFACES) + " | winner |"
    L.append(head)
    L.append("|" + "---|" * (4 + len(IFACES)))
    for c in agg:
        row = ["%s" % c["structure"], "%d" % c["max_order"], "%d" % c["alpha"]]
        for k in IFACES:
            row.append(fmt(c["interfaces"][k]["ll"]))
        row.append("**%s**" % winner(c))
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # gain_captured + DL table
    L.append("## gain_captured (alpha>0) and description length\n")
    L.append("gain_captured = (LL_M1 - LL_M)/(LL_M1 - LL_M2); M2=1.0 by "
             "definition; may exceed 1.0 (not clipped). DL in [feature/nonzero "
             "count].\n")
    head = ("| structure | mo | alpha | M3@1 | M3@2 | M3@3 | M4 (gain / nnz) | "
            "winner |")
    L.append(head)
    L.append("|---|---|---|---|---|---|---|---|")
    for c in agg:
        if c["alpha"] == 0:
            continue
        def g(k):
            v = c["interfaces"][k]["gain"]
            return "n/a" if v is None else "%.2f" % v
        m4 = c["interfaces"]["M4"]
        row = ["%s" % c["structure"], "%d" % c["max_order"], "%d" % c["alpha"],
               g("M3@1"), g("M3@2"), g("M3@3"),
               "%s / %s" % (("n/a" if m4["gain"] is None else "%.2f" % m4["gain"]),
                            fmt(m4["dl"], 1)),
               "**%s**" % winner(c)]
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # AUC table
    L.append("## Test AUC (secondary)\n")
    head = "| structure | mo | alpha | " + " | ".join(IFACES) + " |"
    L.append(head)
    L.append("|" + "---|" * (3 + len(IFACES)))
    for c in agg:
        row = ["%s" % c["structure"], "%d" % c["max_order"], "%d" % c["alpha"]]
        for k in IFACES:
            row.append(fmt(c["interfaces"][k]["auc"]))
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # ladder saturation
    L.append("## Ladder saturation order (P1 read-out)\n")
    L.append("Highest k in {1,2,3} where M3@k beats M3@(k-1) beyond noise.\n")
    L.append("| structure | mo | alpha | M3@2>M3@1 | M3@3>M3@2 | saturation order |")
    L.append("|---|---|---|---|---|---|")
    for c in agg:
        L.append("| %s | %d | %d | %s | %s | %d |" % (
            c["structure"], c["max_order"], c["alpha"],
            c["cmp"]["M3_2_vs_1"]["beats"], c["cmp"]["M3_3_vs_2"]["beats"],
            saturation_order(c)))
    L.append("")

    # base rate / sanity
    L.append("## Sanity\n")
    if flags:
        L.append("FLAGS RAISED:\n")
        for f in flags:
            L.append("- %s" % f)
    else:
        L.append("All sanity checks passed (base rate ~0.5, M2 in-sample "
                 "dominant, homogeneous sufficiency, P0 null).")
    L.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(L))


def write_adjudication_md(agg, adj, flags, path):
    L = []
    L.append("# Adjudication: P0-P4\n")
    L.append("Verdicts computed by the LOCKED rules in pre_registration.md from "
             "the locked results. Predictions are unchanged. Nulls and failures "
             "are reported as findings.\n")

    sup = lambda b: "SUPPORTED" if b else "NOT SUPPORTED"

    # P0
    L.append("## P0 NULL (overfit catch)\n")
    L.append("**%s.** At alpha=0 (purely additive substrate), no interface "
             "(M3@*, M4, M2) beats M1 out-of-sample beyond noise.\n"
             % sup(adj["P0"]["supported"]))
    if adj["P0"]["violations"]:
        L.append("Violations (interface beats M1 at alpha=0):\n")
        for v in adj["P0"]["violations"]:
            L.append("- %s/mo%d/a%d: %s beats M1 by %.4f log-loss" % v)
        L.append("\nPer the locked rule this is a PIPELINE FAILURE: the "
                 "regularization/CV/sample-size must be fixed and the experiment "
                 "re-run before interpreting other cells.")
    else:
        L.append("No interface beats M1 at any alpha=0 cell beyond noise. The "
                 "held-out-config split plus regularization prevent the pipeline "
                 "from manufacturing structure where the substrate is additive.")
    L.append("")

    # P1
    L.append("## P1 LADDER SATURATION\n")
    L.append("**%s.** M3@k beats M3@(k-1) beyond noise only where the substrate "
             "carries genuine aggregate k-way signal (necessary conditions: "
             "2>1 requires alpha>0; 3>2 requires alpha>0 and max_order=3).\n"
             % sup(adj["P1"]["supported"]))
    if adj["P1"]["violations"]:
        L.append("Necessary-condition violations:\n")
        for desc, key in adj["P1"]["violations"]:
            L.append("- %s at %s" % (desc, key))
        L.append("")
    L.append("Saturation order per cell (primary read-out):\n")
    L.append("| cell | saturation order |")
    L.append("|---|---|")
    for k, v in adj["P1"]["saturation"].items():
        L.append("| %s | %d |" % (k, v))
    L.append("")

    # P2
    L.append("## P2 QUANTITY (M3 = identity-blind power sums)\n")
    d = adj["P2"]
    L.append("- homogeneous (power sums sufficient: M2 does not beat M3@max "
             "beyond noise; identity-blind counts suffice): **%s**"
             % sup(d["homogeneous_sufficient"]))
    for x in d["detail"]["homogeneous"]:
        L.append("  - %s: LL[M1=%.4f %s=%.4f M2=%.4f]; sufficient (M3@max >= M2) "
                 "= %s; M3@max beats M1 = %s; winner = %s"
                 % (x["cell"], x["ll_m1"], x["m3"], x["ll_m3"], x["ll_m2"],
                    x["sufficient"], x["m3_beats_m1"], x["winner"]))
    L.append("- heterogeneous (partial: 0 < gain < 0.9; CASE B cells excluded "
             "from rollup): **%s**" % sup(d["heterogeneous_partial"]))
    for x in d["detail"]["heterogeneous"]:
        notes = []
        if x.get("degenerate"):
            notes.append("CASE B degenerate denominator, excluded from rollup")
        if x.get("m3_worse_than_m1"):
            notes.append("M3 WORSE than M1")
        suffix = (" [" + "; ".join(notes) + "]") if notes else ""
        L.append("  - %s: gain(M3@3) = %s, M3 beats M1 beyond noise = %s%s"
                 % (x["cell"], ("n/a" if x["gain_M3at3"] is None else "%.3f" % x["gain_M3at3"]),
                    x["m3_beats_m1"], suffix))
    L.append("- sparse (aggregate counts blind -> M3 ~ M1): **%s**"
             % sup(d["sparse_blind"]))
    for x in d["detail"]["sparse"]:
        L.append("  - %s: gain(M3@3) = %s, M3 beats M1 beyond noise = %s"
                 % (x["cell"], ("n/a" if x["gain_M3at3"] is None else "%.3f" % x["gain_M3at3"]),
                    x["m3_beats_m1"]))
    L.append("")

    # P3
    L.append("## P3 STRUCTURE (M4 = L1 sparse-specific)\n")
    d = adj["P3"]
    L.append("- sparse (M4 recovers gain at small nonzero count, compact wins): **%s**"
             % sup(d["sparse_structure_wins"]))
    for x in d["detail"]["sparse"]:
        L.append("  - %s: gain(M4) = %s, nonzeros = %.1f, winner = %s"
                 % (x["cell"], ("n/a" if x["gain_M4"] is None else "%.3f" % x["gain_M4"]),
                    x["nonzeros"], x["winner"]))
    L.append("- homogeneous (M4 no compression advantage over M3): **%s**"
             % sup(d["homogeneous_no_compression"]))
    for x in d["detail"]["homogeneous"]:
        L.append("  - %s: nonzeros = %.1f (vs M3 DL <= 3), winner = %s"
                 % (x["cell"], x["nonzeros"], x["winner"]))
    L.append("- heterogeneous (dense: M4 approaches M2 AND not compact; 'not "
             "compact' = nonzero fraction >= %.2f, a REPORTING HEURISTIC, not a "
             "locked threshold): **%s**"
             % (d["compact_frac_heuristic"], sup(d["heterogeneous_dense"])))
    for x in d["detail"]["heterogeneous"]:
        L.append("  - %s: nonzeros = %.1f (%.0f%% of micro), M4 approaches M2 = %s, "
                 "gain(M4) = %s, winner = %s"
                 % (x["cell"], x["nonzeros"], 100.0 * x["nonzero_frac"],
                    x["m4_approaches_m2"],
                    ("n/a" if x["gain_M4"] is None else "%.3f" % x["gain_M4"]),
                    x["winner"]))
    L.append("")

    # P4 winners map
    L.append("## P4 ADJUDICATION (winners map)\n")
    L.append("Winner = lowest-DL interface reaching gain_captured >= %.2f "
             "(M1 at alpha=0 iff null holds). 'Coherence is a good persistence "
             "interface' is SUPPORTED in a cell iff a compact interface wins: "
             "QUANTITY-sense if M3@k, STRUCTURE-sense if M4.\n" % P["GAIN_WIN"])
    L.append("| cell | winner | sense |")
    L.append("|---|---|---|")
    for k, w in adj["P4"]["winners"].items():
        if w.startswith("M3"):
            sense = "QUANTITY (compact)"
        elif w == "M4":
            sense = "STRUCTURE (compact)"
        elif w == "M2":
            sense = "micro irreducible"
        elif w == "M1":
            sense = "additive null"
        else:
            sense = "PIPELINE FAILURE"
        L.append("| %s | %s | %s |" % (k, w, sense))
    L.append("")

    # sanity
    L.append("## Sanity checks\n")
    if flags:
        L.append("FLAGS:\n")
        for f in flags:
            L.append("- %s" % f)
    else:
        L.append("All passed: base rate ~0.5; M2 in-sample dominant; "
                 "homogeneous sufficiency (M3@max ~ M2); P0 null holds.")
    L.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(L))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cells = []
    cid = 0
    for structure in STRUCTURES:
        for mo in MAX_ORDERS:
            for alpha in ALPHAS:
                cells.append((cid, structure, mo, alpha))
                cid += 1

    tasks = [(c[0], c[1], c[2], c[3], rep)
             for c in cells for rep in range(P["R"])]
    print("Running %d cells x %d replicates = %d fits ..."
          % (len(cells), P["R"], len(tasks)))
    records = Parallel(n_jobs=P["N_JOBS"], verbose=5)(
        delayed(run_cell_replicate)(*t) for t in tasks)

    agg = aggregate(records)
    adj = build_adjudication(agg)
    flags = sanity(agg, adj)

    # results.json: everything
    out = dict(
        meta=dict(
            master_seed=P["MASTER_SEED"],
            params={k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in P.items()},
            structures=STRUCTURES, max_orders=MAX_ORDERS, alphas=ALPHAS,
            replicates=P["R"], n_configs=FEAT["n_cfg"],
            full_feature_count=FEAT["XFULL"].shape[1],
            environment=dict(python=sys.version.split()[0],
                             platform=platform.platform(),
                             numpy=np.__version__, scikit_learn=sklearn.__version__,
                             scipy=scipy.__version__),
        ),
        cells=agg,
        adjudication=adj,
        sanity_flags=flags,
        raw_records=records,
    )
    with open("results.json", "w") as fh:
        json.dump(out, fh, indent=2, default=_json_default)
    write_results_md(agg, adj, flags, "results.md")
    write_adjudication_md(agg, adj, flags, "adjudication.md")

    print("\nWrote results.json, results.md, adjudication.md")
    print("P0 supported:", adj["P0"]["supported"])
    print("P1 supported:", adj["P1"]["supported"])
    print("Sanity flags:", len(flags))
    for f in flags:
        print("  FLAG:", f)


if __name__ == "__main__":
    main()
