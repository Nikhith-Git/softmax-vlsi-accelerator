"""Stage 8: evaluation and comparison of the three softmax implementations.

    1. exact   : softmax_reference.softmax_reference (float64, mathematical truth)
    2. float   : floating-point Wang model (WangConfig(), d1/d2 PROVISIONAL)
    3. fixed   : fixed-point Wang model, two EXPERIMENTAL presets
                 fixed8  = config.FIXED_8BIT   (UQ4.4 input, UQ1.8 output)
                 fixed16 = config.FIXED_16BIT  (UQ8.8 input, UQ1.15 output)
                 (rounding "trunc", overflow "saturate"; none of this is from the paper)

Everything reported is MEASURED by running the Python models.  No hardware
(area/power/timing) metric is produced or implied.

    python evaluate.py                 # full run, writes ./results/
    python evaluate.py --quick         # small smoke run (seconds)
    python evaluate.py --seed 7 --out results_seed7

Outputs (organised under the output directory, default <project>/results):
    report.md                 human-readable summary of everything below
    data/named_cases.json     required + boundary inputs: all probabilities and errors
    data/random_summary.csv   percentile table per (length, input range, comparison)
    data/random_vectors.csv   one row per random vector and comparison
    data/precision_sweep.csv  error vs fixed-point fraction bits
    data/equal_logit_sweep.csv  equal logits, N = 1..1024
    data/checks.json          pass/fail of every evaluation check
    data/analysis.json        argmax-mismatch attribution, overflow counts, worst float vector
    plots/*.png

Metric definitions (all vs a stated reference):
    abs_err[i]  = |p_hw[i] - p_ref[i]|        per output
    max_abs     = max_i abs_err ; mean_abs = mean_i abs_err ; rmse = sqrt(mean_i abs_err^2)
    sum_p       = sum_i p_hw   (Wang outputs are NOT renormalised, so != 1 is expected)
    rel_err[i]  = (p_hw[i] - p_ref[i]) / p_ref[i], reported ONLY where p_ref[i] >= REL_FLOOR
                  (below that the reference probability is so small that a relative
                  error mostly measures output resolution, not the algorithm)
    argmax_match            : first-index argmax of p_hw is among the reference maxima
    argmax_hw_tied          : p_hw has several equal maxima (resolution collapse)
    argmax_ref_in_hw_top    : a reference maximum is among p_hw's tied maxima
Percentiles use numpy's default linear interpolation.
"""
import argparse
import csv
import dataclasses
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

import config as CFG
import coefficients as C
from config import WangConfig, FixedConfig
from softmax_reference import softmax_reference
from wang_softmax import WangSoftmax, wang_softmax

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = PROJECT_ROOT / "results"
DEFAULT_SEED = 2026
REL_FLOOR = 1e-2          # relative error is reported only where the exact p >= this

FLOAT_CFG = WangConfig()                                   # d1, d2: PROVISIONAL
FIXED_CFGS = {"fixed8": WangConfig(fixed=CFG.FIXED_8BIT),
              "fixed16": WangConfig(fixed=CFG.FIXED_16BIT)}
IMPLS = ("float", "fixed8", "fixed16")
COMPARISONS = [("float", "exact"), ("fixed8", "exact"), ("fixed16", "exact"),
               ("fixed8", "float"), ("fixed16", "float")]

REQUIRED_CASES = {
    "required_2_1_0_-1": [2.0, 1.0, 0.0, -1.0],
    "required_1_1_1": [1.0, 1.0, 1.0],
    "required_0_0_0": [0.0, 0.0, 0.0],
    "required_10_9_8": [10.0, 9.0, 8.0],
    "required_-2_-1_0": [-2.0, -1.0, 0.0],
}
_ZB = 16.0 / 23.0         # z = y*1.4375 crosses an integer every 16/23 in y
BOUNDARY_CASES = {
    # (description, logits)
    "z_integer_below_k1": ("y = -16/23 + 1e-6: z just above integer -1", [0.0, -_ZB + 1e-6]),
    "z_integer_above_k1": ("y = -16/23 - 1e-6: z just below integer -1", [0.0, -_ZB - 1e-6]),
    "z_integer_below_k3": ("y = -48/23 + 1e-6", [0.0, -3 * _ZB + 1e-6]),
    "z_integer_above_k3": ("y = -48/23 - 1e-6", [0.0, -3 * _ZB - 1e-6]),
    "z_exact_integer": ("y = -16 gives z = -23 exactly", [0.0, -16.0]),
    "spread_8bit_limit": ("|y| = 15.9375: largest t of the 8-bit preset", [0.0, -15.9375]),
    "spread_beyond_8bit": ("|y| = 16.0: 8-bit input saturates", [0.0, -16.0625]),
    "spread_30": ("|y| = 30", [0.0, -30.0]),
    "spread_100": ("|y| = 100: exact p ~ 3.7e-44", [0.0, -100.0]),
    "dominant_20": ("one logit 20 above two equal ones", [20.0, 0.0, 0.0]),
    "dominant_100": ("one logit 100 above three equal ones", [100.0, 0.0, 0.0, 0.0]),
    "near_tie_1_32": ("gap 2^-5: below the 8-bit input LSB 2^-4", [1.0, 1.0 + 2.0 ** -5]),
    "near_tie_1_16": ("gap 2^-4: exactly one 8-bit input LSB", [1.0, 1.0 + 2.0 ** -4]),
    "near_tie_0.01": ("gap 0.01", [0.0, 0.01]),
    "offset_+1000": ("large common offset (shift invariance)", [1000.0, 999.0, 998.0]),
    "offset_-1000": ("large negative common offset", [-1000.0, -1001.0, -1002.0]),
    "single_element": ("N = 1: exact p = 1", [3.0]),
    "equal_N10": ("10 equal logits: F = 10*d1 -> k ~ 1.45, near the worst Mitchell k", [0.0] * 10),
    "equal_N256": ("256 equal logits: p = 0.0039 ~ 1 LSB of UQ1.8", [0.0] * 256),
    "equal_N1024": ("1024 equal logits: p = 0.00098 < 1 LSB of UQ1.8", [0.0] * 1024),
}

RANGES = [("u0.5", "uniform", 0.5), ("u2", "uniform", 2.0), ("u5", "uniform", 5.0),
          ("u10", "uniform", 10.0), ("n3", "normal", 3.0)]
LENGTHS = (4, 8, 16, 64, 256, 1024)
N_PER_LENGTH = {4: 200, 8: 200, 16: 200, 64: 100, 256: 50, 1024: 20}
PCTS = (50, 90, 95, 99, 100)


# =============================== metrics =====================================
def percentiles(values, qs=PCTS) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {f"p{q}": float("nan") for q in qs}
    return {f"p{q}": float(np.percentile(a, q)) for q in qs}


def compute_metrics(p_ref, p_hw, rel_floor: float = REL_FLOOR) -> dict:
    """All per-vector metrics of p_hw against p_ref (see module docstring)."""
    ref, hw = np.asarray(p_ref, float), np.asarray(p_hw, float)
    if ref.shape != hw.shape or ref.ndim != 1 or ref.size == 0:
        raise ValueError("p_ref and p_hw must be non-empty 1-D arrays of equal length")
    if not (np.all(np.isfinite(ref)) and np.all(np.isfinite(hw))):
        raise ValueError("probabilities must be finite")
    err = hw - ref
    ae = np.abs(err)
    meaningful = ref >= rel_floor
    rel = np.full(ref.shape, np.nan)
    rel[meaningful] = err[meaningful] / ref[meaningful]
    ref_top = np.flatnonzero(ref == ref.max())
    hw_top = np.flatnonzero(hw == hw.max())
    has_rel = bool(meaningful.any())
    return {
        "abs_err": ae, "rel_err": rel,
        "max_abs": float(ae.max()), "mean_abs": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "sum_ref": float(ref.sum()), "sum_hw": float(hw.sum()),
        "n_rel": int(meaningful.sum()), "n_zero_hw": int((hw == 0).sum()),
        "max_rel": float(np.nanmax(np.abs(rel))) if has_rel else float("nan"),
        "mean_rel": float(np.nanmean(np.abs(rel))) if has_rel else float("nan"),
        "argmax_match": bool(int(np.argmax(hw)) in ref_top),
        "argmax_hw_tied": bool(hw_top.size > 1),
        "argmax_ref_in_hw_top": bool(np.intersect1d(ref_top, hw_top).size > 0),
    }


def order_violations(x, p) -> int:
    """Number of adjacent pairs (logits sorted ascending) that break monotonicity:
    x strictly larger but p strictly smaller, or equal x but different p.
    The Wang datapath is monotone in the logits, so this must be 0."""
    x, p = np.asarray(x, float), np.asarray(p, float)
    o = np.argsort(x, kind="stable")
    xs, ps = x[o], p[o]
    inv = (xs[1:] > xs[:-1]) & (ps[1:] < ps[:-1])
    eq = (xs[1:] == xs[:-1]) & (ps[1:] != ps[:-1])
    return int(inv.sum() + eq.sum())


# ================== a-priori error envelope of the float model ================
def _fit_err_range(d: float, n: int = 1 << 16):
    """min / max of (v + d)/2**v - 1 over v in [0, 1] (v = 1 included: conservative)."""
    v = np.append(np.arange(n) / n, 1.0)
    e = (v + d) / np.exp2(v) - 1.0
    return float(e.min()), float(e.max())


def wang_relative_error_envelope(n_elems: int, spread: float, cfg: WangConfig = FLOAT_CFG):
    """A-PRIORI bounds [lo, hi] on p_wang[i]/p_exact[i] - 1 for the FLOAT model.

    Uses only design constants, N and spread = max(x) - min(x); derived from the
    exact identity (see decompose_wang):
        p_wang_i / p_exact_i = (1 + e2_i) * exp(-dL) / (F / S)
    with e1/e2 the relative errors of the two EXPU passes (constant error
    exp(a*(c*ln2-1)) times linear-fit error (v+d)/2**v), dL = lnF_hw - ln F the LNU
    error, F/S in [1+min e1, 1+max e1].  Intervals are combined independently, so
    the bound is valid but conservative.  Needs pass1_ln_f == 0 and d1, d2 >= 1.
    """
    if cfg.pass1_ln_f != 0.0 or cfg.d1 < 1.0 or cfg.d2 < 1.0:
        raise ValueError("envelope derived for pass1_ln_f = 0 and d1, d2 >= 1")
    ln2 = math.log(2.0)
    kappa = cfg.log2e_const * ln2 - 1.0            # log of the constant error per unit a
    spread = float(spread)

    def pass_range(d, a_lo, a_hi):
        f_lo, f_hi = _fit_err_range(d)
        k = [math.exp(a_lo * kappa), math.exp(a_hi * kappa)]
        return (1 + f_lo) * min(k) - 1.0, (1 + f_hi) * max(k) - 1.0

    e1_lo, e1_hi = pass_range(cfg.d1, -spread, 0.0)
    if 1 + e1_lo <= 0:
        raise ValueError("envelope invalid: pass-1 error range reaches -100 %")
    F_min = cfg.d1                                   # the max element contributes d1
    F_max = n_elems * max(cfg.d1, (1.0 + cfg.d1) / 2.0)
    mit_min = (1 / ln2 - 1) - math.log2(1 / ln2)     # min of (k-1) - log2 k  (-0.0861)
    c2 = cfg.ln2_const
    dl = [(c2 - ln2) * math.log2(F) for F in (F_min, F_max)]
    dL_lo, dL_hi = min(dl) + c2 * mit_min, max(dl)
    lnF_lo, lnF_hi = math.log(F_min) + dL_lo, math.log(F_max) + dL_hi
    e2_lo, e2_hi = pass_range(cfg.d2, -spread - lnF_hi, -lnF_lo)
    lo = (1 + e2_lo) * math.exp(-dL_hi) / (1 + e1_hi) - 1.0
    hi = (1 + e2_hi) * math.exp(-dL_lo) / (1 + e1_lo) - 1.0
    return lo, hi


def decompose_wang(x, cfg: WangConfig = FLOAT_CFG) -> dict:
    """Split the float Wang error of one vector into its three sources and verify
    the exact identity  p_wang_i / p_exact_i = (1 + e2_i) * exp(-dL) / (F/S).

      e1_i = pass-1 output / exp(y_i) - 1   (constant + linear-fit error)
      F/S  = adder-tree sum / true sum of exp(y)
      dL   = lnF_hw - ln F                  (LNU error, common-mode)
      e2_i = pass-2 output / exp(y_i - lnF_hw) - 1
    Elements whose exact exp underflows (< 1e-290) are excluded.
    """
    x = np.asarray(x, float)
    tr = WangSoftmax(cfg).run(x, debug=True)
    y = tr.y
    ok = np.exp(y) > 1e-290
    S = float(np.exp(y).sum())
    e1 = tr.pass1.out[ok] / np.exp(y[ok]) - 1.0
    e2 = tr.pass2.out[ok] / np.exp(tr.pass2.a[ok]) - 1.0
    dL = tr.lnu.ln_f - math.log(tr.F)
    pred = (1 + e2) * math.exp(-dL) / (tr.F / S)
    p_ref = softmax_reference(x)[ok]
    actual = tr.p_raw[ok] / p_ref
    return {"e1": e1, "e2": e2, "dL": dL, "F_over_S": tr.F / S, "ratio_actual": actual,
            "ratio_pred": pred, "identity_resid": float(np.max(np.abs(pred / actual - 1.0))),
            "mask": ok}


# ============================ running the models ==============================
def run_all(x) -> tuple:
    """-> (dict impl -> probabilities, dict impl -> overflow-stage flags)."""
    x = np.asarray(x, dtype=float)
    out = {"exact": softmax_reference(x), "float": wang_softmax(x, FLOAT_CFG)}
    flags = {}
    for name, cfg in FIXED_CFGS.items():
        tr = WangSoftmax(cfg).run(x, debug=True)
        out[name], flags[name] = tr.p_raw, tr.overflow
    return out, flags


def evaluate_vector(x) -> dict:
    """Run all implementations on one vector and compute every comparison + checks."""
    x = np.asarray(x, dtype=float)
    p, flags = run_all(x)
    mets = {f"{h}_vs_{r}": compute_metrics(p[r], p[h]) for h, r in COMPARISONS}
    dec = decompose_wang(x)
    lo, hi = wang_relative_error_envelope(x.size, float(x.max() - x.min()))
    ratio = dec["ratio_actual"] - 1.0
    return {"x": x, "p": p, "flags": flags, "metrics": mets,
            "identity_resid": dec["identity_resid"],
            "env": (lo, hi), "env_obs": (float(ratio.min()), float(ratio.max())),
            "env_violations": int(np.sum((ratio < lo) | (ratio > hi))),
            "order_viol": {n: order_violations(x, p[n]) for n in IMPLS},
            "unique_max": bool(np.sum(x == x.max()) == 1)}


# ============================ experiments =====================================
def _row_metrics(m: dict) -> dict:
    return {k: v for k, v in m.items() if k not in ("abs_err", "rel_err")}


def run_named_cases() -> dict:
    out = {}
    for name, x in REQUIRED_CASES.items():
        out[name] = {"group": "required", "description": "", "x": list(x)}
    for name, (desc, x) in BOUNDARY_CASES.items():
        out[name] = {"group": "boundary", "description": desc, "x": list(x)}
    for name, rec in out.items():
        r = evaluate_vector(rec["x"])
        rec["result"] = r
    return out


def equal_logit_sweep(ns=None) -> list:
    ns = ns or (list(range(1, 65)) + [96, 128, 192, 256, 384, 512, 768, 1024])
    rows = []
    for n in ns:
        p, _ = run_all(np.zeros(n))
        row = {"N": n, "p_exact": float(p["exact"][0])}
        for impl in IMPLS:
            row[f"p_{impl}"] = float(p[impl][0])
            row[f"abs_err_{impl}"] = abs(row[f"p_{impl}"] - row["p_exact"])
            row[f"sum_{impl}"] = float(p[impl].sum())
        rows.append(row)
    return rows


def cell_rng(seed: int, length: int, range_idx: int):
    return np.random.default_rng(np.random.SeedSequence([seed, length, range_idx]))


def draw_vector(rng, kind: str, scale: float, n: int):
    return rng.uniform(-scale, scale, n) if kind == "uniform" else rng.normal(0.0, scale, n)


def run_random(seed: int, lengths, n_per_length, ranges=RANGES) -> dict:
    """Reproducible random grid.  Each (length, range) cell has its own seeded
    generator (SeedSequence([seed, length, range_index])), so cells do not depend
    on run order or on which other cells are run."""
    cells = {}
    for length in lengths:
        for ri, (label, kind, scale) in enumerate(ranges):
            rng = cell_rng(seed, length, ri)
            recs = [evaluate_vector(draw_vector(rng, kind, scale, length))
                    for _ in range(n_per_length[length])]
            cells[(length, label)] = {"length": length, "range": label, "kind": kind,
                                      "scale": scale, "n": len(recs), "recs": recs}
    return cells


def summarise_cell(cell: dict) -> list:
    """One row per comparison with percentile statistics over the cell's vectors."""
    rows = []
    recs = cell["recs"]
    for h, r in COMPARISONS:
        key = f"{h}_vs_{r}"
        ms = [rec["metrics"][key] for rec in recs]
        pooled = np.concatenate([m["abs_err"] for m in ms])
        rel = np.concatenate([m["rel_err"] for m in ms])
        rel = np.abs(rel[~np.isnan(rel)])
        row = {"length": cell["length"], "range": cell["range"], "comparison": key,
               "n_vectors": cell["n"]}
        for q, v in percentiles([m["max_abs"] for m in ms]).items():
            row[f"max_abs_{q}"] = v
        row["mean_abs_mean"] = float(np.mean([m["mean_abs"] for m in ms]))
        row["rmse_mean"] = float(np.mean([m["rmse"] for m in ms]))
        for q, v in percentiles(pooled, (50, 90, 99, 100)).items():
            row[f"per_output_abs_{q}"] = v
        s = [m["sum_hw"] for m in ms]
        row.update(sum_hw_min=float(np.min(s)), sum_hw_mean=float(np.mean(s)),
                   sum_hw_max=float(np.max(s)))
        row["argmax_match_rate"] = float(np.mean([m["argmax_match"] for m in ms]))
        row["argmax_hw_tied_count"] = int(sum(m["argmax_hw_tied"] for m in ms))
        row["n_rel_outputs"] = int(rel.size)
        for q, v in percentiles(rel, (50, 95, 100)).items():
            row[f"rel_abs_{q}"] = v
        row["zero_output_frac"] = float(np.mean(np.concatenate(
            [rec["p"][h] == 0 for rec in recs]))) if h in FIXED_CFGS else 0.0
        ov = 0
        if h in FIXED_CFGS:
            ov = int(sum(any(rec["flags"][h].values()) for rec in recs))
        row["vectors_with_overflow_event"] = ov
        rows.append(row)
    return rows


def uniform_fraction_config(f: int, rounding: str = "trunc") -> FixedConfig:
    """EXPERIMENTAL family for the precision sweep: every fraction field = f bits,
    integer parts as in the 8-bit preset (t UQ4.f, a/z SQ7.f, out UQ1.f, F UQ16.f)."""
    return FixedConfig(data_width=4 + f, input_frac_bits=f, exp_out_frac_bits=f,
                       exp_out_width=1 + f, acc_width=16 + f, acc_frac_bits=f,
                       a_width=8 + f, a_frac_bits=f, z_width=8 + f, z_frac_bits=f,
                       lnu_frac_bits=f, rounding=rounding)


def precision_sweep(seed: int, fracs, modes=("trunc", "round_half_up"), n_vec: int = 150):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 99991]))
    vecs = [rng.uniform(-5, 5, 16 if i % 2 == 0 else 64) for i in range(n_vec)]
    ref = [softmax_reference(x) for x in vecs]
    flt = [wang_softmax(x, FLOAT_CFG) for x in vecs]
    floor = [compute_metrics(r, f) for r, f in zip(ref, flt)]
    rows = []
    for mode in modes:
        for f in fracs:
            cfg = WangConfig(fixed=uniform_fraction_config(f, mode))
            m_ex, m_fl, m_fq = [], [], []
            for x, r, fl in zip(vecs, ref, flt):
                tr = WangSoftmax(cfg).run(x, debug=True)
                p_fq = wang_softmax(tr.y_q, FLOAT_CFG)          # float model, same quantized input
                m_ex.append(compute_metrics(r, tr.p_raw))
                m_fl.append(compute_metrics(fl, tr.p_raw))
                m_fq.append(compute_metrics(p_fq, tr.p_raw))
            rows.append({
                "rounding": mode, "frac_bits": f, "n_vectors": len(vecs),
                "vs_exact_mean_abs": float(np.mean([m["mean_abs"] for m in m_ex])),
                "vs_exact_max_abs_worst": float(max(m["max_abs"] for m in m_ex)),
                "vs_float_mean_abs": float(np.mean([m["mean_abs"] for m in m_fl])),
                "vs_float_max_abs_worst": float(max(m["max_abs"] for m in m_fl)),
                "vs_float_quantized_input_mean_abs": float(np.mean([m["mean_abs"] for m in m_fq])),
                "vs_float_quantized_input_max_abs_worst": float(max(m["max_abs"] for m in m_fq)),
                "argmax_match_rate_vs_exact": float(np.mean([m["argmax_match"] for m in m_ex])),
            })
    floor_row = {"float_vs_exact_mean_abs": float(np.mean([m["mean_abs"] for m in floor])),
                 "float_vs_exact_max_abs_worst": float(max(m["max_abs"] for m in floor))}
    return rows, floor_row


# ============================ error attribution ===============================
def argmax_attribution(cells: dict) -> dict:
    """Why does a fixed-point first-index argmax differ from the exact one?
    (vectors with a unique exact maximum only)
      gap_below_input_lsb : top-2 logit gap < input LSB -> both quantize to the same
                            code -> tied outputs (input quantization)
      other_ties          : gap >= input LSB but outputs still tie (output resolution)
      genuine_inversions  : the true maximum is NOT among the tied maxima (must be 0)"""
    out = {}
    for impl, cfg in FIXED_CFGS.items():
        lsb = 2.0 ** -cfg.fixed.input_frac_bits
        n = mis = gap = inv = gap_total = 0
        for c in cells.values():
            for r in c["recs"]:
                if not r["unique_max"] or r["x"].size < 2:
                    continue
                n += 1
                xs = np.sort(r["x"])[::-1]
                below = bool(xs[0] - xs[1] < lsb)
                gap_total += below
                m = r["metrics"][f"{impl}_vs_exact"]
                if not m["argmax_match"]:
                    mis += 1
                    gap += below
                    inv += (not m["argmax_ref_in_hw_top"])
        out[impl] = {"vectors": n, "input_lsb": lsb, "mismatches": mis,
                     "mismatch_gap_below_input_lsb": gap,
                     "mismatch_other_ties": mis - gap - inv, "genuine_inversions": inv,
                     "vectors_with_gap_below_input_lsb": gap_total}
    return out


def overflow_stage_counts(cells: dict) -> dict:
    out = {}
    for impl in FIXED_CFGS:
        cnt = {}
        for c in cells.values():
            for r in c["recs"]:
                for stage, f in r["flags"][impl].items():
                    cnt[stage] = cnt.get(stage, 0) + int(bool(f))
        out[impl] = {k: v for k, v in cnt.items() if v}
    return out


def worst_float_vector(cells: dict) -> dict:
    """Largest float-Wang absolute error on the random grid, decomposed into the
    three error sources (identity of decompose_wang)."""
    best = None
    for c in cells.values():
        for r in c["recs"]:
            m = r["metrics"]["float_vs_exact"]["max_abs"]
            if best is None or m > best[0]:
                best = (m, c["length"], c["range"], r)
    m, L, lab, r = best
    d = decompose_wang(r["x"])
    i = int(np.argmax(r["metrics"]["float_vs_exact"]["abs_err"]))
    lf = {"pass-2 EXPU (1+e2)": math.log(1 + d["e2"][i]), "LNU exp(-dL)": -d["dL"],
          "pass-1 sum 1/(F/S)": -math.log(d["F_over_S"])}
    return {"max_abs": m, "length": L, "range": lab, "index": i, "x_spread": float(np.ptp(r["x"])),
            "p_exact": float(r["p"]["exact"][i]), "p_float": float(r["p"]["float"][i]),
            "ratio_p_float_over_p_exact": float(d["ratio_actual"][i]),
            "factor_pass2_1_plus_e2": float(1 + d["e2"][i]),
            "factor_lnu_exp_minus_dL": math.exp(-d["dL"]),
            "factor_pass1_F_over_S": float(d["F_over_S"]), "lnu_error_dL": float(d["dL"]),
            "log_factors": lf, "dominant": max(lf, key=lambda k: abs(lf[k]))}


def worst_fixed_vectors(cells: dict) -> dict:
    """For each fixed preset: the random vector with the largest error vs the float Wang model
    on the original input, and the pass-1 mechanism behind it.  Compares F and lnF with the
    float model run on the SAME quantized input, and counts pass-1 outputs that truncated to 0."""
    out = {}
    for impl, cfg in FIXED_CFGS.items():
        key = f"{impl}_vs_float"
        best = None
        for c in cells.values():
            for r in c["recs"]:
                m = r["metrics"][key]["max_abs"]
                if best is None or m > best[0]:
                    best = (m, c["length"], c["range"], r)
        m, L, lab, r = best
        tr = WangSoftmax(cfg).run(r["x"], debug=True)
        ft = WangSoftmax(FLOAT_CFG).run(tr.y_q, debug=True)       # float model, same quantized input
        zero = tr.pass1.out_code == 0
        j = int(np.argmax(np.abs(r["p"][impl] - r["p"]["float"])))
        p_fq = float(wang_softmax(tr.y_q, FLOAT_CFG)[j])           # float model, quantized input
        f_dev = abs(float(tr.F) / float(ft.F) - 1.0)
        if f_dev > 0.01:
            mech = "pass-1 truncation: terms below one output LSB are dropped, F and lnF are biased low"
        elif abs(r["p"]["float"][j] - p_fq) > 0.5 * m:
            mech = ("input quantization moved z across an integer boundary: the linear fit is "
                    "discontinuous for d > 1, so u changes and the output jumps (up to 2*d2/(1+d2)-1)")
        else:
            mech = "mixed / no single dominant mechanism"
        out[impl] = {"mechanism": mech, "p_float_quantized_input": p_fq, "F_relative_deviation": f_dev,
            "max_abs_vs_float": m, "length": L, "range": lab, "output_index": j,
            "p_exact": float(r["p"]["exact"][j]), "p_float": float(r["p"]["float"][j]),
            "p_fixed": float(r["p"][impl][j]),
            "F_fixed": float(tr.F), "F_float_same_quantized_input": float(ft.F),
            "lnF_fixed": float(tr.lnu.ln_f), "lnF_float_same_quantized_input": float(ft.lnu.ln_f),
            "pass1_outputs_truncated_to_zero": int(zero.sum()), "pass1_outputs": int(zero.size),
            "true_sum_of_those_terms": float(ft.pass1.out[zero].sum()),
            "overflow_events": [k for k, v in r["flags"][impl].items() if v]}
    return out


# ================================ checks ======================================
def run_checks(named: dict, cells: dict) -> list:
    """Evaluation checks with explicit pass/fail.  None is a hand-picked accuracy
    tolerance: they test properties the design guarantees (monotonicity, an
    algebraic identity, an a-priori error envelope, shift invariance)."""
    recs = [r for c in cells.values() for r in c["recs"]]
    recs_all = recs + [rec["result"] for rec in named.values()]
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    resid = max(r["identity_resid"] for r in recs_all)
    add("float_error_identity", resid < 1e-9,
        f"max |predicted/actual - 1| = {resid:.3e} over {len(recs_all)} vectors (limit 1e-9)")
    viol = sum(r["env_violations"] for r in recs_all)
    worst_margin = min(min(r["env_obs"][0] - r["env"][0], r["env"][1] - r["env_obs"][1])
                       for r in recs_all)
    add("float_within_a_priori_envelope", viol == 0,
        f"{viol} outputs outside the envelope; smallest margin to a bound = {worst_margin:.4f} "
        f"(relative-error units)")
    uniq = [r for r in recs if r["unique_max"]]
    bad = sum(not r["metrics"]["float_vs_exact"]["argmax_match"] for r in uniq)
    add("float_argmax_equals_exact", bad == 0,
        f"{bad} mismatches in {len(uniq)} random vectors (float model is strictly monotone)")
    for impl in IMPLS:
        v = sum(r["order_viol"][impl] for r in recs_all)
        add(f"{impl}_monotone_in_logits", v == 0,
            f"{v} order violations in {len(recs_all)} vectors (equal logits -> equal outputs, "
            f"larger logit never gets a smaller output)")
    for impl in FIXED_CFGS:
        mis = [r for r in uniq if not r["metrics"][f"{impl}_vs_exact"]["argmax_match"]]
        untied = sum(not r["metrics"][f"{impl}_vs_exact"]["argmax_ref_in_hw_top"] for r in mis)
        add(f"{impl}_argmax_mismatch_only_via_ties", untied == 0,
            f"{len(mis)}/{len(uniq)} first-index mismatches; {untied} are genuine inversions "
            f"(the rest are output ties that include the true maximum)")
    for impl, cfg in FIXED_CFGS.items():
        fmt = cfg.fixed.fmt_exp_out
        bad = sum(float(np.min(r["p"][impl])) < 0 or float(np.max(r["p"][impl])) > fmt.max_value
                  for r in recs_all)
        add(f"{impl}_outputs_within_format", bad == 0,
            f"outputs must lie in [0, {fmt.max_value:.6f}] ({fmt.name}); {bad} vectors outside")
    # exact algebraic invariances of the model
    a, b = named["required_1_1_1"]["result"]["p"], named["required_0_0_0"]["result"]["p"]
    same = all(np.array_equal(a[k], b[k]) for k in a)
    add("shift_invariance_1_1_1_vs_0_0_0", same, "outputs identical for [1,1,1] and [0,0,0]")
    a = named["required_10_9_8"]["result"]["p"]
    c = named["required_-2_-1_0"]["result"]["p"]
    # Fixed point is integer arithmetic: permuting the inputs must be bit-exact.  The float
    # model adds the pass-1 terms in a different order, which changes F by rounding only
    # (~1 ulp); a relative tolerance of 1e-12 is justified by that (observed ~2e-16).
    exact_ok = all(np.array_equal(a[k], c[k][::-1]) for k in ("exact",) + tuple(FIXED_CFGS)
                   if k != "exact")
    fl_rel = float(np.max(np.abs(a["float"] - c["float"][::-1]) / np.abs(a["float"])))
    add("shift_and_permutation_10_9_8_vs_-2_-1_0", exact_ok and fl_rel < 1e-12,
        f"[10,9,8] vs [-2,-1,0] reversed (same y multiset): fixed8/fixed16 bit-exact = {exact_ok}; "
        f"float max rel diff = {fl_rel:.2e} (summation order; limit 1e-12)")
    off = [named[n]["result"]["p"] for n in ("offset_+1000", "required_10_9_8")]
    same = all(np.array_equal(off[0][k], off[1][k]) for k in IMPLS)
    add("large_offset_invariance", same, "[1000,999,998] equals [10,9,8] bit-for-bit")
    return checks


# =============================== output files =================================
def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_csv(path: Path, rows: list):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v)
                        for k, v in r.items()})


def named_case_json(named: dict) -> dict:
    out = {}
    for name, rec in named.items():
        r = rec["result"]
        item = {"group": rec["group"], "description": rec["description"], "x": rec["x"],
                "probabilities": {k: v for k, v in r["p"].items()},
                "overflow_events": {k: [s for s, f in fl.items() if f] for k, fl in r["flags"].items()},
                "comparisons": {}}
        for key, m in r["metrics"].items():
            item["comparisons"][key] = {**_row_metrics(m), "abs_err": m["abs_err"],
                                        "rel_err": m["rel_err"]}
        out[name] = item
    return _clean(out)


def vector_rows(cells: dict) -> list:
    rows = []
    for (length, label), cell in cells.items():
        for i, rec in enumerate(cell["recs"]):
            for key, m in rec["metrics"].items():
                rows.append({"length": length, "range": label, "index": i, "comparison": key,
                             "max_abs": m["max_abs"], "mean_abs": m["mean_abs"], "rmse": m["rmse"],
                             "sum_hw": m["sum_hw"], "argmax_match": int(m["argmax_match"]),
                             "argmax_hw_tied": int(m["argmax_hw_tied"]),
                             "max_rel": m["max_rel"], "n_rel": m["n_rel"]})
    return rows


# ================================== plots =====================================
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


COLORS = {"exact": "#222222", "float": "#1f77b4", "fixed8": "#d62728", "fixed16": "#2ca02c"}
LABELS = {"exact": "exact", "float": "float Wang", "fixed8": "fixed 8-bit preset",
          "fixed16": "fixed 16-bit preset"}


def plot_named(named, path):
    plt = _plt()
    names = list(REQUIRED_CASES)
    fig, ax = plt.subplots(2, len(names), figsize=(4 * len(names), 7))
    for j, name in enumerate(names):
        r = named[name]["result"]
        n = len(r["x"])
        idx = np.arange(n)
        for k, impl in enumerate(("exact",) + IMPLS):
            ax[0, j].bar(idx + (k - 1.5) * 0.2, r["p"][impl], 0.2, label=LABELS[impl],
                         color=COLORS[impl])
        ax[0, j].set_xticks(idx)
        ax[0, j].set_xticklabels([f"{v:g}" for v in r["x"]])
        ax[0, j].set_title(str(r["x"]), fontsize=10)
        ax[0, j].set_xlabel("logit")
        for k, impl in enumerate(IMPLS):
            ax[1, j].bar(idx + (k - 1) * 0.25, r["metrics"][f"{impl}_vs_exact"]["abs_err"], 0.25,
                         color=COLORS[impl], label=LABELS[impl])
        ax[1, j].set_xticks(idx)
        ax[1, j].set_xticklabels([f"{v:g}" for v in r["x"]])
        ax[1, j].set_xlabel("logit")
    ax[0, 0].set_ylabel("probability")
    ax[1, 0].set_ylabel("|p - p_exact|")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Required inputs: output probabilities (top) and per-output absolute error "
                 "vs exact softmax (bottom)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_distribution(cells, path):
    plt = _plt()
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    allrecs = [r for c in cells.values() for r in c["recs"]]
    for impl in IMPLS:
        v = np.sort([r["metrics"][f"{impl}_vs_exact"]["max_abs"] for r in allrecs])
        ax[0, 0].plot(v, np.arange(1, v.size + 1) / v.size, color=COLORS[impl], label=LABELS[impl],
                      lw=3.5 if impl == "float" else 1.5, ls="--" if impl == "fixed16" else "-")
    ax[0, 0].set_xscale("log")
    ax[0, 0].set(xlabel="per-vector max |p - p_exact|", ylabel="cumulative fraction of vectors",
                 title=f"CDF of max abs error vs exact ({len(allrecs)} random vectors)")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.3)
    pooled = {i: np.concatenate([r["metrics"][f"{i}_vs_exact"]["abs_err"] for r in allrecs])
              for i in IMPLS}
    bins = np.logspace(-8, 0, 60)
    for impl in IMPLS:
        ax[0, 1].hist(np.clip(pooled[impl], 1e-8, None), bins=bins, histtype="step",
                      color=COLORS[impl], label=LABELS[impl])
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set(xlabel="per-output |p - p_exact| (clipped at 1e-8)", ylabel="count",
                 title="Histogram of per-output absolute error (all outputs)")
    ax[0, 1].legend(fontsize=8)
    lengths = sorted({c["length"] for c in cells.values()})
    for k, impl in enumerate(IMPLS):
        data = [np.array([r["metrics"][f"{impl}_vs_exact"]["max_abs"]
                          for lab in [x[1] for x in cells if x[0] == L]
                          for r in cells[(L, lab)]["recs"]]) for L in lengths]
        pos = np.arange(len(lengths)) + (k - 1) * 0.27
        bp = ax[1, 0].boxplot(data, positions=pos, widths=0.24, patch_artist=True,
                              showfliers=False)
        for b in bp["boxes"]:
            b.set_facecolor(COLORS[impl])
            b.set_alpha(0.6)
    ax[1, 0].set_xticks(range(len(lengths)))
    ax[1, 0].set_xticklabels(lengths)
    ax[1, 0].set(xlabel="vector length N", ylabel="per-vector max abs error",
                 title="Max abs error vs exact by length (all ranges; fliers hidden)")
    ax[1, 0].legend([plt.Rectangle((0, 0), 1, 1, color=COLORS[i], alpha=0.6) for i in IMPLS],
                    [LABELS[i] for i in IMPLS], fontsize=8)
    for impl in IMPLS:
        rel = np.concatenate([r["metrics"][f"{impl}_vs_exact"]["rel_err"] for r in allrecs])
        rel = np.abs(rel[~np.isnan(rel)])
        v = np.sort(rel)
        ax[1, 1].plot(v, np.arange(1, v.size + 1) / v.size, color=COLORS[impl], label=LABELS[impl],
                      lw=3.5 if impl == "float" else 1.5, ls="--" if impl == "fixed16" else "-")
    ax[1, 1].set_xscale("log")
    ax[1, 1].set(xlabel="|relative error|", ylabel="cumulative fraction of outputs",
                 title=f"CDF of |relative error| (outputs with exact p >= {REL_FLOOR:g})")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_heatmaps(cells, path):
    plt = _plt()
    lengths = sorted({c["length"] for c in cells.values()})
    labels = [r[0] for r in RANGES if any(k[1] == r[0] for k in cells)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    grids = {impl: np.array([[np.percentile([r["metrics"][f"{impl}_vs_exact"]["max_abs"]
                                             for r in cells[(L, lab)]["recs"]], 95)
                              for L in lengths] for lab in labels]) for impl in IMPLS}
    vmax = max(g.max() for g in grids.values())            # same colour scale in all panels
    for a, impl in zip(ax, IMPLS):
        grid = grids[impl]
        im = a.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        a.set_xticks(range(len(lengths)))
        a.set_xticklabels(lengths)
        a.set_yticks(range(len(labels)))
        a.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(lengths)):
                a.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
        a.set_title(f"{LABELS[impl]}: p95 of max abs err vs exact")
        a.set_xlabel("vector length N")
        a.set_ylabel("input distribution")
        fig.colorbar(im, ax=a)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_scatter(cells, path, seed):
    plt = _plt()
    rng = np.random.default_rng(seed)
    allrecs = [r for c in cells.values() for r in c["recs"]]
    ex = np.concatenate([r["p"]["exact"] for r in allrecs])
    sel = rng.choice(ex.size, size=min(20000, ex.size), replace=False)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for a, impl in zip(ax, IMPLS):
        hw = np.concatenate([r["p"][impl] for r in allrecs])
        a.scatter(ex[sel], hw[sel], s=3, alpha=0.3, color=COLORS[impl])
        a.plot([0, 1], [0, 1], "k--", lw=0.8)
        a.set(xlabel="exact p", ylabel="p (model)", title=f"{LABELS[impl]} vs exact "
              f"({sel.size} of {ex.size} outputs)")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_precision(rows, floor, path):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    specs = [("vs_float_quantized_input", "vs float Wang on the same quantized input\n"
              "(pure arithmetic error)"),
             ("vs_float", "vs float Wang on the original input\n(+ input quantization)"),
             ("vs_exact", "vs exact softmax")]
    for a, (suffix, title) in zip(ax, specs):
        for mode, ls in (("trunc", "-"), ("round_half_up", "--")):
            rr = [r for r in rows if r["rounding"] == mode]
            f = [r["frac_bits"] for r in rr]
            a.plot(f, [r[f"{suffix}_mean_abs"] for r in rr], "o" + ls, color="#1f77b4",
                   label=f"mean abs ({mode})")
            a.plot(f, [r[f"{suffix}_max_abs_worst"] for r in rr], "s" + ls, color="#d62728",
                   label=f"worst max abs ({mode})")
        if suffix == "vs_exact":
            a.axhline(floor["float_vs_exact_mean_abs"], color="#1f77b4", ls=":", lw=1,
                      label="float Wang mean abs (floor)")
            a.axhline(floor["float_vs_exact_max_abs_worst"], color="#d62728", ls=":", lw=1,
                      label="float Wang worst max abs (floor)")
        a.set_yscale("log")
        a.set(xlabel="fraction bits f (all fields)", ylabel="absolute probability error",
              title=title)
        a.grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7)
    ax[2].legend(fontsize=7)
    fig.suptitle(f"Error vs fixed-point precision ({rows[0]['n_vectors']} random vectors, "
                 f"uniform [-5,5], N in {{16, 64}}); EXPERIMENTAL uniform-fraction formats")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_equal_sweep(rows, path):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    N = [r["N"] for r in rows]
    ax[0].plot(N, [r["p_exact"] for r in rows], "k-", label="exact 1/N")
    for impl in IMPLS:
        ax[0].plot(N, [r[f"p_{impl}"] for r in rows], ".-", ms=3, color=COLORS[impl],
                   label=LABELS[impl])
    ax[0].set(xscale="log", yscale="log", xlabel="N equal logits", ylabel="probability per output",
              title="Equal logits: p vs 1/N")
    ax[0].legend(fontsize=8)
    for impl in IMPLS:
        ax[1].plot(N, [r[f"p_{impl}"] / r["p_exact"] - 1 for r in rows], ".-", ms=3,
                   color=COLORS[impl], label=LABELS[impl])
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(xscale="log", xlabel="N equal logits", ylabel="relative error",
              title="Relative error (sawtooth = LNU/Mitchell + linear-fit error)")
    for impl in IMPLS:
        ax[2].plot(N, [r[f"sum_{impl}"] for r in rows], ".-", ms=3, color=COLORS[impl],
                   label=LABELS[impl])
    ax[2].axhline(1, color="k", lw=0.8)
    ax[2].set(xscale="log", xlabel="N equal logits", ylabel="sum of outputs",
              title="Probability sum (exact = 1; not renormalised)")
    for a in ax:
        a.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ================================== report ====================================
def md_table(headers, rows) -> str:
    esc = lambda c: str(c).replace("|", "\\|")
    lines = ["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _f(v, fmt="{:.5f}"):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


def build_report(meta, named, cells, summary_rows, sweep_rows, floor_row, equal_rows, checks,
                 extras) -> str:
    L = []
    L.append("# Stage 8 evaluation report\n")
    L.append("All numbers below were MEASURED by executing the Python models in this repository "
             "(seed and counts stated). No hardware (area, power, timing) metric is produced.\n")
    L.append("## Run configuration\n")
    for k, v in meta.items():
        if k not in ("plots", "limitations", "extras"):
            L.append(f"- **{k}**: {v}")
    L.append("")
    L.append("## Checks (pass/fail)\n")
    L.append(md_table(["check", "status", "detail"],
                      [[c["name"], c["status"], c["detail"]] for c in checks]))
    L.append("")
    for group, title in (("required", "Required inputs"), ("boundary", "Approximation boundary cases")):
        L.append(f"## {title}\n")
        for name, rec in named.items():
            if rec["group"] != group:
                continue
            r = rec["result"]
            x = rec["x"]
            L.append(f"### `{name}`  (N = {len(x)})" + (f" - {rec['description']}" if rec["description"] else ""))
            if len(x) <= 10:
                L.append(f"x = {x}\n")
                rows = []
                for i in range(len(x)):
                    rows.append([i] + [f"{r['p'][k][i]:.6f}" for k in ("exact",) + IMPLS]
                                + [f"{r['metrics'][f'{k}_vs_exact']['abs_err'][i]:.6f}" for k in IMPLS])
                L.append(md_table(["i", "exact", "float", "fixed8", "fixed16",
                                   "abs err float", "abs err fixed8", "abs err fixed16"], rows))
            else:
                L.append(f"x = {len(x)} logits (first 3: {x[:3]}, max-min = {max(x) - min(x):g}); "
                         f"exact p[0] = {r['p']['exact'][0]:.3e}; "
                         + "; ".join(f"{k} p[0] = {r['p'][k][0]:.3e}" for k in IMPLS))
            L.append("")
            rows = []
            for h, ref in COMPARISONS:
                m = r["metrics"][f"{h}_vs_{ref}"]
                rows.append([f"{h} vs {ref}", f"{m['max_abs']:.6f}", f"{m['mean_abs']:.6f}",
                             f"{m['rmse']:.6f}", f"{m['sum_hw']:.6f}",
                             ("yes (hw tied)" if m["argmax_hw_tied"] and not
                              np.all(np.asarray(r["p"][ref]) == np.max(r["p"][ref])) else "yes")
                             if m["argmax_match"] else ("TIE: first index wrong" if m["argmax_ref_in_hw_top"] else "NO"),
                             m["n_zero_hw"],
                             _f(m["max_rel"], "{:.2%}") + f" (n={m['n_rel']})"])
            L.append(md_table(["comparison", "max abs", "mean abs", "RMSE", "sum(p_hw)",
                               "argmax", "hw outputs == 0", f"max |rel| (p_ref >= {REL_FLOOR:g})"], rows))
            ov = {k: [s for s, f in fl.items() if f] for k, fl in r["flags"].items()}
            ov = {k: v for k, v in ov.items() if v}
            if ov:
                L.append(f"\nFixed-point overflow events (before saturation): {ov}")
            L.append("")
    L.append("## Random tests\n")
    L.append(f"Seed {meta['seed']}; each (length, distribution) cell uses its own generator "
             f"`SeedSequence([seed, length, range_index])`. Total random vectors: "
             f"**{meta['n_random_vectors']}**.\n")
    L.append("Per-cell vector counts: " + ", ".join(f"N={k}: {v}" for k, v in meta["n_per_length"].items())
             + " vectors per distribution; distributions: "
             + ", ".join(f"`{l}` = {'uniform[-%g, %g]' % (s, s) if kd == 'uniform' else 'normal(0, %g)' % s}"
                         for l, kd, s in RANGES) + ".\n")
    for key in ("float_vs_exact", "fixed8_vs_exact", "fixed16_vs_exact", "fixed8_vs_float", "fixed16_vs_float"):
        L.append(f"### {key.replace('_', ' ')}\n")
        rows = []
        for r in summary_rows:
            if r["comparison"] != key:
                continue
            rows.append([r["length"], r["range"], r["n_vectors"],
                         f"{r['max_abs_p50']:.5f}", f"{r['max_abs_p90']:.5f}", f"{r['max_abs_p95']:.5f}",
                         f"{r['max_abs_p99']:.5f}", f"{r['max_abs_p100']:.5f}",
                         f"{r['mean_abs_mean']:.5f}", f"{r['rmse_mean']:.5f}",
                         f"{r['sum_hw_min']:.3f}-{r['sum_hw_max']:.3f}",
                         f"{r['argmax_match_rate']:.3f}", _f(r["rel_abs_p95"], "{:.3f}"),
                         f"{r['zero_output_frac']:.3f}", r["vectors_with_overflow_event"]])
        L.append(md_table(["N", "dist", "n", "max abs p50", "p90", "p95", "p99", "max",
                           "mean abs (avg)", "RMSE (avg)", "sum(p) range", "argmax rate",
                           "abs rel err p95", "zero outputs", "overflow vecs"], rows))
        L.append("")
    L.append("(`max abs pXX` = percentile over vectors of the per-vector maximum absolute error; "
             "full per-output percentiles are in `data/random_summary.csv`.)\n")
    L.append("## Error analysis\n")
    w = extras["worst_float"]
    L.append("### Largest float-Wang error on the random grid\n")
    L.append(f"Max abs error {w['max_abs']:.4f} (N={w['length']}, distribution `{w['range']}`, logit spread "
             f"{w['x_spread']:.2f}), at output {w['index']}: exact {w['p_exact']:.4f}, float Wang "
             f"{w['p_float']:.4f}. Exact identity p_wang/p_exact = (1+e2) * exp(-dL) / (F/S): "
             f"{w['ratio_p_float_over_p_exact']:.4f} = {w['factor_pass2_1_plus_e2']:.4f} * "
             f"{w['factor_lnu_exp_minus_dL']:.4f} / {w['factor_pass1_F_over_S']:.4f} "
             f"(LNU error dL = {w['lnu_error_dL']:+.4f}). Dominant factor: {w['dominant']} "
             "(log-magnitudes: " + ", ".join(f"{k} {v:+.4f}" for k, v in w["log_factors"].items()) +
             "). d1, d2 are PROVISIONAL (Stage 6 calibrated them for mean RMS, not worst case).\n")
    L.append("### Largest fixed-point error (vs the float Wang model on the original input)\n")
    for impl, w in extras["worst_fixed"].items():
        L.append(f"- **{impl}**: max abs {w['max_abs_vs_float']:.4f} (N={w['length']}, `{w['range']}`), output "
                 f"{w['output_index']}: exact {w['p_exact']:.4f}, float {w['p_float']:.4f}, fixed "
                 f"{w['p_fixed']:.4f}. F = {w['F_fixed']:.4f} (float on the same quantized input: "
                 f"{w['F_float_same_quantized_input']:.4f}), lnF = {w['lnF_fixed']:.4f} "
                 f"(float: {w['lnF_float_same_quantized_input']:.4f}). Pass-1 outputs truncated to 0: "
                 f"{w['pass1_outputs_truncated_to_zero']} of {w['pass1_outputs']}, whose true sum is "
                 f"{w['true_sum_of_those_terms']:.4f}. Float model on the quantized input: "
                 f"{w['p_float_quantized_input']:.4f}. Input-stage overflow events: "
                 f"{w['overflow_events'] or 'none'}. **Mechanism (derived from these numbers):** {w['mechanism']}.")
    L.append("\nThese are resolution / quantization properties of the EXPERIMENTAL formats, not arithmetic "
             "bugs: the integer model is bit-exact against an independent implementation (Stage 7 tests).\n")
    L.append("### Fixed-point argmax disagreement (vectors with a unique exact maximum)\n")
    L.append(md_table(["impl", "vectors", "input LSB", "first-index mismatches",
                       "of which: top-2 gap < input LSB", "of which: output-resolution ties",
                       "genuine inversions", "all vectors with gap < input LSB"],
                      [[k, v["vectors"], v["input_lsb"], v["mismatches"], v["mismatch_gap_below_input_lsb"],
                        v["mismatch_other_ties"], v["genuine_inversions"],
                        v["vectors_with_gap_below_input_lsb"]] for k, v in extras["argmax"].items()]))
    L.append("\nA mismatch is a TIE at the top of the fixed-point output (np.argmax returns the first "
             "index), caused by input quantization (two logits closer than one input LSB collapse to the "
             "same code) or by output resolution (for large N each probability spans only a few output "
             "LSBs). It is never an order inversion. A consumer that needs the true argmax must not rely "
             "on first-index tie breaking.\n")
    L.append("### Fixed-point overflow events on the random grid (vectors per stage, before saturation)\n")
    L.append("\n".join(f"- {k}: {v if v else 'none'}" for k, v in extras["overflow"].items()))
    L.append("\n`input` = a logit more than the input range below the maximum was saturated (benign: its "
             "exact probability is ~0).\n")
    L.append("## Error vs fixed-point precision\n")
    L.append(f"{sweep_rows[0]['n_vectors']} random vectors, uniform[-5,5], N in {{16, 64}}. All fraction "
             "fields = f bits (EXPERIMENTAL family `uniform_fraction_config`). Float Wang vs exact on the "
             f"same vectors: mean abs {floor_row['float_vs_exact_mean_abs']:.5f}, worst max abs "
             f"{floor_row['float_vs_exact_max_abs_worst']:.5f}.\n")
    L.append(md_table(["rounding", "f", "mean abs vs float (quantized input)", "worst vs float (quantized input)",
                       "mean abs vs float", "mean abs vs exact", "worst max abs vs exact", "argmax rate vs exact"],
                      [[r["rounding"], r["frac_bits"], f"{r['vs_float_quantized_input_mean_abs']:.2e}",
                        f"{r['vs_float_quantized_input_max_abs_worst']:.2e}", f"{r['vs_float_mean_abs']:.2e}",
                        f"{r['vs_exact_mean_abs']:.2e}", f"{r['vs_exact_max_abs_worst']:.2e}",
                        f"{r['argmax_match_rate_vs_exact']:.3f}"] for r in sweep_rows]))
    L.append("\n## Plots\n")
    for p in meta["plots"]:
        L.append(f"- `plots/{p}`")
    L.append("\n## Limitations of this evaluation\n")
    L.extend(meta["limitations"])
    return "\n".join(L) + "\n"


# ================================== driver ====================================
def run_evaluation(out_dir: Path, seed: int = DEFAULT_SEED, quick: bool = False,
                   make_plots: bool = True, verbose: bool = True) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    lengths = (4, 16, 64) if quick else LENGTHS
    npl = {L: (6 if quick else N_PER_LENGTH[L]) for L in lengths}
    fracs = (4, 8, 12) if quick else (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20)
    say = print if verbose else (lambda *a, **k: None)

    say("named + boundary cases ...")
    named = run_named_cases()
    say("random grid ...")
    cells = run_random(seed, lengths, npl)
    say("equal-logit sweep ...")
    equal_rows = equal_logit_sweep([1, 2, 3, 4, 8, 10, 16] if quick else None)
    say("precision sweep ...")
    sweep_rows, floor_row = precision_sweep(seed, fracs, n_vec=10 if quick else 150)
    checks = run_checks(named, cells)
    summary_rows = [r for c in cells.values() for r in summarise_cell(c)]

    write_csv(out_dir / "data" / "random_summary.csv", summary_rows)
    write_csv(out_dir / "data" / "random_vectors.csv", vector_rows(cells))
    write_csv(out_dir / "data" / "precision_sweep.csv", sweep_rows)
    write_csv(out_dir / "data" / "equal_logit_sweep.csv", equal_rows)
    with open(out_dir / "data" / "named_cases.json", "w") as fh:
        json.dump(named_case_json(named), fh, indent=1)
    with open(out_dir / "data" / "checks.json", "w") as fh:
        json.dump(checks, fh, indent=1)

    plots = []
    if make_plots:
        plot_named(named, out_dir / "plots" / "required_inputs.png")
        plot_distribution(cells, out_dir / "plots" / "error_distribution.png")
        plot_heatmaps(cells, out_dir / "plots" / "error_by_length_and_range.png")
        plot_scatter(cells, out_dir / "plots" / "output_scatter.png", seed)
        plot_precision(sweep_rows, floor_row, out_dir / "plots" / "error_vs_precision.png")
        plot_equal_sweep(equal_rows, out_dir / "plots" / "equal_logit_sweep.png")
        plots = sorted(p.name for p in (out_dir / "plots").glob("*.png"))

    fx8, fx16 = CFG.FIXED_8BIT, CFG.FIXED_16BIT
    meta = {
        "seed": seed, "mode": "quick" if quick else "full",
        "n_random_vectors": sum(c["n"] for c in cells.values()),
        "n_per_length": npl,
        "python / numpy": f"{platform.python_version()} / {np.__version__}",
        "float Wang": f"log2e={FLOAT_CFG.log2e_const} (PAPER 1.0111b), ln2={FLOAT_CFG.ln2_const} (PAPER 0.1011b), "
                      f"d1={FLOAT_CFG.d1}, d2={FLOAT_CFG.d2} (PROVISIONAL, not in paper), pass-1 lnF={FLOAT_CFG.pass1_ln_f} (ASSUMED), "
                      f"lanes={FLOAT_CFG.lanes}",
        "fixed8": f"EXPERIMENTAL: t {fx8.fmt_t.name}, a {fx8.fmt_a.name}, out {fx8.fmt_exp_out.name}, "
                  f"F {fx8.fmt_acc.name}, rounding={fx8.rounding}, overflow={fx8.overflow}",
        "fixed16": f"EXPERIMENTAL: t {fx16.fmt_t.name}, a {fx16.fmt_a.name}, out {fx16.fmt_exp_out.name}, "
                   f"F {fx16.fmt_acc.name}, rounding={fx16.rounding}, overflow={fx16.overflow}",
        "relative error floor": f"reported only where exact p >= {REL_FLOOR:g}",
        "plots": plots,
        "limitations": [
            "- Software-model accuracy only; nothing here is measured hardware behaviour.",
            "- d1, d2, all fixed-point formats and the pass-1 lnF = 0 are provisional/experimental, "
            "not paper values; results describe THESE settings.",
            "- Random inputs are synthetic (uniform / normal logits), not logits from a trained network.",
            "- The a-priori envelope is conservative (independent interval combination), so passing it "
            "is a sanity check, not a tight accuracy claim.",
            "- Only the default rounding/overflow (trunc, saturate) is evaluated on the main grid; "
            "rounding mode is compared only in the precision sweep.",
        ]}
    extras = {"argmax": argmax_attribution(cells), "overflow": overflow_stage_counts(cells),
              "worst_float": worst_float_vector(cells), "worst_fixed": worst_fixed_vectors(cells)}
    with open(out_dir / "data" / "analysis.json", "w") as fh:
        json.dump(_clean(extras), fh, indent=1)
    report = build_report(meta, named, cells, summary_rows, sweep_rows, floor_row, equal_rows, checks,
                          extras)
    (out_dir / "report.md").write_text(report)
    return {"extras": extras, "named": named, "cells": cells, "summary_rows": summary_rows, "sweep_rows": sweep_rows,
            "floor_row": floor_row, "equal_rows": equal_rows, "checks": checks, "meta": meta}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    ap.add_argument("--quick", action="store_true", help="small smoke run")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    res = run_evaluation(a.out, a.seed, a.quick, not a.no_plots)
    fails = [c for c in res["checks"] if c["status"] == "FAIL"]
    print(f"\nwrote {a.out}  ({res['meta']['n_random_vectors']} random vectors, seed {a.seed})")
    for c in res["checks"]:
        print(f"  [{c['status']}] {c['name']}: {c['detail']}")
    if fails:
        print(f"\n{len(fails)} CHECK(S) FAILED - see report.md")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
