"""EXPERIMENTAL system-level calibration of the biases d1 and d2.

NOT a reproduction of the paper (which gives no d1/d2).  We search a grid of
(d1, d2) for the values that make the *whole pipeline* closest to the golden
softmax on a synthetic workload, then check them on a different, held-out
workload.  The result depends on the workload and objective stated below.

Workload (uniform logits in [-r, r]):  r in {0.1, 1, 5, 10}  (the paper's
rand0.1/1/5/10 ranges) x N in a set of vector lengths x `per` vectors each.
Objective (primary, matches the paper's absolute-error flavour): mean over
vectors of RMS absolute error.  Also reported: mean relative error, mean
max-abs error, argmax agreement, mean |sum(p) - 1|.

Fast path: pass-1 depends on d1 only through F, and pass-2 output is
shift_unit(v + d2, u) with (u, v) independent of d2, so for each d1 all d2
values are evaluated by broadcasting.  The fast path is verified against the
real pipeline in the tests.
"""
import numpy as np
from config import WangConfig
from softmax_reference import softmax_reference
from preprocessing import preprocess
from expu import expu_pass1, expu, shift_unit
from adder_tree import accumulate_chunks
from lnu import lnu

RANGES = (0.1, 1.0, 5.0, 10.0)
CAL_LENGTHS = (8, 64, 512)            # calibration set
VAL_LENGTHS = (16, 128, 2048)         # held-out set (different N, different seed)
METRICS = ("mean_rel_err", "mean_max_abs_err", "mean_rms_err", "argmax_agree", "mean_abs_sum_dev")


def make_workload(seed, lengths, per=40):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-r, r, n) for r in RANGES for n in lengths for _ in range(per)]


def evaluate_grid(vectors, d1s, d2s, base_cfg=None):
    """Return {metric: array[len(d1s), len(d2s)]} averaged over `vectors`."""
    base_cfg = base_cfg or WangConfig()
    d1s, d2s = np.asarray(d1s, float), np.asarray(d2s, float)
    out = {m: np.zeros((len(d1s), len(d2s))) for m in METRICS}
    for i, d1 in enumerate(d1s):
        cfg = WangConfig(d1=float(d1), ln2_const=base_cfg.ln2_const,
                         log2e_const=base_cfg.log2e_const, lanes=base_cfg.lanes)
        for x in vectors:
            ref = softmax_reference(x)
            y = preprocess(x).y
            F = accumulate_chunks(expu_pass1(y, cfg).out, cfg.lanes, cfg)
            ln_f = lnu(F, cfg).ln_f
            t = expu(y, ln_f, 0.0, cfg)                   # (u, v) of pass 2
            u = np.broadcast_to(t.u, (len(d2s), len(y)))
            p = shift_unit(t.v[None, :] + d2s[:, None], u)
            err = p - ref[None, :]
            out["mean_rel_err"][i] += np.abs(err / ref[None, :]).mean(axis=1)
            out["mean_max_abs_err"][i] += np.abs(err).max(axis=1)
            out["mean_rms_err"][i] += np.sqrt((err ** 2).mean(axis=1))
            out["argmax_agree"][i] += (p.argmax(axis=1) == ref.argmax())
            out["mean_abs_sum_dev"][i] += np.abs(p.sum(axis=1) - 1.0)
    n = len(vectors)
    return {m: a / n for m, a in out.items()}


def best(result, metric="mean_rms_err", d1s=None, d2s=None, require_argmax=True):
    m = result[metric].copy()
    if require_argmax:
        m = np.where(result["argmax_agree"] >= 0.9999, m, np.inf)
    i, j = np.unravel_index(np.argmin(m), m.shape)
    return float(d1s[i]), float(d2s[j]), float(result[metric][i, j])


def calibrate(seed=1, per=40, verbose=True):
    vecs = make_workload(seed, CAL_LENGTHS, per)
    d1s = np.round(np.arange(0.90, 1.80001, 0.02), 4)
    d2s = np.round(np.arange(0.90, 1.10001, 0.005), 4)
    coarse = evaluate_grid(vecs, d1s, d2s)
    a, b, _ = best(coarse, d1s=d1s, d2s=d2s)
    d1f = np.round(np.arange(a - 0.04, a + 0.04001, 0.005), 4)
    d2f = np.round(np.arange(b - 0.02, b + 0.02001, 0.0025), 4)
    fine = evaluate_grid(vecs, d1f, d2f)
    d1, d2, val = best(fine, d1s=d1f, d2s=d2f)
    if verbose:
        print(f"coarse best d1={a}, d2={b}; refined best d1={d1}, d2={d2}  (mean RMS {val:.5f})")
    return d1, d2


def compare(configs: dict, seed, lengths, per=40):
    """Metrics for named (d1, d2) pairs on a workload."""
    vecs = make_workload(seed, lengths, per)
    rows = {}
    for name, (d1, d2) in configs.items():
        r = evaluate_grid(vecs, [d1], [d2])
        rows[name] = {m: float(r[m][0, 0]) for m in METRICS}
    return rows


if __name__ == "__main__":
    import coefficients as C
    d1, d2 = calibrate()
    configs = {
        "d1=d2=1.0 (chord, continuous)": (1.0, 1.0),
        "d1=d2=0.957 (local minimax)": (C.D_MINIMAX.value, C.D_MINIMAX.value),
        f"calibrated d1={d1}, d2={d2}": (d1, d2),
    }
    for title, seed, lengths in (("CALIBRATION set (seed 1, N=8/64/512)", 1, CAL_LENGTHS),
                                 ("HELD-OUT set (seed 2, N=16/128/2048)", 2, VAL_LENGTHS)):
        print("\n" + title)
        rows = compare(configs, seed, lengths)
        print(f"{'config':38s}" + "".join(f"{m:>18s}" for m in METRICS))
        for name, r in rows.items():
            print(f"{name:38s}" + "".join(f"{r[m]:18.5f}" for m in METRICS))
