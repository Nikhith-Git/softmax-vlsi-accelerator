"""Compare the fixed-point Wang model with the float Wang model and the exact softmax.

    python compare_fixed.py                          # [2,1,0,-1], 8-bit preset
    python compare_fixed.py --x 3 1 -2 0.5 --preset 16bit
    python compare_fixed.py --random 200 --len 64 --range 5 --preset 8bit
    python compare_fixed.py --preset 8bit --rounding round_half_up --overflow wrap
    python compare_fixed.py --formats               # print every intermediate's format

Three comparisons per vector (float Wang uses the SAME d1, d2, constants):
  fixed vs float   : fixed model vs float Wang on the ORIGINAL logits
  fixed vs float*  : fixed model vs float Wang on the QUANTIZED input (isolates
                     arithmetic error from input quantization)
  fixed vs exact   : fixed model vs exact softmax
  float vs exact   : float Wang vs exact softmax (the Stages 1-6 baseline)
Outputs are never renormalised.
"""
import argparse
import dataclasses
import numpy as np
import config as CFG
from config import WangConfig
from wang_softmax import WangSoftmax
from softmax_reference import softmax_reference


def error_metrics(p, ref) -> dict:
    p, ref = np.asarray(p, float), np.asarray(ref, float)
    e = p - ref
    return {"max_abs": float(np.max(np.abs(e))), "mean_abs": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e ** 2))), "sum_p": float(p.sum()),
            "argmax_match": bool(int(np.argmax(p)) == int(np.argmax(ref)))}


def compare_fixed(x, cfg: WangConfig) -> dict:
    """cfg.fixed must be set.  Returns outputs, metrics and the fixed trace."""
    if cfg.fixed is None:
        raise ValueError("compare_fixed needs cfg.fixed")
    fcfg = dataclasses.replace(cfg, fixed=None)
    tr = WangSoftmax(cfg).run(x, debug=True)
    p_exact = softmax_reference(x)
    p_float = WangSoftmax(fcfg).run(x)
    p_float_q = WangSoftmax(fcfg).run(tr.y_q)       # float model on quantized y
    return {"p_exact": p_exact, "p_float": p_float, "p_float_q": p_float_q,
            "p_fixed": tr.p_raw, "trace": tr,
            "fixed_vs_float": error_metrics(tr.p_raw, p_float),
            "fixed_vs_float_quantized_input": error_metrics(tr.p_raw, p_float_q),
            "fixed_vs_exact": error_metrics(tr.p_raw, p_exact),
            "float_vs_exact": error_metrics(p_float, p_exact)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, nargs="+", default=[2.0, 1.0, 0.0, -1.0])
    ap.add_argument("--preset", choices=sorted(CFG.PRESETS), default="8bit")
    ap.add_argument("--rounding", default=None, choices=sorted(set(CFG.FP.ROUNDING_MODES)))
    ap.add_argument("--overflow", default=None, choices=CFG.FP.OVERFLOW_MODES)
    ap.add_argument("--d1", type=float, default=WangConfig().d1)
    ap.add_argument("--d2", type=float, default=WangConfig().d2)
    ap.add_argument("--random", type=int, default=0, help="N random vectors instead of --x")
    ap.add_argument("--len", type=int, default=16)
    ap.add_argument("--range", type=float, default=5.0, help="logits uniform in [-r, r]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--formats", action="store_true")
    a = ap.parse_args()
    fx = CFG.PRESETS[a.preset]
    kw = {k: v for k, v in (("rounding", a.rounding), ("overflow", a.overflow)) if v}
    fx = dataclasses.replace(fx, **kw)
    cfg = WangConfig(d1=a.d1, d2=a.d2, fixed=fx)
    print(f"preset={a.preset} (EXPERIMENTAL; not from the paper)  d1={a.d1} d2={a.d2} (PROVISIONAL)")
    if a.formats:
        print(fx.describe())

    if a.random:
        rng = np.random.default_rng(a.seed)
        rows, ovf = [], 0
        for _ in range(a.random):
            r = compare_fixed(rng.uniform(-a.range, a.range, a.len), cfg)
            rows.append(r)
            ovf += r["trace"].any_overflow
        print(f"{a.random} vectors, length {a.len}, logits in [-{a.range}, {a.range}], "
              f"vectors with an overflow event: {ovf}")
        print(f"{'comparison':32s} {'max_abs(worst)':>14s} {'mean_abs':>10s} {'rmse':>10s} {'argmax ok':>10s}")
        for key in ("fixed_vs_float", "fixed_vs_float_quantized_input",
                    "fixed_vs_exact", "float_vs_exact"):
            m = [r[key] for r in rows]
            print(f"{key:32s} {max(v['max_abs'] for v in m):14.6f} "
                  f"{np.mean([v['mean_abs'] for v in m]):10.6f} "
                  f"{np.mean([v['rmse'] for v in m]):10.6f} "
                  f"{sum(v['argmax_match'] for v in m):>5d}/{len(m)}")
        return

    r = compare_fixed(a.x, cfg)
    tr = r["trace"]
    print(f"x = {a.x}\nt codes = {tr.t_code.tolist()}  y_q = {tr.y_q.tolist()}")
    print(f"F = {tr.F:.6f} (code {tr.F_code})   lnF = {tr.lnu.ln_f:.6f} "
          f"(w={tr.lnu.w}, code {tr.lnu.ln_f_code})   overflow events: {tr.overflow}")
    print("   i   exact       float-Wang  fixed-Wang  p_code   fixed-exact")
    for i in range(len(r["p_exact"])):
        print(f"  {i:2d} {r['p_exact'][i]:10.6f} {r['p_float'][i]:11.6f} {r['p_fixed'][i]:11.6f}"
              f" {int(tr.p_code[i]):7d} {r['p_fixed'][i] - r['p_exact'][i]:+12.6f}")
    for key in ("fixed_vs_float", "fixed_vs_float_quantized_input", "fixed_vs_exact", "float_vs_exact"):
        m = r[key]
        print(f"  {key:32s} max_abs={m['max_abs']:.6f} mean_abs={m['mean_abs']:.6f} "
              f"rmse={m['rmse']:.6f} sum={m['sum_p']:.6f} argmax_match={m['argmax_match']}")
    print("  (outputs NOT renormalised; sum != 1 is expected)")


if __name__ == "__main__":
    main()
