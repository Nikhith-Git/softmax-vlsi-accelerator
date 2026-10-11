#!/usr/bin/env python3
"""Golden-vector exporter for tb/tb_softmax_top.v (uses the existing Python model).

Writes (default dir ./vectors):
  in.hex     one input beat per line, 64 hex digits = 16 x UQ8.8 t codes, lane i at bits [16i+:16]
  exp.hex    one expected output beat per line, 16 x UQ1.15 p codes, same lane order (pad lanes 0)
  meta.hex   line 0 = number of cases; line c = 160-bit word {reject,exp_status,out_off,in_off,n_elems}
             (each field 32 bits, n_elems in the low word; offsets in beats)
  cases.txt  index, name, n, expected status (human readable)
  params.txt RTL parameter values that match the vectors (ROUNDING/OVERFLOW/D1_CODE/D2_CODE/MAX_N)

Every expected value comes from the existing model functions (expu_fixed_pass1,
accumulate_chunks_fixed, lnu_fixed, expu_fixed_pass2); no arithmetic is reimplemented
except the status bits, which are derived from the model's own overflow flags.
Status bit map (matches softmax_top): 0 a ovf, 1 z ovf, 2 out ovf, 3 acc ovf, 4 lnu ovf,
5 F==0, 6 invalid n (reject cases only), 7 no zero-t element.
Cases whose F==0 are not representable by run_fixed (it raises); they are built from the
same stage functions with lnF forced to 0, as the RTL does.
"""
import argparse, dataclasses, os, sys
import numpy as np

ROUND = {"trunc": 0, "toward_zero": 1, "round_half_up": 2, "round_half_even": 3}
OVFL = {"saturate": 0, "wrap": 1}
LANES = 16


def find_model(arg):
    cands = [arg, os.environ.get("WANG_PY"), "wang_softmax_python",
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wang_softmax_python")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "wang_softmax.py")):
            return os.path.abspath(c)
    sys.exit("Python model not found: pass --model-dir or set WANG_PY")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--out", default="vectors")
    ap.add_argument("--d1", type=float, default=None, help="default: model's provisional d1")
    ap.add_argument("--d2", type=float, default=None, help="default: model's provisional d2")
    ap.add_argument("--rounding", default="trunc", choices=list(ROUND))
    ap.add_argument("--overflow", default="saturate", choices=list(OVFL))
    ap.add_argument("--max-n", type=int, default=1024, help="must equal the RTL MAX_N")
    ap.add_argument("--random", type=int, default=12, help="number of seeded random cases")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()

    sys.path.insert(0, find_model(a.model_dir))
    import config as K
    import fixed_point as FP
    from config import WangConfig
    from wang_softmax import WangSoftmax
    from preprocessing import preprocess_fixed
    from expu import expu_fixed_pass1, expu_fixed_pass2
    from adder_tree import accumulate_chunks_fixed
    from lnu import lnu_fixed

    fx = dataclasses.replace(K.FIXED_16BIT, rounding=a.rounding, overflow=a.overflow)
    kw = {}
    if a.d1 is not None: kw["d1"] = a.d1
    if a.d2 is not None: kw["d2"] = a.d2
    cfg = WangConfig(fixed=fx, lanes=LANES, **kw)
    align = fx.a_frac_bits - fx.input_frac_bits          # 7
    d1c = FP.quantize(cfg.d1, fx.fmt_m, fx.rounding, "saturate")
    d2c = FP.quantize(cfg.d2, fx.fmt_m, fx.rounding, "saturate")

    def golden(t):
        y = np.array([-(int(v) << align) for v in t], dtype=np.int64)
        p1 = expu_fixed_pass1(y, cfg)
        acc = accumulate_chunks_fixed(p1.out_code, LANES, fx, return_trace=True)
        F = acc["F"]
        if F == 0:                                       # RTL: flag + lnF forced to 0
            ln_code, fz, lnovf = 0, True, False
        else:
            ln = lnu_fixed(F, cfg)
            ln_code, fz, lnovf = ln.ln_f_code, False, ln.overflow
        p2 = expu_fixed_pass2(y, ln_code, cfg)
        st = 0
        st |= int(p1.overflow["a"] or p2.overflow["a"]) << 0
        st |= int(p1.overflow["z"] or p2.overflow["z"]) << 1
        st |= int(p1.overflow["out"] or p2.overflow["out"]) << 2
        st |= int(acc["overflow"]) << 3
        st |= int(lnovf) << 4
        st |= int(fz) << 5
        st |= int(0 not in [int(v) for v in t]) << 7
        return [int(v) for v in p2.out_code], st, F, ln_code

    cases = []          # dict(name, t|None, n, pad, reject)

    def add_t(name, t, pad=0):
        t = [int(v) for v in t]
        assert 1 <= len(t) <= a.max_n and all(0 <= v <= 0xFFFF for v in t), name
        cases.append(dict(name=name, t=t, n=len(t), pad=pad, reject=False))

    def add_x(name, x, pad=0):
        t = preprocess_fixed(np.array(x, dtype=np.float64), fx).t_code
        add_t(name, t, pad)

    def add_rej(name, n):
        cases.append(dict(name=name, t=None, n=n, pad=0, reject=True))

    # ---- directed ----
    add_x("example [2,1,0,-1]", [2, 1, 0, -1])
    add_x("equal N=3 zeros", [0, 0, 0])
    add_x("equal N=16", [0.0] * 16)
    add_x("equal N=17 (pad lanes 0xFFFF)", [0.0] * 17, pad=0xFFFF)
    add_x("equal N=64", [0.0] * 64)
    add_x("equal N=1024 (max)", [0.0] * a.max_n)
    add_rej("invalid n=0", 0)
    add_x("N=1 (single zero)", [3.0])
    for n in (15, 16, 17, 33):
        add_x(f"ramp N={n}", [-0.25 * i for i in range(n)])
    add_x("large diff [0,-100]", [0.0, -100.0])
    add_x("large diff mix", [0.0, -1.0, -10.0, -50.0, -100.0, -200.0, -255.0])
    add_t("t max [0,65535]", [0, 65535])
    add_t("t max all but one", [0] + [65535] * 20)
    add_t("z-int boundary t=4095/4096/4097", [0, 4095, 4096, 4097])
    add_t("z-int boundary t=8191/8192/8193", [0, 8191, 8192, 8193])
    add_t("z sat boundary 45589..45593", [0, 45589, 45590, 45591, 45592, 45593])
    add_t("tiny t [0,1,2,3]", [0, 1, 2, 3])
    add_rej("invalid n=max+1", a.max_n + 1)
    add_rej("invalid n=65535", 65535)
    # contract-violating / flag cases (software must not send these; RTL must flag them)
    add_t("no zero element [256,512,768]", [256, 512, 768], pad=0)
    add_t("no zero, F==0 [2894]", [2894])
    add_t("no zero, F==0 [65535]", [65535])
    add_t("pad lanes 0 hide no-zero N=3", [1000, 2000, 3000], pad=0)
    # ---- seeded random ----
    rng = np.random.default_rng(a.seed)
    for i in range(a.random):
        n = int(rng.integers(1, 200))
        r = float(rng.choice([0.5, 2.0, 8.0, 30.0]))
        add_x(f"random #{i} N={n} r={r}", rng.uniform(-r, r, size=n))

    # ---- build files ----
    in_lines, exp_lines, meta, names = [], [], [], []
    for ci, c in enumerate(cases, start=1):
        in_off, out_off = len(in_lines), len(exp_lines)
        if c["reject"]:
            st, exp_st = 0x40, 0x40
        else:
            p, st, F, ln = golden(c["t"])
            k = (c["n"] + LANES - 1) // LANES
            for b in range(k):
                tw = ow = 0
                for l in range(LANES):
                    i = b * LANES + l
                    tw |= (c["t"][i] if i < c["n"] else c["pad"]) << (16 * l)
                    ow |= (p[i] if i < c["n"] else 0) << (16 * l)
                in_lines.append("%064x" % tw)
                exp_lines.append("%064x" % ow)
            # cross-check against the model's own run_fixed when it is defined
            if 0 in c["t"] and F != 0:
                x = [-v / 256.0 for v in c["t"]]
                tr = WangSoftmax(cfg).run_fixed(np.array(x), debug=True)
                assert [int(v) for v in tr.p_code] == p and tr.F_code == F, c["name"]
            exp_st = st
        word = c["n"] | (in_off << 32) | (out_off << 64) | (exp_st << 96) | (int(c["reject"]) << 128)
        meta.append("%040x" % word)
        names.append("%3d  %-40s n=%-5d exp_status=0x%02x%s" %
                     (ci, c["name"], c["n"], exp_st, "  (reject)" if c["reject"] else ""))
    assert len(in_lines) <= 16384, "raise IN_DEPTH in the testbench"

    os.makedirs(a.out, exist_ok=True)
    def w(name, lines):
        with open(os.path.join(a.out, name), "w") as f:
            f.write("\n".join(lines) + "\n")
    w("in.hex", in_lines or ["0"])
    w("exp.hex", exp_lines or ["0"])
    w("meta.hex", ["%040x" % len(cases)] + meta)
    w("cases.txt", names)
    params = [f"ROUNDING={ROUND[a.rounding]}", f"OVERFLOW={OVFL[a.overflow]}",
              f"D1_CODE={d1c}", f"D2_CODE={d2c}", f"MAX_N={a.max_n}"]
    w("params.txt", params)
    print("\n".join(names))
    print(f"\n{len(cases)} cases, {len(in_lines)} input beats -> {a.out}/")
    print("RTL parameters matching these vectors: " + " ".join(params))


if __name__ == "__main__":
    main()
