"""Prints every intermediate value of the Wang softmax model for one example.

    python run_demo.py                      # x = [2, 1, 0, -1]
    python run_demo.py --x 3 1 -2 0.5       # your own logits
    python run_demo.py --d1 1.0 --d2 1.0    # override the provisional biases
"""
import argparse
import numpy as np
import coefficients as C
from config import WangConfig
from wang_softmax import WangSoftmax
from softmax_reference import softmax_reference
from lnu import lnu_diagnostic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", type=float, nargs="+", default=[2.0, 1.0, 0.0, -1.0])
    ap.add_argument("--d1", type=float, default=C.D1.value)
    ap.add_argument("--d2", type=float, default=C.D2.value)
    a = ap.parse_args()
    cfg = WangConfig(d1=a.d1, d2=a.d2)
    tr = WangSoftmax(cfg).run(a.x, debug=True)
    ref = softmax_reference(a.x)

    print("CONFIGURATION")
    print(f"  log2e_const = {cfg.log2e_const}   (PAPER: 1.0111b)")
    print(f"  ln2_const   = {cfg.ln2_const}   (PAPER: 0.1011b)")
    print(f"  d1 = {cfg.d1}, d2 = {cfg.d2}   (PROVISIONAL: not in paper)")
    print(f"  pass-1 lnF input = {cfg.pass1_ln_f}   (ASSUMPTION)")

    print("\n1. PREPROCESSING  y = x - x_max")
    print(f"  x_max = {tr.x_max}")
    print("  x   =", np.round(tr.x, 6))
    print("  y   =", np.round(tr.y, 6))

    def expu_table(t, title):
        print(f"\n{title}")
        print("   a=y-lnF    z=a*log2e     u      v      m=v+d      out=m*2^u")
        for i in range(len(t.y)):
            print(f"  {t.a[i]:8.4f} {t.z[i]:10.4f} {t.u[i]:5d} {t.v[i]:8.4f} {t.mantissa[i]:9.4f} {t.out[i]:14.6f}")

    expu_table(tr.pass1, f"2. EXPU PASS 1 (d1 = {cfg.d1}, lnF = 0)")
    print(f"\n3. ADDER TREE / ACCUMULATOR   F = sum(pass-1 outputs) = {tr.F:.6f}")
    l = tr.lnu
    print(f"\n4. LNU   F = 2^w * k:  w = {l.w}, k = {l.k:.6f}")
    print(f"   k - 1 + w = {l.log2_approx:.6f}")
    print(f"   lnF = {cfg.ln2_const} * {l.log2_approx:.6f} = {l.ln_f:.6f}"
          f"   (exact ln F = {lnu_diagnostic(tr.F):.6f})")
    expu_table(tr.pass2, f"5. EXPU PASS 2 (d2 = {cfg.d2}, input y - lnF)")

    print("\n6. RESULT vs GOLDEN SOFTMAX")
    print("   i   golden      wang        abs err     rel err")
    for i in range(len(ref)):
        e = tr.p_raw[i] - ref[i]
        print(f"  {i:2d} {ref[i]:10.6f} {tr.p_raw[i]:11.6f} {e:+12.6f} {e / ref[i]:+10.4f}")
    err = np.abs(tr.p_raw - ref)
    print(f"  max abs err = {err.max():.6f}, mean abs err = {err.mean():.6f}")
    print(f"  sum(wang) = {tr.p_raw.sum():.6f}   (NOT renormalised; the paper does not renormalise)")
    renorm = tr.p_raw / tr.p_raw.sum()
    print(f"  [experimental] renormalised max abs err = {np.abs(renorm - ref).max():.6f}")
    print(f"  argmax golden = {ref.argmax()}, wang = {tr.p_raw.argmax()}")


if __name__ == "__main__":
    main()
