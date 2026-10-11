"""Full Wang softmax pipeline (floating-point architecture model).

    x --preprocess--> y = x - max
      --EXPU pass 1 (d1)--> e1_i ~ exp(y_i)
      --adder tree + accumulator--> F = sum_i e1_i
      --LNU--> lnF ~ ln F
      --EXPU pass 2 (d2, input y_i - lnF)--> p_i ~ exp(y_i - lnF)

Fixed-point mode (Stage 7): if cfg.fixed is a FixedConfig the SAME dataflow runs
on integer codes (preprocess_fixed -> expu_fixed_pass1 -> accumulate_chunks_fixed
-> lnu_fixed -> expu_fixed_pass2) and returns dequantized float probabilities
(exact: code * 2**-frac); debug=True returns a FixedWangTrace with every code.
cfg.fixed = None (default) leaves the floating-point model unchanged.
Fixed-point + renormalize_output is refused (no divider is modelled).

Wiring only; each block lives in its own module.  No library exp/log is used
here.  Outputs are NOT renormalised unless cfg.renormalize_output is True
(experimental, NOT in the paper); the trace keeps the raw outputs `p_raw`
either way so both can be reported.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

from config import WangConfig
from preprocessing import preprocess, preprocess_fixed
from expu import expu_pass1, expu_pass2, expu_fixed_pass1, expu_fixed_pass2
from adder_tree import accumulate_chunks, accumulate_chunks_fixed
from lnu import lnu, lnu_fixed
import fixed_point as FP


@dataclass(frozen=True)
class WangTrace:
    x: np.ndarray            # original logits
    x_max: float
    y: np.ndarray            # preprocessed logits (x - x_max)
    pass1: object            # ExpuTrace: a, z (base-2 value), u, v, mantissa, out
    F: float                 # adder-tree / accumulator sum of pass-1 outputs
    lnu: object              # LnuTrace: w, k, log2_approx, ln_f
    pass2: object            # ExpuTrace for y - lnF
    p_raw: np.ndarray        # pass-2 outputs, exactly as the hardware produces them
    p: np.ndarray            # returned probabilities (= p_raw unless renormalised)


@dataclass(frozen=True)
class FixedWangTrace:
    """Every integer code of the fixed-point pipeline (formats: cfg.fixed)."""
    fx: object               # FixedConfig
    x: np.ndarray
    x_max: float
    y: np.ndarray            # float x - x_max (before quantization)
    y_q: np.ndarray          # float y the hardware actually sees
    t_code: np.ndarray       # UQ fmt_t codes of t = -y
    y_code: np.ndarray       # SQ a-format codes of y = -t (aligned)
    pass1: object            # ExpuFixedTrace (bias d1)
    acc: dict                # accumulate_chunks_fixed trace (F, overflow, tree_bits, ...)
    F_code: int
    lnu: object              # LnuFixedTrace
    pass2: object            # ExpuFixedTrace (bias d2)
    p_code: np.ndarray       # UQ fmt_exp_out codes, exactly as hardware produces them
    p_raw: np.ndarray        # dequantized p_code (never renormalised)
    p: np.ndarray            # == p_raw
    overflow: dict           # stage -> bool: overflow occurred before saturate/wrap

    @property
    def F(self) -> float:
        return float(FP.dequantize(self.F_code, self.fx.fmt_acc))

    @property
    def any_overflow(self) -> bool:
        return any(self.overflow.values())


class WangSoftmax:
    def __init__(self, cfg: Optional[WangConfig] = None):
        self.cfg = cfg or WangConfig()

    def run_fixed(self, x, debug: bool = False):
        """Bit-exact fixed-point pipeline (requires cfg.fixed)."""
        cfg = self.cfg
        fx = cfg.fixed
        if fx is None:
            raise ValueError("cfg.fixed is None")
        if cfg.renormalize_output:
            raise ValueError("renormalize_output is not modelled in fixed point "
                             "(it needs a divider; not in the paper)")
        pre = preprocess_fixed(x, fx)
        align = fx.a_frac_bits - fx.input_frac_bits
        y_code = np.array([-(int(t) << align) for t in pre.t_code], dtype=np.int64)
        p1 = expu_fixed_pass1(y_code, cfg)
        acc = accumulate_chunks_fixed(p1.out_code, cfg.lanes, fx, return_trace=True)
        ln = lnu_fixed(acc["F"], cfg)
        p2 = expu_fixed_pass2(y_code, ln.ln_f_code, cfg)
        p = FP.dequantize(p2.out_code, fx.fmt_exp_out)
        if not debug:
            return p
        ovf = {"input": pre.overflow,
               "pass1_a": p1.overflow["a"], "pass1_z": p1.overflow["z"],
               "pass1_out": p1.overflow["out"], "acc": acc["overflow"],
               "lnu": ln.overflow,
               "pass2_a": p2.overflow["a"], "pass2_z": p2.overflow["z"],
               "pass2_out": p2.overflow["out"]}
        return FixedWangTrace(fx, pre.x, pre.x_max, pre.y, pre.y_q, pre.t_code, y_code,
                              p1, acc, acc["F"], ln, p2, p2.out_code, p, p, ovf)

    def run(self, x, debug: bool = False):
        cfg = self.cfg
        if cfg.fixed is not None:
            return self.run_fixed(x, debug)
        pre = preprocess(x)
        p1 = expu_pass1(pre.y, cfg)
        F = accumulate_chunks(p1.out, cfg.lanes, cfg)
        ln = lnu(F, cfg)
        p2 = expu_pass2(pre.y, ln.ln_f, cfg)
        p_raw = p2.out
        p = p_raw / p_raw.sum() if cfg.renormalize_output else p_raw   # experimental
        if debug:
            return WangTrace(pre.x, pre.x_max, pre.y, p1, F, ln, p2, p_raw, p)
        return p


def wang_softmax(x, cfg: Optional[WangConfig] = None, debug: bool = False):
    """Convenience wrapper: WangSoftmax(cfg).run(x, debug)."""
    return WangSoftmax(cfg).run(x, debug)
