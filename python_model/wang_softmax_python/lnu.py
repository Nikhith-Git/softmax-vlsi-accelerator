"""LNU (paper Fig. 4, Eqs. 5-8): natural log of F = sum of first-pass outputs.

Convention (Eq. 5, Table I):  F = 2**w * k,   k in [1, 2),  w integer, unique.
  w = position of the leading 1 of F relative to the binary point
  k = F normalised so its leading 1 sits in the units place ("1.xxxx")

Approximation (Eqs. 6-8):
  ln F = ln2 * (w + log2 k)                 exact              (Eq. 6)
  log2 k ~ k - 1                            approximation      (Eq. 7)
  ln F ~ ln2_const * (k - 1 + w)            approximation      (Eq. 8)
with ln2_const = 0.1011b = 0.6875 (Sec. IV-B).

(k - 1 + w) needs no adder: k - 1 is in [0,1) so it is just the fractional
bits, and w is the integer part (two's complement for w < 0).

Exact error identity (derived, tested), with c = ln2_const:
  lnF_approx - ln F = (c - ln2) * log2 F + c * ((k - 1) - log2 k)
                      constant error         Mitchell error, in [-0.0861, 0]
For F >= 1 both terms are <= 0: the LNU underestimates ln F.  Because lnF is
subtracted from every input in pass 2, an LNU error delta scales EVERY output
by the same factor exp(-delta) (common-mode error).

The architectural model uses no library logarithm.  The LOD+shift is modelled
with math.frexp, which only reads the binary exponent/mantissa fields and is
exact.  The library logarithm appears only in the diagnostic functions.
"""
from dataclasses import dataclass
import math
import numpy as np
from config import WangConfig, FixedConfig
import fixed_point as FP


@dataclass(frozen=True)
class LnuTrace:
    F: float
    w: int              # leading-one position (LOD output), Eq. 5
    k: float            # F / 2**w, in [1, 2)
    k_minus_1: float    # fractional part
    log2_approx: float  # k - 1 + w   (Eq. 7)
    ln_f: float         # ln2_const * log2_approx  (Eq. 8)


def _check_positive_finite(F) -> float:
    try:
        f = float(F)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"F must be a real scalar: {exc}") from exc
    if np.ndim(F) != 0:
        raise ValueError("F must be a scalar")
    if math.isnan(f):
        raise ValueError("F is NaN")
    if math.isinf(f):
        raise ValueError("F is infinite (overflow)")
    if f <= 0.0:
        raise ValueError("F must be > 0 (LNU is undefined for zero or negative input)")
    return f


def lod(F: float):
    """Float model of LOD + shift: returns (w, k) with F = 2**w * k, k in [1,2)."""
    f = _check_positive_finite(F)
    m, e = math.frexp(f)          # f = m * 2**e, m in [0.5, 1)
    return e - 1, 2.0 * m


def lod_fixed(code: int, width: int, frac_bits: int):
    """Bit-level LOD for an unsigned fixed-point code (the Table I model).

    code: integer in [1, 2**width); value F = code / 2**frac_bits.
    Returns (w, k_code): w = (index of highest 1 bit) - frac_bits, and k_code
    is the width-bit normalised mantissa "1.xxxxxxx" (width-1 fraction bits,
    value k = k_code / 2**(width-1)).  Maps directly to a priority encoder
    plus a barrel shifter in RTL.
    """
    if not (isinstance(code, (int, np.integer)) and isinstance(width, int)
            and isinstance(frac_bits, int)):
        raise ValueError("code, width, frac_bits must be integers")
    if width < 2 or frac_bits < 0:
        raise ValueError("need width >= 2 and frac_bits >= 0")
    code = int(code)
    if code <= 0:
        raise ValueError("code must be > 0")
    if code >= (1 << width):
        raise ValueError("code does not fit in width bits (overflow)")
    msb = code.bit_length() - 1
    return msb - frac_bits, code << (width - 1 - msb)


def lnu(F: float, cfg: WangConfig) -> LnuTrace:
    """Floating-point LNU (Eqs. 5-8)."""
    w, k = lod(F)
    k_minus_1 = k - 1.0
    log2_approx = k_minus_1 + float(w)
    ln_f = cfg.ln2_const * log2_approx
    return LnuTrace(F=float(F), w=w, k=k, k_minus_1=k_minus_1,
                    log2_approx=log2_approx, ln_f=ln_f)


# ------------------------------ diagnostics ---------------------------------
def lnu_diagnostic(F: float) -> float:
    """DIAGNOSTIC ONLY (not hardware): exact ln(F) via numpy."""
    return float(np.log(_check_positive_finite(F)))


def lnu_errors(F_values, cfg: WangConfig) -> dict:
    """DIAGNOSTIC ONLY: LNU error against exact ln F, split by source.

    abs_err  = ln_f_hw - ln F
    rel_err  = abs_err / ln F  (NaN where |ln F| < 1e-12, i.e. F ~ 1: not meaningful)
    const_term   = (c - ln2) * log2 F      (error from ln2 ~ 0.6875)
    mitchell_term = c * ((k-1) - log2 k)   (error from log2 k ~ k - 1)
    prob_scale_err = exp(-abs_err) - 1     (relative error this LNU error puts on
                                            EVERY softmax output)
    """
    F = np.atleast_1d(np.array(F_values, dtype=np.float64))
    traces = [lnu(f, cfg) for f in F]
    ln_hw = np.array([t.ln_f for t in traces])
    k = np.array([t.k for t in traces])
    exact = np.log(F)
    abs_err = ln_hw - exact
    c = cfg.ln2_const
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(np.abs(exact) > 1e-12, abs_err / exact, np.nan)
    return {
        "F": F, "ln_f_hw": ln_hw, "ln_f_exact": exact,
        "abs_err": abs_err, "rel_err": rel,
        "const_term": (c - np.log(2.0)) * np.log2(F),
        "mitchell_term": c * ((k - 1.0) - np.log2(k)),
        "prob_scale_err": np.expm1(-abs_err),
    }


# =============================================================================
# Stage 7: bit-exact integer LNU  (formats EXPERIMENTAL, see FixedConfig)
# =============================================================================
#   F_code   UQ acc-format (acc_width, acc_frac), > 0
#   LOD:     (w, k_code) = lod_fixed(F_code, acc_width, acc_frac);
#            k_code = "1.xxx" with acc_width-1 fraction bits
#   kf       = k_code - 2**(acc_width-1)           UQ0.(acc_width-1)   = k - 1
#   kf_r     = kf >> ((acc_width-1) - lnu_frac)    rounding; may carry to 2**lnu_frac
#   log2c    = (w << lnu_frac) + kf_r              signed, lnu_frac frac bits (k-1+w, Eq. 7)
#   prod     = log2c * N     N = exact code of ln2_const (cb frac bits; 0.1011b = 11)
#   ln_f     = prod >> (lnu_frac + cb - a_frac)    rounding; overflow -> a-format SQ
# A carry in kf_r simply propagates into w through the addition (no special case).

@dataclass(frozen=True)
class LnuFixedTrace:
    fx: FixedConfig
    F_code: int
    w: int
    k_code: int            # acc_width-1 fraction bits
    kf_code: int           # (k-1), lnu_frac fraction bits, after rounding
    log2_code: int         # (k-1+w), SQ lnu_frac
    prod_code: int         # exact log2_code * ln2_code, frac = lnu_frac + cb
    ln_f_code: int         # a-format
    overflow: bool         # ln_f overflowed the a-format before handling

    @property
    def F(self): return float(math.ldexp(self.F_code, -self.fx.acc_frac_bits))
    @property
    def k(self): return math.ldexp(self.k_code, -(self.fx.acc_width - 1))
    @property
    def log2_approx(self): return math.ldexp(self.log2_code, -self.fx.lnu_frac_bits)
    @property
    def ln_f(self): return math.ldexp(self.ln_f_code, -self.fx.a_frac_bits)


def lnu_fixed(F_code, cfg: WangConfig) -> LnuFixedTrace:
    """Bit-exact LNU.  F_code == 0 (e.g. produced by a wrapped accumulator or
    by truncation) has no leading one: raises ValueError rather than guessing."""
    fx = cfg.fixed
    if fx is None:
        raise ValueError("cfg.fixed is None: use the floating-point lnu()")
    if not isinstance(F_code, (int, np.integer)) or isinstance(F_code, bool):
        raise ValueError("F_code must be an integer code")
    F_code = int(F_code)
    if F_code == 0:
        raise ValueError("F == 0: LOD undefined (accumulator value is zero)")
    if not (1 <= F_code <= fx.fmt_acc.max_code):
        raise ValueError("F_code outside the accumulator format")
    N = FP.exact_const_code(cfg.ln2_const, fx.const_frac_bits, "ln2_const")
    w, k_code = lod_fixed(F_code, fx.acc_width, fx.acc_frac_bits)
    kf = k_code - (1 << (fx.acc_width - 1))
    kf_r = FP.rshift_round(kf, (fx.acc_width - 1) - fx.lnu_frac_bits, fx.rounding)
    log2_code = (w << fx.lnu_frac_bits) + kf_r
    if not (fx.fmt_log2.min_code <= log2_code <= fx.fmt_log2.max_code):
        raise ValueError("internal: log2 code exceeds fmt_log2 (width formula bug)")
    prod = log2_code * N
    exact = FP.rshift_round(prod, fx.lnu_frac_bits + fx.const_frac_bits - fx.a_frac_bits,
                            fx.rounding)
    ln_f = FP.saturate_or_wrap(exact, fx.fmt_a, fx.overflow)
    return LnuFixedTrace(fx=fx, F_code=F_code, w=w, k_code=k_code, kf_code=kf_r,
                         log2_code=log2_code, prod_code=prod, ln_f_code=int(ln_f),
                         overflow=bool(FP.would_overflow(exact, fx.fmt_a)))
