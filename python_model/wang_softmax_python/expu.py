"""EXPU (Fig. 2): the exponential unit, used for BOTH passes.

Datapath (paper Eqs. 3-4, Sec. III-B/C):

    a = y - lnF                  adder #1   (lnF = 0 in pass 1: ASSUMPTION)
    z = a * log2e_const          constant multiplier (paper: 1.0111b = 1.4375)
    u = floor(z)                 integer part      (Eq. 4)
    v = z - u                    fraction, in [0, 1)
    m = v + d                    adder #2: linear fit 2**v ~ v + d
                                 (d = d1 in pass 1, d2 in pass 2: PROVISIONAL)
    out = m * 2**u               shift unit (left if u>0, right if u<=0)

Where the error comes from (exact identity, proved in the tests):

    out / exp(a) = [2**(c*a) / exp(a)]  *  [(v + d) / 2**v]
                     constant error          linear-fit error
    with c = log2e_const.

  * Constant error: 2**(c*a)/exp(a) = exp(a*(c*ln2 - 1)).  With c = 1.4375,
    c*ln2 - 1 = -0.003587, so the relative error GROWS with |a|:
    about +3.7% at a = -10, +7.4% at a = -20.
  * Linear-fit error: depends only on v and d (independent of a).
  * Shift: exact.  floor/split: exact.

Note on continuity (derived, not from the paper): across an integer boundary
of z the left limit is (1+d)/2 * 2**n while the value at z = n is d * 2**n.
  d = 1 : continuous.
  d > 1 : the output jumps UP at each boundary; it is still monotonic in y.
  d < 1 : the output jumps DOWN, so it is NOT monotonic and can reverse the
          order of two nearby inputs (factor 2d/(1+d), ~2% for d = 0.957).

Only floor, subtraction, multiplication, addition and a power-of-two scaling
(the shift unit) are used here; no library exponential or logarithm.
Fixed-point version (Stage 7): see the second half of this file (expu_fixed).
"""
from dataclasses import dataclass
import numpy as np
from config import WangConfig, FixedConfig
import fixed_point as FP

_MAX_ABS_Z = 2 ** 30      # model range guard so the shift amount fits an int32


@dataclass(frozen=True)
class ExpuTrace:
    y: np.ndarray        # EXPU input
    ln_f: float          # lnF subtracted (0 in pass 1 -- assumption)
    d: float             # bias used in this pass
    a: np.ndarray        # y - lnF
    z: np.ndarray        # a * log2e_const
    u: np.ndarray        # floor(z), int64
    v: np.ndarray        # z - u, in [0, 1)
    mantissa: np.ndarray # v + d
    out: np.ndarray      # mantissa * 2**u


def _finite_1d(y) -> np.ndarray:
    arr = np.atleast_1d(np.array(y, dtype=np.float64))
    if arr.ndim != 1:
        raise ValueError("EXPU input must be a scalar or 1-D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("EXPU input must be finite")
    return arr


def _finite_scalar(val, name: str) -> float:
    f = float(val)
    if not np.isfinite(f):
        raise ValueError(f"{name} must be finite")
    return f


def split_integer_fraction(z: np.ndarray):
    """Eq. 4 decomposition with the signed floor: u = floor(z), v = z - u.

    Exact in float64.  v is always in [0, 1) (the paper writes (0, 1], which
    floor() cannot produce).  Negative z gives negative u and positive v,
    e.g. z = -1.4375 -> u = -2, v = 0.5625.
    """
    u = np.floor(z)
    v = z - u
    return u.astype(np.int64), v


def shift_unit(mantissa: np.ndarray, u: np.ndarray) -> np.ndarray:
    """mantissa * 2**u: left shift for u > 0, right shift for u <= 0 (Eq. 4).
    Exact in floating point; underflows to 0.0 for very negative u."""
    return np.ldexp(mantissa, u.astype(np.int32))


def expu(y, ln_f: float, d: float, cfg: WangConfig) -> ExpuTrace:
    """One EXPU pass (see module docstring)."""
    y = _finite_1d(y)
    ln_f = _finite_scalar(ln_f, "ln_f")
    d = _finite_scalar(d, "d")
    a = y - ln_f
    z = a * cfg.log2e_const
    if np.any(np.abs(z) >= _MAX_ABS_Z):
        raise ValueError("|y - lnF| * log2e is outside the model range")
    u, v = split_integer_fraction(z)
    mantissa = v + d
    out = shift_unit(mantissa, u)
    return ExpuTrace(y=y, ln_f=ln_f, d=d, a=a, z=z, u=u, v=v, mantissa=mantissa, out=out)


def expu_pass1(y, cfg: WangConfig) -> ExpuTrace:
    """First pass: bias d1, lnF input = cfg.pass1_ln_f (assumed 0)."""
    return expu(y, cfg.pass1_ln_f, cfg.d1, cfg)


def expu_pass2(y, ln_f: float, cfg: WangConfig) -> ExpuTrace:
    """Second pass: bias d2, input y - lnF.  Output is the softmax probability."""
    return expu(y, ln_f, cfg.d2, cfg)


# ------------------------------ diagnostics ---------------------------------
def expu_exact_diagnostic(y, ln_f: float = 0.0) -> np.ndarray:
    """DIAGNOSTIC ONLY (not hardware): exact exp(y - ln_f) via numpy."""
    return np.exp(_finite_1d(y) - _finite_scalar(ln_f, "ln_f"))


def expu_isolated_errors(y, cfg: WangConfig, ln_f: float = 0.0, d: float = None) -> dict:
    """DIAGNOSTIC ONLY: split the EXPU relative error into its two sources.

    Returns arrays (same shape as y):
      const_rel_err : 2**(c*a)/exp(a) - 1       (from log2e ~ 1.4375)
      fit_rel_err   : (v + d)/2**v - 1          (from the linear fit)
      total_rel_err : out/exp(a) - 1
    and total satisfies (1+total) = (1+const)*(1+fit) exactly.
    """
    d = cfg.d1 if d is None else d
    tr = expu(y, ln_f, d, cfg)
    exact = np.exp(tr.a)
    const = np.exp2(tr.z) / exact - 1.0
    fit = tr.mantissa / np.exp2(tr.v) - 1.0
    total = tr.out / exact - 1.0
    return {"const_rel_err": const, "fit_rel_err": fit, "total_rel_err": total, "trace": tr}


# =============================================================================
# Stage 7: bit-exact integer EXPU  (all formats EXPERIMENTAL, see FixedConfig)
# =============================================================================
#   y_code   SQ a-format   input (y, or y - lnF's y); pass 1 y <= 0
#   a        = y_code - ln_f_code                 a-format, overflow policy
#   z_exact  = a * L        L = round-free code of log2e_const (cb frac bits)
#                           frac = a_frac + cb, exact integer product
#   z        = z_exact >> (a_frac + cb - z_frac)  rounding; then overflow
#   u        = z >> z_frac     (arithmetic = floor)       signed integer
#   v        = z & (2**z_frac - 1)                        UQ0.z_frac, [0, 1)
#   d_code   = quantize(d)      UQ2.z_frac, 0 <= d < 2    (d1 or d2, separate)
#   m        = v + d_code                                 UQ2.z_frac (no overflow)
#   out      = m * 2**u  =  m shifted by (u + out_frac - z_frac) with rounding
#              (left if > 0, right if < 0), then overflow into UQ.out_frac
# Only shifts, adds, one constant multiply and masks: directly mappable to RTL.

@dataclass(frozen=True)
class ExpuFixedTrace:
    fx: FixedConfig
    y_code: np.ndarray        # int64, a-format
    ln_f_code: int            # a-format (0 in pass 1: ASSUMPTION, cfg.pass1_ln_f)
    d_code: int               # bias code, UQ2.z_frac  (d1 or d2)
    a_code: np.ndarray
    z_code: np.ndarray
    u: np.ndarray
    v_code: np.ndarray
    m_code: np.ndarray
    out_code: np.ndarray      # UQ.out_frac
    overflow: dict            # stage -> True if any value overflowed BEFORE handling

    # float views (exact: code * 2**-frac)
    @property
    def y(self): return FP.dequantize(self.y_code, self.fx.fmt_a)
    @property
    def ln_f(self): return FP.dequantize(self.ln_f_code, self.fx.fmt_a)
    @property
    def d(self): return FP.dequantize(self.d_code, self.fx.fmt_m)
    @property
    def a(self): return FP.dequantize(self.a_code, self.fx.fmt_a)
    @property
    def z(self): return FP.dequantize(self.z_code, self.fx.fmt_z)
    @property
    def v(self): return FP.dequantize(self.v_code, self.fx.fmt_v)
    @property
    def mantissa(self): return FP.dequantize(self.m_code, self.fx.fmt_m)
    @property
    def out(self): return FP.dequantize(self.out_code, self.fx.fmt_exp_out)


def _bias_code(d: float, fx: FixedConfig) -> int:
    d = _finite_scalar(d, "d")
    if not (0.0 <= d < 2.0):
        raise ValueError("fixed-point EXPU needs 0 <= d < 2 (mantissa has 2 integer bits)")
    return FP.quantize(d, fx.fmt_m, fx.rounding, "saturate")


def _shift_into_out(m: int, sh: int, fx: FixedConfig):
    """m * 2**sh into the EXPU output format.  Returns (code, overflowed)."""
    fmt = fx.fmt_exp_out
    if sh > fmt.width + 1:                      # avoid building 2**huge integers
        if m == 0:
            return 0, False
        return (fmt.max_code if fx.overflow == "saturate" else 0), True
    exact = FP.rshift_round(m, -sh, fx.rounding)
    return FP.saturate_or_wrap(exact, fmt, fx.overflow), bool(FP.would_overflow(exact, fmt))


def expu_fixed(y_code, ln_f_code: int, d: float, cfg: WangConfig) -> ExpuFixedTrace:
    """One bit-exact EXPU pass.  y_code: integer codes in the a-format."""
    fx = cfg.fixed
    if fx is None:
        raise ValueError("cfg.fixed is None: use the floating-point expu()")
    y_code = np.atleast_1d(np.asarray(y_code))
    if y_code.ndim != 1 or y_code.dtype.kind not in "iu":
        raise ValueError("y_code must be a 1-D integer array")
    if np.any(FP.would_overflow(y_code, fx.fmt_a)):
        raise ValueError("y_code outside the a-format range")
    ln_f_code = int(ln_f_code)
    if not (fx.fmt_a.min_code <= ln_f_code <= fx.fmt_a.max_code):
        raise ValueError("ln_f_code outside the a-format range")
    L = FP.exact_const_code(cfg.log2e_const, fx.const_frac_bits, "log2e_const")
    d_code = _bias_code(d, fx)
    zf, shift_z = fx.z_frac_bits, fx.a_frac_bits + fx.const_frac_bits - fx.z_frac_bits

    a_exact = y_code.astype(np.int64) - ln_f_code
    a_code = FP.saturate_or_wrap(a_exact, fx.fmt_a, fx.overflow)
    z_exact = [int(a) * L for a in a_code]                       # exact product
    z_rounded = [FP.rshift_round(z, shift_z, fx.rounding) for z in z_exact]
    z_code = np.array([FP.saturate_or_wrap(z, fx.fmt_z, fx.overflow) for z in z_rounded],
                      dtype=np.int64)
    u = z_code >> zf
    v_code = z_code & ((1 << zf) - 1)
    m_code = v_code + d_code
    outs = [_shift_into_out(int(m), int(uu) + fx.exp_out_frac_bits - zf, fx)
            for m, uu in zip(m_code, u)]
    out_code = np.array([o for o, _ in outs], dtype=np.int64)
    overflow = {
        "a": bool(np.any(FP.would_overflow(a_exact, fx.fmt_a))),
        "z": any(not fx.fmt_z.min_code <= z <= fx.fmt_z.max_code for z in z_rounded),
        "out": any(f for _, f in outs),
    }
    return ExpuFixedTrace(fx=fx, y_code=y_code.astype(np.int64), ln_f_code=ln_f_code,
                          d_code=d_code, a_code=a_code, z_code=z_code, u=u,
                          v_code=v_code, m_code=m_code, out_code=out_code, overflow=overflow)


def expu_fixed_pass1(y_code, cfg: WangConfig) -> ExpuFixedTrace:
    """First pass: bias d1; lnF input = quantized cfg.pass1_ln_f (ASSUMED 0)."""
    fx = cfg.fixed
    ln0 = FP.quantize(cfg.pass1_ln_f, fx.fmt_a, fx.rounding, fx.overflow)
    return expu_fixed(y_code, ln0, cfg.d1, cfg)


def expu_fixed_pass2(y_code, ln_f_code: int, cfg: WangConfig) -> ExpuFixedTrace:
    """Second pass: bias d2 (separate from d1); input y - lnF."""
    return expu_fixed(y_code, ln_f_code, cfg.d2, cfg)
