"""Preprocessing: y_i = x_i - x_max   (paper Sec. V-A).

In the paper this step is done in SOFTWARE, outside the hardware.  The
hardware receives y_i (as fixed-point numbers plus N_in).

Why it is mathematically valid (exact, not an approximation):
    exp(x_i) / sum_j exp(x_j)
  = exp(x_i - m) * exp(m) / (sum_j exp(x_j - m) * exp(m))      for any m
  = exp(x_i - m) / sum_j exp(x_j - m)
Choosing m = x_max gives y_i <= 0, hence exp(y_i) in (0, 1].  The paper cites
two benefits: no numerical overflow, and no sign bit needed downstream.

Range after preprocessing:
  * every y_i is in (-inf, 0]
  * the maximum element becomes exactly 0.0 (x_max - x_max is exact in IEEE)
  * at least one y_i == 0, so in exact arithmetic F = sum exp(y_i) >= 1
  * equal logits give all zeros; ties for the maximum all map to 0

This module does NOT quantize (float -> fixed, N_in lives in fixed_point.py)
and does NOT clip or otherwise alter values.  Differences x_i - x_max are
computed in float64; if that subtraction itself overflows (|spread| beyond the
float64 range) we raise instead of silently returning -inf.
"""
from dataclasses import dataclass
import numpy as np
from softmax_reference import validate_logits
import fixed_point as FP


@dataclass(frozen=True)
class PreprocessResult:
    x: np.ndarray       # original logits (float64 copy)
    x_max: float        # maximum logit
    y: np.ndarray       # x - x_max; all <= 0; max element exactly 0

    def summary(self) -> str:
        """Human-readable table of the intermediate values."""
        lines = [f"x_max = {self.x_max!r}", "  i        x_i        y_i = x_i - x_max"]
        for i, (xi, yi) in enumerate(zip(self.x, self.y)):
            lines.append(f"{i:3d} {xi:12.6f} {yi:20.6f}")
        return "\n".join(lines)


def preprocess(x) -> PreprocessResult:
    """Validate logits and subtract the maximum (see module docstring)."""
    arr = validate_logits(x)
    x_max = float(arr.max())
    with np.errstate(over="ignore"):
        y = arr - x_max
    if not np.all(np.isfinite(y)):
        raise ValueError("x - max(x) overflowed float64; logit spread too large")
    return PreprocessResult(x=arr, x_max=x_max, y=y)


# ------------------------- Stage 7: fixed-point input ------------------------
@dataclass(frozen=True)
class PreprocessFixedResult:
    x: np.ndarray        # original logits
    x_max: float
    y: np.ndarray        # float x - x_max (<= 0), before quantization
    t_code: np.ndarray   # int64 codes of t = -y in fx.fmt_t (UQ, data_width bits)
    y_q: np.ndarray      # float value actually seen by the hardware: -t_code*2**-N_in
    overflow: bool       # some t exceeded the format BEFORE saturate/wrap


def preprocess_fixed(x, fx) -> PreprocessFixedResult:
    """Software preprocessing (paper Sec. V-A) + float -> fixed quantization.

    y = x - max is computed in float64 (as the paper does in software); the
    magnitude t = -y >= 0 is quantized to fx.fmt_t with fx.rounding and
    fx.overflow.  Storing t instead of y follows the paper's "no sign bit
    needed" remark (y <= 0).  NOTE: rounding acts on t, so "trunc" moves y
    toward zero (exp(y) is overestimated).  Saturation clips very negative y
    to -t_max (benign: exp ~ 0); "wrap" turns a large |y| into a small one and
    is therefore dangerous -- use it only to study failure behaviour.
    """
    pre = preprocess(x)
    t = -pre.y                                   # exact; +0.0 for the max element
    t = np.where(t == 0.0, 0.0, t)               # normalise -0.0
    exact_codes = [FP.quantize(v, FP.QFormat(62, fx.input_frac_bits, False),
                               fx.rounding, "saturate") for v in t]
    over = any(c > fx.fmt_t.max_code for c in exact_codes)
    t_code = FP.quantize(t, fx.fmt_t, fx.rounding, fx.overflow)
    y_q = -FP.dequantize(t_code, fx.fmt_t)
    return PreprocessFixedResult(x=pre.x, x_max=pre.x_max, y=pre.y, t_code=t_code,
                                 y_q=np.where(y_q == 0.0, 0.0, y_q), overflow=bool(over))
