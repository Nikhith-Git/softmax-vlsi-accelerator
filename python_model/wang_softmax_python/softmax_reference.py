"""Golden softmax reference.

This is the *mathematical* ground truth that the Wang hardware model is
compared against.  It is the only module (besides explicitly labelled
diagnostics) allowed to use numpy.exp.  It knows nothing about the paper's
approximations.

    p_i = exp(x_i) / sum_j exp(x_j)                      (paper Eq. 1)

Numerically stable form (exactly equal in real arithmetic):

    y_i = x_i - max(x)          (so y_i <= 0 and exp(y_i) <= 1: no overflow)
    p_i = exp(y_i) / sum_j exp(y_j)

Why equal: multiplying numerator and denominator of Eq. 1 by exp(-max) leaves
the ratio unchanged.
"""
import numpy as np


def validate_logits(x) -> np.ndarray:
    """Return `x` as a 1-D float64 array, or raise ValueError.

    Rejects: empty input, non-1-D input (scalars, matrices), NaN, +/-inf, and
    anything that cannot be converted to float.  The input is never modified;
    a new float64 array is returned.
    """
    try:
        arr = np.array(x, dtype=np.float64)          # copy, never aliases input
    except (TypeError, ValueError) as exc:
        raise ValueError(f"logits must be numeric: {exc}") from exc
    if arr.ndim != 1:
        raise ValueError(f"logits must be a 1-D array, got {arr.ndim}-D")
    if arr.size == 0:
        raise ValueError("logits must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("logits must be finite (no NaN or +/-inf)")
    return arr


def softmax_reference(x) -> np.ndarray:
    """Stable floating-point softmax of a 1-D logit vector (float64 output)."""
    arr = validate_logits(x)
    y = arr - arr.max()
    e = np.exp(y)
    return e / e.sum()
