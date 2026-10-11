"""Every numeric constant used by the Wang softmax model, with provenance.

Provenance tiers (never mix them silently):
  PAPER       - stated in the Wang et al. paper (section/figure cited)
  DERIVED     - follows mathematically from paper facts or pure math
  PROVISIONAL - NOT in the paper; our own experimental choice (e.g. d1, d2)
"""
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np


class Provenance(str, Enum):
    PAPER = "paper"
    DERIVED = "derived"
    PROVISIONAL = "provisional"


@dataclass(frozen=True)
class Coefficient:
    name: str
    value: float
    provenance: Provenance
    source: str
    note: str = ""


def from_binary(s: str) -> float:
    """Convert a binary string like '1.0111' to float (exact)."""
    whole, _, frac = s.partition(".")
    val = int(whole or "0", 2)
    for i, bit in enumerate(frac, start=1):
        val += int(bit) * 2.0 ** -i
    return val


# ---- Paper constants -------------------------------------------------------
LOG2E_ARCH = Coefficient(
    "log2e_arch", from_binary("1.0111"), Provenance.PAPER,
    "Sec. III-C: log2(e) ~ 1.0111b = 1 + 0.1b - 0.0001b",
    "= 1.4375; true 1.442695 (-0.36%)")
LN2_ARCH = Coefficient(
    "ln2_arch", from_binary("0.1011"), Provenance.PAPER,
    "Sec. IV-B: ln2 ~ 0.1011b",
    "= 0.6875; true 0.693147 (-0.82%)")

# ---- Exact values: golden reference / diagnostics ONLY ---------------------
LOG2E_EXACT = Coefficient("log2e_exact", math.log2(math.e), Provenance.DERIVED,
                          "pure math", "diagnostic use only")
LN2_EXACT = Coefficient("ln2_exact", math.log(2.0), Provenance.DERIVED,
                        "pure math", "diagnostic use only")

# ---- Candidate bias values for 2**v ~ v + d on [0,1) ------------------------
# Derived (not from paper): chord through endpoints gives d = 1;
# minimax offset for slope-1 fit equalises error at v=0 and at the interior
# maximum of (2**v - v), located at v* = log2(1/ln2).
_v_star = math.log2(1.0 / math.log(2.0))
_interior = 2.0 ** _v_star - _v_star          # = 0.9139...
D_CHORD = Coefficient("d_chord", 1.0, Provenance.DERIVED,
                      "2**v - v equals 1 at v=0 (and v->1)")
D_MINIMAX = Coefficient("d_minimax", (1.0 + _interior) / 2.0, Provenance.DERIVED,
                        "equalised-error offset for slope-1 fit",
                        "max |error| ~ 0.043")

# ---- Provisional d1 / d2 (paper gives NO values) ---------------------------
# Stage 6 SYSTEM-LEVEL grid calibration (calibrate_d1_d2.py), NOT from the paper.
# Objective: mean RMS absolute error of the whole pipeline vs the golden softmax.
# Calibration set: uniform logits in [-r, r], r in {0.1, 1, 5, 10}, N = 8/64/512,
# 40 vectors each, seed 1.  Held-out check (seed 2, N = 16/128/2048): RMS 0.00114
# against 0.00271 for d1 = d2 = 1.0.  Because these values mostly compensate the
# LNU/log2e constant biases, they are NOT a fit of 2**v alone (see D_MINIMAX).
# Kept as two separate entries: the paper defines two distinct biases.
D1 = Coefficient("d1", 1.16, Provenance.PROVISIONAL,
                 "Stage 6 grid calibration (calibrate_d1_d2.py); NOT in paper",
                 "pass-1 bias; compensates first-pass/LNU bias through F")
D2 = Coefficient("d2", 1.0825, Provenance.PROVISIONAL,
                 "Stage 6 grid calibration (calibrate_d1_d2.py); NOT in paper",
                 "pass-2 bias; >= 1 keeps the EXPU output monotonic in y")


def _v_grid(n: int = 1 << 16) -> np.ndarray:
    """Dense grid over v in [0, 1) (the EXPU fractional part is never 1)."""
    return np.arange(n) / n


def calibrate_d(objective: str = "max_abs", n: int = 1 << 16) -> Coefficient:
    """Calibrate ONE bias d of the linear fit 2**v ~ v + d on v in [0,1).

    PROVISIONAL / NOT FROM THE PAPER.  The paper gives no d1 or d2; this
    estimates a d for the mantissa fit alone, ignoring the rest of the
    pipeline.  Joint (d1, d2) tuning against the full softmax needs the
    complete pipeline and is done in Stage 6/8.

    objective:
      "max_abs"       minimise max |(v+d) - 2**v|          (minimax absolute)
      "max_rel"       minimise max |(v+d)/2**v - 1|        (minimax relative)
      "mean_rel_zero" choose d so mean((v+d)/2**v - 1) = 0 (unbiased on average)
    Returns a PROVISIONAL Coefficient; the achieved error is in `.note`.
    """
    v = _v_grid(n)
    p = np.exp2(v)                                  # diagnostic exact 2**v
    if objective == "mean_rel_zero":
        d = (1.0 - np.mean(v / p)) / np.mean(1.0 / p)
    elif objective in ("max_abs", "max_rel"):
        if objective == "max_abs":
            def cost(d): return np.max(np.abs(v + d - p))
        else:
            def cost(d): return np.max(np.abs((v + d) / p - 1.0))
        lo, hi = 0.5, 1.5                           # cost is convex in d
        for _ in range(200):                        # ternary search
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if cost(m1) < cost(m2):
                hi = m2
            else:
                lo = m1
        d = 0.5 * (lo + hi)
    else:
        raise ValueError(f"unknown objective {objective!r}")
    rel = (v + d) / p - 1.0
    note = (f"objective={objective}; max|abs err|={np.max(np.abs(v + d - p)):.5f}; "
            f"max|rel err|={np.max(np.abs(rel)):.5f}; mean rel err={np.mean(rel):+.5f}")
    return Coefficient(f"d_calibrated_{objective}", float(d), Provenance.PROVISIONAL,
                       "Stage 4 calibration of the mantissa fit only (NOT in paper)", note)
