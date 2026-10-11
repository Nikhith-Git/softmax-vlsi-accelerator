"""Integer fixed-point primitives (Stage 7).

The paper fixes NO internal formats (only that 8- and 16-bit input/output are
supported, Sec. V-A).  Every format built on these primitives is therefore a
configurable EXPERIMENTAL choice; see config.FixedConfig.

Representation
--------------
A value is stored as an INTEGER CODE `c` in a QFormat(width, frac_bits, signed):

    value = c * 2**-frac_bits
    unsigned: c in [0, 2**width - 1]
    signed  : c in [-2**(width-1), 2**(width-1) - 1]   (two's complement,
              `width` INCLUDES the sign bit)

Naming: UQi.f = unsigned, i integer bits, f fraction bits (width = i + f);
        SQi.f = signed,   i integer bits EXCLUDING sign, f fraction bits
                (width = 1 + i + f).

All arithmetic is done on Python integers (arbitrary precision), so a
primitive never overflows silently: overflow is handled ONLY by
`saturate_or_wrap`, selected explicitly.  Every function accepts a scalar int
or a numpy integer array (arrays are returned as int64; widths <= 62).

Rounding modes (applied whenever fraction bits are discarded)
-------------------------------------------------------------
  "trunc" / "floor" : discard bits = arithmetic right shift = round toward
                      -infinity.  For unsigned values this is toward zero;
                      for negative signed values it rounds AWAY from zero.
                      Cheapest in hardware (just drop wires).
  "toward_zero"     : round toward zero (sign-magnitude style truncation).
  "round_half_up"   : round to nearest, ties toward +infinity.  Hardware:
                      add 1 at the first discarded bit, then drop.
  "round_half_even" : round to nearest, ties to even (unbiased).

Overflow modes
--------------
  "saturate" : clamp to [min_code, max_code].
  "wrap"     : modulo 2**width (two's complement for signed): keep the low
               `width` bits.  Can flip the sign / change magnitude wildly.
"""
from dataclasses import dataclass
import math
import numpy as np

MAX_WIDTH = 62          # codes are returned in int64 arrays
ROUNDING_MODES = ("trunc", "floor", "toward_zero", "round_half_up", "round_half_even")
OVERFLOW_MODES = ("saturate", "wrap")


def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


@dataclass(frozen=True)
class QFormat:
    width: int
    frac_bits: int
    signed: bool = False

    def __post_init__(self):
        if not (_is_int(self.width) and _is_int(self.frac_bits)):
            raise ValueError("width and frac_bits must be integers")
        if not isinstance(self.signed, (bool, np.bool_)):
            raise ValueError("signed must be a bool")
        s = 1 if self.signed else 0
        if not (1 + s <= self.width <= MAX_WIDTH):
            raise ValueError(f"width must be in [{1 + s}, {MAX_WIDTH}], got {self.width}")
        if not (0 <= self.frac_bits <= self.width - s):
            raise ValueError(f"frac_bits must be in [0, {self.width - s}], got {self.frac_bits}")

    @property
    def int_bits(self) -> int:
        """Integer bits, EXCLUDING the sign bit."""
        return self.width - self.frac_bits - (1 if self.signed else 0)

    @property
    def lsb(self) -> float:
        return math.ldexp(1.0, -self.frac_bits)

    @property
    def min_code(self) -> int:
        return -(1 << (self.width - 1)) if self.signed else 0

    @property
    def max_code(self) -> int:
        return (1 << (self.width - 1)) - 1 if self.signed else (1 << self.width) - 1

    @property
    def min_value(self) -> float:
        return math.ldexp(self.min_code, -self.frac_bits)

    @property
    def max_value(self) -> float:
        return math.ldexp(self.max_code, -self.frac_bits)

    @property
    def name(self) -> str:
        return f"{'S' if self.signed else 'U'}Q{self.int_bits}.{self.frac_bits}"


# ------------------------------ helpers -------------------------------------
def _canon_rounding(rounding: str) -> str:
    if rounding not in ROUNDING_MODES:
        raise ValueError(f"unknown rounding {rounding!r}; use one of {ROUNDING_MODES}")
    return "trunc" if rounding == "floor" else rounding


def _check_overflow_mode(overflow: str) -> None:
    if overflow not in OVERFLOW_MODES:
        raise ValueError(f"unknown overflow {overflow!r}; use one of {OVERFLOW_MODES}")


def _elementwise(fn, code, *args):
    """Apply a scalar-int function to a scalar or every element of an array."""
    if np.ndim(code) == 0:
        if not _is_int(code):
            raise TypeError(f"integer code expected, got {type(code).__name__}")
        return fn(int(code), *args)
    arr = np.asarray(code)
    if arr.dtype.kind not in "iu":
        raise TypeError("integer codes expected, got dtype " + str(arr.dtype))
    out = [fn(int(c), *args) for c in arr.ravel()]
    return np.array(out, dtype=np.int64).reshape(arr.shape)


def _rshift_round_scalar(x: int, n: int, rounding: str) -> int:
    if n < 0:
        return x << (-n)                       # exact left shift
    if n == 0:
        return x
    # Beyond |x|.bit_length()+2 the result no longer depends on n (|x| < half
    # an LSB of the result); clamping avoids building 2**huge integers.
    n = min(n, abs(x).bit_length() + 2)
    if rounding == "trunc":
        return x >> n
    if rounding == "toward_zero":
        return -((-x) >> n) if x < 0 else x >> n
    if rounding == "round_half_up":
        return (x + (1 << (n - 1))) >> n
    # round_half_even
    q = x >> n
    r = x - (q << n)                           # 0 <= r < 2**n
    half = 1 << (n - 1)
    if r > half or (r == half and (q & 1)):
        q += 1
    return q


def rshift_round(code, n: int, rounding: str = "trunc"):
    """Arithmetic right shift by n with the given rounding (n<0: exact left
    shift, n=0: identity).  NO overflow handling (result is exact-width)."""
    if not _is_int(n):
        raise ValueError("shift amount must be an integer")
    r = _canon_rounding(rounding)
    return _elementwise(_rshift_round_scalar, code, n, r)


# ------------------------------ overflow ------------------------------------
def would_overflow(code, fmt: QFormat):
    """True where `code` is outside the format's range (before any handling)."""
    if np.ndim(code) == 0:
        return not (fmt.min_code <= int(code) <= fmt.max_code)
    arr = np.asarray(code)
    return (arr < fmt.min_code) | (arr > fmt.max_code)


def _fit_scalar(c: int, fmt: QFormat, overflow: str) -> int:
    if overflow == "saturate":
        return max(fmt.min_code, min(fmt.max_code, c))
    mod = 1 << fmt.width                       # wrap: keep the low `width` bits
    c &= mod - 1
    if fmt.signed and c >= (1 << (fmt.width - 1)):
        c -= mod
    return c


def saturate_or_wrap(code, fmt: QFormat, overflow: str = "saturate"):
    """Bring an (exact-width) integer code into `fmt` by saturation or wrap."""
    _check_overflow_mode(overflow)
    return _elementwise(_fit_scalar, code, fmt, overflow)


# ------------------------- float <-> code ------------------------------------
def _quantize_scalar(x, fmt: QFormat, rounding: str, overflow: str) -> int:
    if _is_int(x):
        n, den = int(x), 1
    else:
        xf = float(x)
        if not math.isfinite(xf):
            raise ValueError("cannot quantize NaN or infinity")
        n, den = xf.as_integer_ratio()         # exact; den is a power of two
    k = den.bit_length() - 1
    scaled = (n << (fmt.frac_bits - k)) if fmt.frac_bits >= k \
        else _rshift_round_scalar(n, k - fmt.frac_bits, rounding)
    return _fit_scalar(scaled, fmt, overflow)


def quantize(x, fmt: QFormat, rounding: str = "trunc", overflow: str = "saturate"):
    """float (or int) -> integer code.  Exact: x is a dyadic rational, so the
    only error is the chosen rounding (when frac bits are dropped) and the
    chosen overflow handling.  Raises on NaN / inf."""
    r = _canon_rounding(rounding)
    _check_overflow_mode(overflow)
    if np.ndim(x) == 0:
        return _quantize_scalar(x, fmt, r, overflow)
    arr = np.asarray(x)
    out = [_quantize_scalar(v.item(), fmt, r, overflow) for v in arr.ravel()]
    return np.array(out, dtype=np.int64).reshape(arr.shape)


def dequantize(code, fmt: QFormat):
    """integer code -> float (exact for width <= 53).  Rejects out-of-range
    codes (they indicate a missing saturate_or_wrap)."""
    if np.any(would_overflow(code, fmt)):
        raise ValueError(f"code outside {fmt.name} range [{fmt.min_code}, {fmt.max_code}]")
    if np.ndim(code) == 0:
        return math.ldexp(float(int(code)), -fmt.frac_bits)
    return np.ldexp(np.asarray(code, dtype=np.float64), -fmt.frac_bits)


def requantize(code, src_frac_bits: int, dst: QFormat,
               rounding: str = "trunc", overflow: str = "saturate"):
    """Change the binary point of an exact code (frac src_frac_bits) to `dst`:
    shift with rounding, then saturate/wrap into dst."""
    _check_overflow_mode(overflow)
    shifted = rshift_round(code, src_frac_bits - dst.frac_bits, rounding)
    return saturate_or_wrap(shifted, dst, overflow)


def exact_const_code(value, frac_bits: int, name: str = "constant") -> int:
    """Integer code of a constant that MUST be exactly representable with
    `frac_bits` fraction bits (e.g. the paper's 1.0111b, 0.1011b).  Raises
    instead of rounding, so a paper constant is never altered silently."""
    n, den = float(value).as_integer_ratio()
    k = den.bit_length() - 1
    if k > frac_bits:
        raise ValueError(f"{name}={value!r} needs {k} fraction bits but only "
                         f"{frac_bits} are configured; refusing to round it")
    return n << (frac_bits - k)


# ------------------------- bit patterns (for RTL) -----------------------------
def to_bits(code, fmt: QFormat):
    """Unsigned bit pattern (two's complement for signed) of an in-range code."""
    if np.any(would_overflow(code, fmt)):
        raise ValueError("code out of range for format")
    mask = (1 << fmt.width) - 1
    if np.ndim(code) == 0:
        return int(code) & mask
    return np.asarray(code, dtype=np.int64) & mask


def from_bits(bits, fmt: QFormat):
    """Inverse of to_bits."""
    def f(b):
        if not 0 <= b < (1 << fmt.width):
            raise ValueError("bit pattern wider than format")
        return b - (1 << fmt.width) if fmt.signed and b >= (1 << (fmt.width - 1)) else b
    return _elementwise(f, bits)


def bit_string(code, fmt: QFormat) -> str:
    """Binary string of a scalar code, MSB first (handy for Verilog vectors)."""
    return format(to_bits(int(code), fmt), f"0{fmt.width}b")
