"""Stage 7: fixed-point primitives (QFormat, rounding, overflow, bit patterns)."""
import math
from fractions import Fraction
import numpy as np
import pytest
import fixed_point as FP
from fixed_point import QFormat

S8_2 = QFormat(8, 2, True)        # lsb 0.25, codes -128..127
S8_4 = QFormat(8, 4, True)
U8_4 = QFormat(8, 4, False)


# ----------------------------- QFormat ---------------------------------------
def test_qformat_ranges_and_names():
    assert (S8_4.min_code, S8_4.max_code, S8_4.int_bits, S8_4.name) == (-128, 127, 3, "SQ3.4")
    assert (U8_4.min_code, U8_4.max_code, U8_4.int_bits, U8_4.name) == (0, 255, 4, "UQ4.4")
    assert U8_4.max_value == 255 / 16 and S8_4.min_value == -8.0 and S8_4.lsb == 0.0625
    assert QFormat(8, 8).name == "UQ0.8"


@pytest.mark.parametrize("args", [(0, 0, False), (1, 0, True), (8, 9, False), (8, 8, True),
                                  (8, -1, False), (63, 4, False), (8.0, 4, False)])
def test_qformat_rejects_invalid(args):
    with pytest.raises(ValueError):
        QFormat(*args)


# ----------------------- negative values -------------------------------------
def test_quantize_negative_values_exact_and_roundtrip():
    assert FP.quantize(-1.5, S8_4) == -24
    assert FP.quantize(-0.0625, S8_4) == -1
    assert FP.quantize(-8.0, S8_4) == -128            # exactly the minimum: no overflow
    assert FP.dequantize(-24, S8_4) == -1.5
    assert list(FP.quantize(np.array([-1.0, 0.0, 1.0]), S8_4)) == [-16, 0, 16]
    for c in range(S8_4.min_code, S8_4.max_code + 1):  # every code round-trips exactly
        assert FP.quantize(FP.dequantize(c, S8_4), S8_4) == c


def test_negative_zero_and_unsigned_negative():
    assert FP.quantize(-0.0, S8_4) == 0
    assert FP.quantize(-1.0, U8_4) == 0               # unsigned cannot hold negatives: saturates
    assert FP.quantize(-1.0, U8_4, overflow="wrap") == 240   # -16 mod 256


# ----------------------- rounding boundaries ---------------------------------
# value -> (trunc, toward_zero, round_half_up, round_half_even) at lsb = 0.25
HALF_CASES = {
    0.125:  (0, 0, 1, 0),      #  +0.5 lsb
    0.375:  (1, 1, 2, 2),      #  +1.5 lsb
    -0.125: (-1, 0, 0, 0),     #  -0.5 lsb
    -0.375: (-2, -1, -1, -2),  #  -1.5 lsb
    0.25:   (1, 1, 1, 1),      # exactly representable: all modes agree
    -0.25:  (-1, -1, -1, -1),
}
MODES = ("trunc", "toward_zero", "round_half_up", "round_half_even")


@pytest.mark.parametrize("x,expected", list(HALF_CASES.items()))
def test_rounding_at_exact_half_lsb(x, expected):
    for mode, want in zip(MODES, expected):
        assert FP.quantize(x, S8_2, rounding=mode) == want, (x, mode)
    assert FP.quantize(x, S8_2, rounding="floor") == expected[0]    # alias


def test_rounding_just_either_side_of_half():
    eps = 2.0 ** -20
    for mode in ("round_half_up", "round_half_even"):
        assert FP.quantize(0.125 - eps, S8_2, rounding=mode) == 0
        assert FP.quantize(0.125 + eps, S8_2, rounding=mode) == 1
        assert FP.quantize(-0.125 - eps, S8_2, rounding=mode) == -1
        assert FP.quantize(-0.125 + eps, S8_2, rounding=mode) == 0
    assert FP.quantize(0.25 - eps, S8_2, rounding="trunc") == 0     # floor below an LSB edge
    assert FP.quantize(-eps, S8_2, rounding="trunc") == -1          # floor goes AWAY from 0
    assert FP.quantize(-eps, S8_2, rounding="toward_zero") == 0


@pytest.mark.parametrize("mode", MODES)
def test_rshift_round_matches_exact_fraction_arithmetic(mode):
    ref = {"trunc": math.floor, "toward_zero": lambda f: int(f),
           "round_half_up": lambda f: math.floor(f + Fraction(1, 2)),
           "round_half_even": lambda f: round(f)}[mode]     # Python round(Fraction) is half-even
    for n in (1, 2, 3, 5):
        for x in range(-70, 71):
            assert FP.rshift_round(x, n, mode) == ref(Fraction(x, 2 ** n)), (x, n, mode)


def test_rshift_identity_left_shift_and_array_form():
    assert FP.rshift_round(5, 0) == 5
    assert FP.rshift_round(-3, -4) == -48                 # n < 0: exact left shift
    out = FP.rshift_round(np.array([-5, -4, 4, 5]), 1, "round_half_even")
    assert out.tolist() == [-2, -2, 2, 2] and out.dtype == np.int64
    assert FP.rshift_round(7, 10**9, "round_half_up") == 0      # huge shift: no blow-up
    assert FP.rshift_round(-7, 10**9, "trunc") == -1


def test_quantize_does_not_double_round_float_inputs():
    # 0.1 is not dyadic; quantize works on its exact binary value
    assert FP.quantize(0.1, QFormat(16, 8, True)) == 25        # floor(25.6)
    assert FP.quantize(0.1, QFormat(16, 8, True), "round_half_up") == 26


# ----------------------- overflow: saturation and wraparound -----------------
def test_saturation_boundaries_signed_and_unsigned():
    assert FP.saturate_or_wrap(127, S8_4, "saturate") == 127
    assert FP.saturate_or_wrap(128, S8_4, "saturate") == 127
    assert FP.saturate_or_wrap(-128, S8_4, "saturate") == -128
    assert FP.saturate_or_wrap(-129, S8_4, "saturate") == -128
    assert FP.saturate_or_wrap(255, U8_4, "saturate") == 255
    assert FP.saturate_or_wrap(256, U8_4, "saturate") == 255
    assert FP.saturate_or_wrap(-1, U8_4, "saturate") == 0
    assert FP.saturate_or_wrap(10 ** 30, S8_4, "saturate") == 127
    assert FP.quantize(100.0, S8_4) == 127                      # float path saturates
    assert FP.quantize(-100.0, S8_4) == -128


def test_wraparound_boundaries_signed_and_unsigned():
    w = lambda c, f: FP.saturate_or_wrap(c, f, "wrap")
    assert (w(127, S8_4), w(128, S8_4), w(255, S8_4), w(256, S8_4)) == (127, -128, -1, 0)
    assert (w(-128, S8_4), w(-129, S8_4), w(-256, S8_4)) == (-128, 127, 0)
    assert (w(255, U8_4), w(256, U8_4), w(257, U8_4), w(-1, U8_4)) == (255, 0, 1, 255)
    assert w(10 ** 30 + 5, U8_4) == (10 ** 30 + 5) % 256
    assert FP.quantize(9.0, S8_4, overflow="wrap") == FP.quantize(-7.0, S8_4)   # 9 -> -7 (mod 16)


def test_wrap_equals_modular_arithmetic_exhaustively():
    for fmt in (QFormat(6, 2, True), QFormat(6, 2, False)):
        span = 1 << fmt.width
        for c in range(-3 * span, 3 * span):
            got = FP.saturate_or_wrap(c, fmt, "wrap")
            assert fmt.min_code <= got <= fmt.max_code
            assert (got - c) % span == 0
            sat = FP.saturate_or_wrap(c, fmt, "saturate")
            assert sat == min(max(c, fmt.min_code), fmt.max_code)


def test_would_overflow_and_array_forms():
    assert not FP.would_overflow(127, S8_4) and FP.would_overflow(128, S8_4)
    assert FP.would_overflow(np.array([-129, 0, 128]), S8_4).tolist() == [True, False, True]
    arr = np.array([-300, -128, 0, 127, 300])
    assert FP.saturate_or_wrap(arr, S8_4, "saturate").tolist() == [-128, -128, 0, 127, 127]
    assert FP.saturate_or_wrap(arr, S8_4, "wrap").tolist() == [-44, -128, 0, 127, 44]


def test_requantize_shifts_then_fits():
    assert FP.requantize(100, 4, QFormat(8, 2, False)) == 25            # 100/4
    assert FP.requantize(7, 4, QFormat(8, 2, False), "round_half_up") == 2    # 1.75 -> 2
    assert FP.requantize(3, 2, QFormat(8, 6, False)) == 48              # exact left shift
    assert FP.requantize(1000, 0, QFormat(8, 0, True), overflow="saturate") == 127
    assert FP.requantize(1000, 0, QFormat(8, 0, True), overflow="wrap") == -24


# ----------------------- error handling / bit patterns -----------------------
def test_rejects_bad_modes_and_values():
    with pytest.raises(ValueError):
        FP.quantize(1.0, S8_4, rounding="nearest")
    with pytest.raises(ValueError):
        FP.quantize(1.0, S8_4, overflow="clip")
    with pytest.raises(ValueError):
        FP.saturate_or_wrap(1, S8_4, "clip")
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            FP.quantize(bad, S8_4)
    with pytest.raises(ValueError):
        FP.dequantize(128, S8_4)                      # out-of-range code is a bug upstream
    with pytest.raises(TypeError):
        FP.saturate_or_wrap(1.5, S8_4)                # codes must be integers
    with pytest.raises(TypeError):
        FP.rshift_round(np.array([1.0]), 1)


def test_bit_patterns_roundtrip():
    assert FP.bit_string(-1, S8_4) == "11111111"
    assert FP.bit_string(-128, S8_4) == "10000000"
    assert FP.bit_string(5, S8_4) == "00000101"
    assert FP.to_bits(-1, S8_4) == 255 and FP.from_bits(255, S8_4) == -1
    for c in range(-128, 128):
        assert FP.from_bits(FP.to_bits(c, S8_4), S8_4) == c
    with pytest.raises(ValueError):
        FP.to_bits(200, S8_4)
    with pytest.raises(ValueError):
        FP.from_bits(256, S8_4)


def test_exact_const_code_never_rounds_paper_constants():
    assert FP.exact_const_code(1.4375, 4, "log2e") == 23        # 1.0111b
    assert FP.exact_const_code(0.6875, 4, "ln2") == 11          # 0.1011b
    assert FP.exact_const_code(1.4375, 6) == 92
    with pytest.raises(ValueError):
        FP.exact_const_code(1.4375, 3)
    with pytest.raises(ValueError):
        FP.exact_const_code(1.442695, 4)                         # exact log2(e): not representable
