"""LNU tests: LOD/normalisation (Eq. 5, Table I), approximation (Eqs. 7-8),
and error against the library logarithm (diagnostic use only)."""
import inspect
import math
import numpy as np
import pytest
import lnu as L
from config import WangConfig
from lnu import lod, lod_fixed, lnu, lnu_diagnostic, lnu_errors

CFG = WangConfig()
C = CFG.ln2_const            # 0.6875
LN2 = math.log(2.0)


# ---- Table I (paper): width 8, binary point after the 5th bit -> frac_bits = 3
@pytest.mark.parametrize("bits, w, k_bits", [
    ("11011111", 4, "11011111"),     # 11011.111 -> k = 1.1011111
    ("01010101", 3, "10101010"),     # 01010.101 -> k = 1.0101010
    ("00001101", 0, "11010000"),     # 00001.101 -> k = 1.1010000
    ("00000011", -2, "11000000"),    # 00000.011 -> k = 1.1000000
])
def test_table_I_rows(bits, w, k_bits):
    got_w, k_code = lod_fixed(int(bits, 2), width=8, frac_bits=3)
    assert got_w == w
    assert format(k_code, "08b") == k_bits


def test_table_I_values_match_float_lod():
    for bits in ("11011111", "01010101", "00001101", "00000011"):
        code = int(bits, 2)
        w_fx, k_code = lod_fixed(code, 8, 3)
        w_fl, k_fl = lod(code / 8)
        assert w_fx == w_fl and k_code / 128 == k_fl


def test_lod_fixed_rejects_bad_codes():
    for code in (0, -1, 256):
        with pytest.raises(ValueError):
            lod_fixed(code, 8, 3)


# ---- Eq. 5: F = 2**w * k, k in [1,2), unique
def test_lod_hand_values():
    assert lod(27.875) == (4, 1.7421875)
    assert lod(1.0) == (0, 1.0)
    assert lod(2.0) == (1, 1.0)
    assert lod(0.375) == (-2, 1.5)


def test_lod_reconstruction_and_range():
    rng = np.random.default_rng(0)
    for F in np.exp(rng.uniform(-20, 20, 2000)):
        w, k = lod(F)
        assert 1.0 <= k < 2.0
        assert np.ldexp(k, w) == F               # exact reconstruction


def test_lod_power_of_two_boundaries():
    for n in (-5, 0, 1, 7, 20):
        assert lod(2.0 ** n) == (n, 1.0)
        w, k = lod(np.nextafter(2.0 ** n, 0.0))  # just below 2**n
        assert w == n - 1 and k > 2 - 1e-12


# ---- Eqs. 7-8
def test_lnu_hand_values():
    assert lnu(1.0, CFG).ln_f == 0.0
    assert lnu(2.0, CFG).ln_f == C                      # w=1,k=1
    assert lnu(3.0, CFG).ln_f == C * 1.5                # w=1,k=1.5
    t = lnu(27.875, CFG)
    assert (t.w, t.k, t.k_minus_1) == (4, 1.7421875, 0.7421875)
    assert t.log2_approx == 4.7421875 and t.ln_f == C * 4.7421875


def test_log2_approx_is_integer_w_plus_fraction_of_k():
    # "no adder" claim: integer part == w, fractional part == k-1
    for F in (0.3, 1.0, 1.7, 5.5, 100.0, 4000.0):
        t = lnu(F, CFG)
        assert math.floor(t.log2_approx) == t.w
        assert t.log2_approx - t.w == pytest.approx(t.k_minus_1, abs=1e-15)


def test_mitchell_error_range():
    k = np.linspace(1, 2, 100001)[:-1]
    e = (k - 1) - np.log2(k)
    assert (e <= 1e-15).all()                            # always underestimates log2
    assert e.min() == pytest.approx(-0.0861, abs=1e-4)
    assert k[np.argmin(e)] == pytest.approx(1 / LN2, abs=1e-4)   # k = 1.4427
    assert abs(e[0]) < 1e-15                             # exact at k = 1


# ---- error against exact ln F (diagnostic)
def test_exact_error_identity():
    F = np.exp(np.random.default_rng(1).uniform(-5, 12, 3000))
    r = lnu_errors(F, CFG)
    np.testing.assert_allclose(r["abs_err"], r["const_term"] + r["mitchell_term"], atol=1e-12)


def test_error_bound_for_F_ge_1():
    F = np.exp(np.linspace(0, math.log(65536), 20001))
    r = lnu_errors(F, CFG)
    bound = 0.0861 * C + (LN2 - C) * np.log2(F) + 1e-9
    assert (r["abs_err"] <= 1e-12).all()                 # never overestimates for F >= 1
    assert (-r["abs_err"] <= bound).all()


def test_power_of_two_has_only_constant_error():
    for n in range(0, 13):
        r = lnu_errors([2.0 ** n], CFG)
        assert r["abs_err"][0] == pytest.approx((C - LN2) * n, abs=1e-12)
        assert r["mitchell_term"][0] == 0.0


def test_relative_error_nan_near_F_equal_1_and_defined_elsewhere():
    r = lnu_errors([1.0, 2.0, 100.0], CFG)
    assert np.isnan(r["rel_err"][0])
    assert np.isfinite(r["rel_err"][1:]).all()


def test_prob_scale_error_identity():
    r = lnu_errors([3.0, 50.0, 400.0], CFG)
    np.testing.assert_allclose(r["prob_scale_err"], np.exp(-r["abs_err"]) - 1, rtol=1e-14)
    assert (r["prob_scale_err"] > 0).all()               # underestimated lnF -> outputs too big


def test_diagnostic_is_numpy_log():
    assert lnu_diagnostic(10.0) == float(np.log(10.0))


def test_F_below_one_is_handled_with_negative_w():
    t = lnu(0.375, CFG)
    assert t.w == -2 and t.ln_f == C * (-2 + 0.5)


# ---- invalid inputs and model hygiene
@pytest.mark.parametrize("bad", [0.0, -1.0, -1e-300, np.nan, np.inf, -np.inf])
def test_invalid_F_rejected(bad):
    with pytest.raises(ValueError):
        lnu(bad, CFG)
    with pytest.raises(ValueError):
        lod(bad)


def test_non_scalar_F_rejected():
    for bad in ([1.0, 2.0], "a"):
        with pytest.raises(ValueError):
            lnu(bad, CFG)


def test_no_library_log_in_hardware_model():
    for fn in (L.lod, L.lod_fixed, L.lnu):
        src = inspect.getsource(fn)
        assert "np.log" not in src and "math.log" not in src and "np.exp" not in src
