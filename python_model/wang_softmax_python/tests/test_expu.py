"""EXPU tests. Exact properties use assert_array_equal; approximation
properties state their expected size and why."""
import inspect
import numpy as np
import pytest
import coefficients as C
import expu as E
from config import WangConfig
from expu import (expu, expu_pass1, expu_pass2, expu_exact_diagnostic,
                  expu_isolated_errors, split_integer_fraction)

CFG1 = WangConfig(d1=1.0, d2=0.9)       # distinct, easy-to-hand-check biases
LN2 = np.log(2.0)


# ---- hand-checkable values (1.4375 = 23/16 is exact in binary floating point)
def test_zero_input_gives_d():
    t = expu(0.0, 0.0, 1.0, CFG1)
    assert t.z[0] == 0 and t.u[0] == 0 and t.v[0] == 0
    assert t.out[0] == 1.0                      # (0 + d) * 2**0


def test_negative_input_hand_values():
    t = expu(-1.0, 0.0, 1.0, CFG1)              # z = -1.4375
    assert t.z[0] == -1.4375
    assert t.u[0] == -2 and t.v[0] == 0.5625    # floor, positive fraction
    assert t.mantissa[0] == 1.5625
    assert t.out[0] == 1.5625 / 4               # right shift by 2


def test_positive_input_uses_left_shift():
    t = expu(1.0, 0.0, 1.0, CFG1)               # z = 1.4375 -> u=1, v=0.4375
    assert t.u[0] == 1 and t.v[0] == 0.4375
    assert t.out[0] == (0.4375 + 1.0) * 2


# ---- decomposition (exact)
def test_decomposition_invariants():
    rng = np.random.default_rng(0)
    z = rng.uniform(-50, 50, 10000)
    u, v = split_integer_fraction(z)
    assert (v >= 0).all() and (v < 1).all()
    np.testing.assert_array_equal(u + v, z)     # exact in float64
    assert u.dtype == np.int64


def test_integer_boundary_split():
    # a = -16 -> z = -23 exactly (16*1.4375). At the boundary v = 0.
    t = expu(-16.0, 0.0, 1.0, CFG1)
    assert t.z[0] == -23.0 and t.u[0] == -23 and t.v[0] == 0.0
    below = np.nextafter(-16.0, -np.inf)        # z just below -23
    above = np.nextafter(-16.0, 0.0)            # z just above -23
    tb, ta = expu(below, 0.0, 1.0, CFG1), expu(above, 0.0, 1.0, CFG1)
    assert tb.u[0] == -24 and tb.v[0] > 1 - 1e-12
    assert ta.u[0] == -23 and 0 <= ta.v[0] < 1e-12


def test_boundary_jump_unless_d_is_one():
    # Derived: left limit (1+d)/2 * 2**n, value at boundary d * 2**n.
    below, at = np.nextafter(-16.0, -np.inf), -16.0
    for d, jumps in [(1.0, False), (0.95, True)]:
        o = expu([below, at], 0.0, d, CFG1).out
        if jumps:
            assert o[1] < o[0] * 0.99               # output DROPS as y rises
        else:
            assert o[1] == pytest.approx(o[0], rel=1e-12)   # continuous


def test_monotonic_only_when_d_is_one():
    y = np.linspace(-12, 0, 20001)
    d1 = expu(y, 0.0, 1.0, CFG1).out
    assert (np.diff(d1) >= -1e-15).all()
    dm = expu(y, 0.0, C.D_MINIMAX.value, CFG1).out
    assert (np.diff(dm) < 0).any()                  # documented non-monotonicity


def test_d_above_one_is_monotonic_with_upward_jumps():
    y = np.linspace(-12, 0, 20001)
    o = expu(y, 0.0, 1.06, CFG1).out
    assert (np.diff(o) >= -1e-15).all()
    below, at = np.nextafter(-16.0, -np.inf), -16.0
    ob = expu([below, at], 0.0, 1.06, CFG1).out
    assert ob[1] > ob[0] * 1.01                        # jumps UP at the boundary


# ---- pass behaviour
def test_pass1_and_pass2_use_separate_biases():
    cfg = WangConfig(d1=1.0, d2=0.9)
    y = np.array([0.0, -1.0, -2.0])
    p1 = expu_pass1(y, cfg)
    p2 = expu_pass2(y, 0.0, cfg)
    assert p1.d == 1.0 and p2.d == 0.9
    np.testing.assert_array_equal(p1.v, p2.v)       # same split, different bias
    np.testing.assert_array_equal(p2.mantissa, p2.v + 0.9)


def test_pass2_subtracts_ln_f_before_multiplying():
    cfg = WangConfig(d1=1.0, d2=0.9)
    y = np.array([0.0, -1.0])
    t = expu_pass2(y, 2.0, cfg)
    np.testing.assert_array_equal(t.a, [-2.0, -3.0])
    np.testing.assert_array_equal(t.z, np.array([-2.0, -3.0]) * 1.4375)
    assert t.ln_f == 2.0


def test_pass1_lnf_is_configurable_assumption():
    cfg = WangConfig(d1=1.0, pass1_ln_f=0.5)
    assert expu_pass1([0.0], cfg).a[0] == -0.5


# ---- approximation error: identity and sizes
def test_error_decomposition_identity():
    rng = np.random.default_rng(1)
    y = rng.uniform(-15, 0, 5000)
    for d in (1.0, 0.957):
        r = expu_isolated_errors(y, WangConfig(), d=d)
        lhs = 1 + r["total_rel_err"]
        rhs = (1 + r["const_rel_err"]) * (1 + r["fit_rel_err"])
        np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


def test_constant_error_closed_form_and_growth():
    c = 1.4375
    y = np.array([-1.0, -10.0, -20.0])
    r = expu_isolated_errors(y, WangConfig())
    np.testing.assert_allclose(r["const_rel_err"], np.expm1(y * (c * LN2 - 1)), rtol=1e-12)
    assert r["const_rel_err"][1] == pytest.approx(0.0365, abs=5e-4)   # ~+3.7% at -10
    assert r["const_rel_err"][2] > r["const_rel_err"][1] > r["const_rel_err"][0] > 0


def test_fit_error_depends_only_on_v():
    cfg = WangConfig()
    r1 = expu_isolated_errors([-3.0], cfg, d=1.0)
    r2 = expu_isolated_errors([-3.0 - 16.0], cfg, d=1.0)    # z shifts by exactly -23
    assert r1["fit_rel_err"][0] == pytest.approx(r2["fit_rel_err"][0], rel=1e-12)


def test_fit_error_sizes_for_d1_and_minimax():
    v = np.arange(1 << 16) / (1 << 16)
    p = 2.0 ** v
    assert (v + 1.0 >= p).all()                            # chord lies above 2**v
    # max of (v+1)/2**v is at v = 1/ln2 - 1 = 0.4427 -> 1.4427/2**0.4427 = 1.0615
    assert np.max((v + 1.0) / p - 1) == pytest.approx(0.0615, abs=5e-4)
    assert np.max(np.abs(v + C.D_MINIMAX.value - p)) == pytest.approx(0.0430, abs=5e-4)


# ---- model hygiene
def test_no_library_exp_or_log_in_hardware_model():
    for fn in (E.expu, E.shift_unit, E.split_integer_fraction,
               E.expu_pass1, E.expu_pass2):
        src = inspect.getsource(fn)
        assert "np.exp" not in src and "np.log" not in src and "math." not in src


def test_underflow_to_zero_and_range_guard():
    t = expu(-1000.0, 0.0, 1.0, WangConfig())
    assert t.out[0] == 0.0
    with pytest.raises(ValueError):
        expu(-1e12, 0.0, 1.0, WangConfig())


@pytest.mark.parametrize("bad", [[np.nan], [np.inf], [[1.0, 2.0]], ["x"]])
def test_invalid_inputs(bad):
    with pytest.raises(ValueError):
        expu(bad, 0.0, 1.0, WangConfig())


@pytest.mark.parametrize("ln_f,d", [(np.nan, 1.0), (0.0, np.inf)])
def test_invalid_scalars(ln_f, d):
    with pytest.raises(ValueError):
        expu([0.0], ln_f, d, WangConfig())


# ---- calibration (PROVISIONAL, mantissa fit only)
def test_calibration_objectives():
    ma = C.calibrate_d("max_abs")
    assert ma.value == pytest.approx(C.D_MINIMAX.value, abs=1e-5)
    assert ma.provenance == C.Provenance.PROVISIONAL
    mr = C.calibrate_d("max_rel")
    v = np.arange(1 << 16) / (1 << 16)
    grid = np.linspace(0.9, 1.0, 2001)
    best = min(np.max(np.abs((v + d) / 2.0 ** v - 1)) for d in grid)
    got = np.max(np.abs((v + mr.value) / 2.0 ** v - 1))
    assert got <= best + 1e-6
    assert got < 0.0615                                     # beats d = 1
    mz = C.calibrate_d("mean_rel_zero")
    assert np.mean((v + mz.value) / 2.0 ** v - 1) == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        C.calibrate_d("nope")
