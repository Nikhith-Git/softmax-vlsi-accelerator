"""End-to-end tests of the Wang softmax model."""
import inspect
import math
import numpy as np
import pytest
import coefficients as C
import wang_softmax as W
import calibrate_d1_d2 as cal
from config import WangConfig
from wang_softmax import WangSoftmax, wang_softmax
from softmax_reference import softmax_reference

HAND = [[2.0, 1.0, 0.0, -1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
        [10.0, 9.0, 8.0], [-2.0, -1.0, 0.0]]


def plain_pipeline(x, d1, d2, c=1.4375, ln2c=0.6875):
    """Independent re-implementation with plain Python floats (math.floor,
    math.frexp, math.ldexp) -- shares no code with the modules under test."""
    m = max(x)
    y = [xi - m for xi in x]

    def expu(a, d):
        z = a * c
        u = math.floor(z)
        return math.ldexp((z - u) + d, u)

    e1 = [expu(yi, d1) for yi in y]
    F = sum(e1)
    mant, e = math.frexp(F)
    w, k = e - 1, 2 * mant
    ln_f = ln2c * ((k - 1) + w)
    return [expu(yi - ln_f, d2) for yi in y], F, ln_f


@pytest.mark.parametrize("d1,d2", [(1.0, 1.0), (1.16, 1.0825), (0.95, 0.9)])
def test_matches_independent_plain_python_pipeline(d1, d2):
    rng = np.random.default_rng(0)
    vectors = HAND + [[3.0], [0.5, -7.25], list(rng.uniform(-10, 10, 37))]
    for x in vectors:
        p, F, ln_f = plain_pipeline(list(map(float, x)), d1, d2)
        tr = WangSoftmax(WangConfig(d1=d1, d2=d2)).run(x, debug=True)
        np.testing.assert_allclose(tr.p_raw, p, rtol=1e-12)
        assert tr.F == pytest.approx(F, rel=1e-13)
        assert tr.lnu.ln_f == pytest.approx(ln_f, rel=1e-12, abs=1e-15)


def test_hand_example_pinned_values_with_d_equal_one():
    tr = WangSoftmax(WangConfig(d1=1.0, d2=1.0)).run([2.0, 1.0, 0.0, -1.0], debug=True)
    np.testing.assert_array_equal(tr.y, [0.0, -1.0, -2.0, -3.0])
    np.testing.assert_array_equal(tr.pass1.out, [1.0, 0.390625, 0.140625, 0.052734375])
    assert tr.F == 1.583984375                      # exact sum
    assert (tr.lnu.w, tr.lnu.k) == (0, 1.583984375)
    assert tr.lnu.ln_f == 0.6875 * 0.583984375


def test_trace_is_consistent_and_complete():
    x = np.random.default_rng(1).uniform(-4, 4, 50)
    tr = WangSoftmax().run(x, debug=True)
    np.testing.assert_array_equal(tr.y, x - x.max())
    assert tr.F == pytest.approx(tr.pass1.out.sum(), rel=1e-13)
    assert tr.pass1.ln_f == 0.0 and tr.pass2.ln_f == tr.lnu.ln_f
    np.testing.assert_array_equal(tr.pass2.a, tr.y - tr.lnu.ln_f)
    assert tr.pass1.d == WangConfig().d1 and tr.pass2.d == WangConfig().d2
    for arr in (tr.pass1.z, tr.pass1.u, tr.pass1.v, tr.pass2.z, tr.p_raw, tr.p):
        assert arr.shape == x.shape
    assert not WangSoftmax().run(x, debug=False) is tr


def test_no_second_normalisation_by_default():
    x = np.array([2.0, 1.0, 0.0, -1.0])
    tr = WangSoftmax().run(x, debug=True)
    np.testing.assert_array_equal(tr.p, tr.p_raw)
    assert abs(tr.p_raw.sum() - 1.0) > 1e-6          # approximate model: sum != 1


def test_optional_renormalisation_is_separate():
    x = np.array([2.0, 1.0, 0.0, -1.0])
    tr = WangSoftmax(WangConfig(renormalize_output=True)).run(x, debug=True)
    assert tr.p.sum() == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(tr.p / tr.p_raw, tr.p[0] / tr.p_raw[0], rtol=1e-12)


def test_equal_logits_symmetry_exact():
    for c in (0.0, 1.0, -3.5, 100.0):
        p = wang_softmax([c] * 7)
        assert (p == p[0]).all() and p[0] > 0


def test_translation_invariance_exact_for_dyadic_inputs():
    # x and x + c are exactly representable and subtract to identical y -> bit-exact.
    x = np.array([2.0, 1.0, 0.0, -1.0, 0.25])
    for c in (-64.0, 8.0, 1024.0):
        np.testing.assert_array_equal(wang_softmax(x + c), wang_softmax(x))


def test_translation_invariance_only_approximate_for_general_inputs():
    # x + c rounds each element (~1e-13), so y shifts slightly; the EXPU output
    # moves by about that much unless a value straddles an integer z-boundary
    # (not hit for this seed), where it would jump by up to a few percent.
    x = np.random.default_rng(5).uniform(-5, 5, 40)
    np.testing.assert_allclose(wang_softmax(x + 1000.0), wang_softmax(x), atol=1e-9)


def test_order_preserved_when_d2_ge_one_and_not_when_below_one():
    x = np.linspace(-12.0, 0.0, 20001)            # ascending logits
    for d2 in (1.0, 1.0825):
        p = wang_softmax(x, WangConfig(d1=1.1, d2=d2))
        assert (np.diff(p) >= -1e-15).all()
    p = wang_softmax(x, WangConfig(d1=1.1, d2=0.95))
    assert (np.diff(p) < 0).any()                  # order reversals exist


def test_argmax_agreement_on_hand_inputs():
    for x in HAND:
        assert wang_softmax(x).argmax() == softmax_reference(x).argmax()


def test_single_element_runs_and_is_positive():
    p = wang_softmax([5.0])
    assert p.shape == (1,) and p[0] > 0


def test_large_vector_multi_cycle():
    x = np.random.default_rng(2).uniform(-10, 10, 4096)
    tr = WangSoftmax().run(x, debug=True)
    assert np.isfinite(tr.p).all() and (tr.p > 0).all()
    assert tr.F == pytest.approx(tr.pass1.out.sum(), rel=1e-12)


@pytest.mark.parametrize("bad", [[], [np.nan, 1.0], [np.inf], [[1.0, 2.0]]])
def test_invalid_inputs_rejected(bad):
    with pytest.raises(ValueError):
        wang_softmax(bad)


def test_pipeline_module_has_no_library_exp_or_log():
    src = inspect.getsource(W)
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("#", '"""')))
    assert "np.exp" not in code and "np.log" not in code and "math." not in code


# ---- calibration study -------------------------------------------------------
def test_calibration_fast_path_matches_real_pipeline():
    vecs = cal.make_workload(7, (8, 40), per=2)
    d1s, d2s = [1.0, 1.16], [0.95, 1.0825]
    r = cal.evaluate_grid(vecs, d1s, d2s)
    for i, d1 in enumerate(d1s):
        for j, d2 in enumerate(d2s):
            model = WangSoftmax(WangConfig(d1=d1, d2=d2))
            rms = np.mean([np.sqrt(np.mean((model.run(x) - softmax_reference(x)) ** 2))
                           for x in vecs])
            assert r["mean_rms_err"][i, j] == pytest.approx(rms, rel=1e-12)


def test_default_biases_are_provisional_and_beat_d_equal_one_on_fresh_data():
    assert C.D1.provenance == C.Provenance.PROVISIONAL
    assert C.D2.provenance == C.Provenance.PROVISIONAL
    assert WangConfig().d1 == C.D1.value and WangConfig().d2 == C.D2.value
    rows = cal.compare({"one": (1.0, 1.0), "cal": (C.D1.value, C.D2.value)}, 3, (16, 100), per=10)
    # Regression on a fresh synthetic set -- not a general accuracy claim.
    assert rows["cal"]["mean_rms_err"] < rows["one"]["mean_rms_err"]
