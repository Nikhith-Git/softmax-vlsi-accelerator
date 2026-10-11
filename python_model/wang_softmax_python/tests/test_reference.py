"""Golden softmax tests. These must pass independently of the Wang model."""
import numpy as np
import pytest
from softmax_reference import softmax_reference, validate_logits


def test_hand_calculated_values():
    # e^2, e^1, e^0, e^-1 = 7.389056, 2.718282, 1, 0.367879 ; sum = 11.475217
    p = softmax_reference([2.0, 1.0, 0.0, -1.0])
    expected = [0.643914, 0.236883, 0.087144, 0.032059]   # 6-decimal hand values
    np.testing.assert_allclose(p, expected, atol=1e-6)   # tolerance = rounding of the 6-decimal values


def test_sums_to_one_and_positive():
    rng = np.random.default_rng(0)
    for n in (2, 10, 100, 1000):
        p = softmax_reference(rng.uniform(-10, 10, n))
        assert p.min() > 0
        assert p.sum() == pytest.approx(1.0, abs=1e-12)


def test_equal_logits_uniform():
    for c in (0.0, 1.0, -7.5, 1e3):
        np.testing.assert_allclose(softmax_reference([c] * 5), 0.2, rtol=1e-14)


def test_maximum_subtraction_prevents_overflow():
    # naive exp(1000) = inf -> nan; the stable form must be finite and correct
    with np.errstate(over="ignore", invalid="ignore"):
        naive = np.exp([1000.0, 1000.0]) / np.exp([1000.0, 1000.0]).sum()
    assert np.isnan(naive).all()
    np.testing.assert_allclose(softmax_reference([1000.0, 1000.0]), [0.5, 0.5])
    np.testing.assert_allclose(softmax_reference([1000.0, 0.0]), [1.0, 0.0], atol=1e-300)


def test_matches_naive_formula_on_moderate_inputs():
    x = np.array([0.3, -1.2, 2.5, 0.0])
    naive = np.exp(x) / np.exp(x).sum()
    np.testing.assert_allclose(softmax_reference(x), naive, rtol=1e-14)


def test_translation_invariance():
    # Exact in real arithmetic. In float64, x+c rounds each element (relative
    # error ~1e-16 on |x+c|), so allow 1e-10 for c up to 1e3.
    x = np.array([2.0, 1.0, 0.0, -1.0])
    for c in (-1000.0, -3.3, 5.0, 1000.0):
        np.testing.assert_allclose(softmax_reference(x + c), softmax_reference(x), atol=1e-10)


def test_order_preserved():
    x = np.array([0.1, 3.0, -2.0, 1.5])
    assert np.array_equal(np.argsort(softmax_reference(x)), np.argsort(x))


def test_input_not_modified_and_list_accepted():
    x = np.array([1.0, 2.0, 3.0])
    before = x.copy()
    softmax_reference(x)
    assert np.array_equal(x, before)
    assert softmax_reference([1, 2, 3]).dtype == np.float64   # ints accepted


def test_single_element():
    np.testing.assert_array_equal(softmax_reference([42.0]), [1.0])


@pytest.mark.parametrize("bad", [
    [], [1.0, np.nan], [1.0, np.inf], [-np.inf, 0.0],
    [[1.0, 2.0], [3.0, 4.0]], 3.0, ["a", "b"],
])
def test_invalid_inputs_rejected(bad):
    with pytest.raises(ValueError):
        softmax_reference(bad)
    with pytest.raises(ValueError):
        validate_logits(bad)
