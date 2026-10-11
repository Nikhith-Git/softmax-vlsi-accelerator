"""Preprocessing (y = x - max) tests. All properties here are exact in float64
unless a comment says otherwise."""
import numpy as np
import pytest
from preprocessing import preprocess
from softmax_reference import softmax_reference


def test_hand_example():
    r = preprocess([2.0, 1.0, 0.0, -1.0])
    assert r.x_max == 2.0
    np.testing.assert_array_equal(r.y, [0.0, -1.0, -2.0, -3.0])


def test_all_y_nonpositive_and_max_is_zero():
    rng = np.random.default_rng(1)
    for lo, hi in [(-10, 10), (0, 5), (-5, 0), (3, 4), (-100, 100)]:
        r = preprocess(rng.uniform(lo, hi, 64))
        assert (r.y <= 0).all()
        assert r.y.max() == 0.0                    # exact: x_max - x_max
        assert r.y[np.argmax(r.x)] == 0.0


def test_equal_logits_all_zero():
    for c in (0.0, 4.2, -9.0):
        np.testing.assert_array_equal(preprocess([c] * 6).y, np.zeros(6))


def test_negative_only_and_positive_only():
    np.testing.assert_array_equal(preprocess([-3.0, -1.0, -2.0]).y, [-2.0, 0.0, -1.0])
    np.testing.assert_array_equal(preprocess([1.0, 5.0, 3.0]).y, [-4.0, 0.0, -2.0])


def test_ties_for_maximum_all_map_to_zero():
    np.testing.assert_array_equal(preprocess([2.0, 2.0, 0.0]).y, [0.0, 0.0, -2.0])


def test_no_silent_clipping():
    # Large negative values must pass through untouched (no clip to some range).
    r = preprocess([0.0, -1000.0, -123.456])
    np.testing.assert_array_equal(r.y, [0.0, -1000.0, -123.456])


def test_softmax_invariance_under_preprocessing():
    # Exact math identity; float64 rounding only -> tight tolerance.
    x = np.array([0.7, -2.1, 3.3, 3.3, 0.0])
    np.testing.assert_allclose(softmax_reference(preprocess(x).y), softmax_reference(x), rtol=1e-14)


def test_translation_of_input_leaves_y_unchanged():
    # y depends only on differences. x+c rounds elements, so tolerance (not exact).
    x = np.array([2.0, 1.0, 0.0, -1.0])
    np.testing.assert_allclose(preprocess(x + 1000.0).y, preprocess(x).y, atol=1e-12)


def test_input_not_modified_and_x_is_copy():
    x = np.array([1.0, 2.0])
    r = preprocess(x)
    r.x[0] = 99.0
    assert x[0] == 1.0


def test_summary_shows_intermediates():
    s = preprocess([2.0, 1.0]).summary()
    assert "x_max = 2.0" in s and "-1.000000" in s


def test_result_is_frozen():
    r = preprocess([1.0, 2.0])
    with pytest.raises(Exception):
        r.x_max = 0.0


@pytest.mark.parametrize("bad", [[], [np.nan, 1.0], [np.inf, 0.0], [[1.0], [2.0]]])
def test_invalid_inputs_rejected(bad):
    with pytest.raises(ValueError):
        preprocess(bad)


def test_spread_overflow_raises_not_silent():
    with pytest.raises(ValueError):
        preprocess([1e308, -1e308])
