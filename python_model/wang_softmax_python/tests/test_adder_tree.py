"""Adder tree and accumulator tests."""
import math
import numpy as np
import pytest
from adder_tree import numpy_sum, balanced_tree_sum, accumulate_chunks
from config import WangConfig, FixedConfig


@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 15, 16, 17, 100, 1000, 4096])
def test_tree_matches_numpy_sum_various_lengths(n):
    rng = np.random.default_rng(n)
    x = rng.uniform(0, 1, n)
    # Different addition ORDER -> rounding differences only; n*eps bound is generous.
    assert balanced_tree_sum(x) == pytest.approx(numpy_sum(x), rel=1e-13)
    assert accumulate_chunks(x, 16) == pytest.approx(numpy_sum(x), rel=1e-13)


def test_exact_when_all_partial_sums_representable():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 1000, 777).astype(float)   # integer sums are exact in float64
    assert balanced_tree_sum(x) == numpy_sum(x) == float(x.sum())
    assert accumulate_chunks(x, 16) == float(x.sum())


def test_different_magnitudes():
    x = np.array([1e-12, 1.0, 1e6, 1e-6, 3.0])
    assert balanced_tree_sum(x) == pytest.approx(numpy_sum(x), rel=1e-14)


def test_tree_depth_and_level_sizes():
    for n in (1, 2, 5, 16, 17, 100):
        _, levels = balanced_tree_sum(np.ones(n), return_levels=True)
        assert len(levels) - 1 == math.ceil(math.log2(n)) if n > 1 else len(levels) == 1
        assert levels[0].size == n and levels[-1].size == 1
        assert levels[-1][0] == n


def test_odd_length_passes_leftover_up():
    _, levels = balanced_tree_sum([1.0, 2.0, 4.0], return_levels=True)
    np.testing.assert_array_equal(levels[1], [3.0, 4.0])
    np.testing.assert_array_equal(levels[2], [7.0])


def test_accumulator_cycles_and_running_sum():
    x = np.ones(37)
    t = accumulate_chunks(x, lanes=16, return_trace=True)
    assert t["cycles"] == 3                                  # 16 + 16 + 5 (padded)
    np.testing.assert_array_equal(t["chunk_sums"], [16, 16, 5])
    np.testing.assert_array_equal(t["running"], [16, 32, 37])
    assert t["F"] == 37


def test_lane_count_does_not_change_result_beyond_rounding():
    x = np.random.default_rng(3).uniform(0, 1, 500)
    ref = numpy_sum(x)
    for lanes in (1, 4, 8, 16, 64):
        assert accumulate_chunks(x, lanes) == pytest.approx(ref, rel=1e-13)


def test_float_accumulator_refuses_fixed_cfg():
    with pytest.raises(ValueError):          # Stage 7: use accumulate_chunks_fixed
        accumulate_chunks([1.0], 16, WangConfig(fixed=FixedConfig()))


@pytest.mark.parametrize("bad", [[], [np.nan], [np.inf], [[1.0, 2.0]]])
def test_invalid_values(bad):
    for fn in (numpy_sum, balanced_tree_sum, accumulate_chunks):
        with pytest.raises(ValueError):
            fn(bad)


@pytest.mark.parametrize("lanes", [0, -1, 2.5])
def test_invalid_lanes(lanes):
    with pytest.raises(ValueError):
        accumulate_chunks([1.0], lanes)
