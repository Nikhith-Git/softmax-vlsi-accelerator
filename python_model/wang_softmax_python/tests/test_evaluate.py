"""Stage 8: evaluation metrics, experiments, checks, reproducibility and output files."""
import copy
import functools
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

import evaluate as E
from config import WangConfig, FixedConfig
from softmax_reference import softmax_reference

ROOT = Path(E.__file__).resolve().parent


@functools.lru_cache(maxsize=1)
def quick():
    """One shared --quick evaluation (seed 2026) written to a temp directory."""
    d = Path(tempfile.mkdtemp(prefix="wang_eval_"))
    return d, E.run_evaluation(d, seed=2026, quick=True, make_plots=True, verbose=False)


# ============================ compute_metrics =================================
def test_compute_metrics_hand_values():
    ref, hw = [0.5, 0.3, 0.2], [0.4, 0.35, 0.25]
    m = E.compute_metrics(ref, hw)
    np.testing.assert_allclose(m["abs_err"], [0.1, 0.05, 0.05], atol=1e-15)
    assert m["max_abs"] == pytest.approx(0.1, abs=1e-15)
    assert m["mean_abs"] == pytest.approx(0.2 / 3, abs=1e-15)
    assert m["rmse"] == pytest.approx(math.sqrt(0.005), rel=1e-12)       # (.01+.0025+.0025)/3
    assert m["sum_hw"] == pytest.approx(1.0, abs=1e-15) and m["sum_ref"] == pytest.approx(1.0)
    np.testing.assert_allclose(m["rel_err"], [-0.2, 0.05 / 0.3, 0.25], rtol=1e-12)
    assert m["n_rel"] == 3 and m["max_rel"] == pytest.approx(0.25, rel=1e-12)
    assert m["argmax_match"] and not m["argmax_hw_tied"]


def test_relative_error_only_where_meaningful():
    m = E.compute_metrics([0.995, 0.005], [0.99, 0.01])           # second ref < REL_FLOOR (0.01)
    assert m["n_rel"] == 1 and np.isnan(m["rel_err"][1])
    assert m["rel_err"][0] == pytest.approx(-0.005 / 0.995, rel=1e-12)
    assert m["max_rel"] == pytest.approx(0.005 / 0.995, rel=1e-12)
    tiny = E.compute_metrics([0.001, 0.0005], [0.002, 0.0])
    assert tiny["n_rel"] == 0 and np.isnan(tiny["max_rel"]) and np.isnan(tiny["mean_rel"])
    edge = E.compute_metrics([0.01, 0.99], [0.02, 0.99])           # exactly at the floor: included
    assert edge["n_rel"] == 2
    assert E.compute_metrics([0.5, 0.5], [0.6, 0.6], rel_floor=0.9)["n_rel"] == 0


def test_argmax_agreement_ties_and_inversions():
    ties = E.compute_metrics([1 / 3] * 3, [0.34] * 3)              # both tied: index 0 is a valid max
    assert ties["argmax_match"] and ties["argmax_hw_tied"]
    hw_tie_ok = E.compute_metrics([0.2, 0.6, 0.2], [0.1, 0.45, 0.45])   # hw tie includes true max
    assert hw_tie_ok["argmax_match"] and hw_tie_ok["argmax_hw_tied"]
    wrong_first = E.compute_metrics([0.2, 0.6, 0.2], [0.45, 0.45, 0.1])  # first index 0 is wrong...
    assert not wrong_first["argmax_match"] and wrong_first["argmax_ref_in_hw_top"]  # ...but a tie
    inversion = E.compute_metrics([0.2, 0.6, 0.2], [0.6, 0.2, 0.2])
    assert not inversion["argmax_match"] and not inversion["argmax_ref_in_hw_top"]
    assert E.compute_metrics([0.1, 0.9], [0.0, 0.5])["n_zero_hw"] == 1


@pytest.mark.parametrize("ref,hw", [([], []), ([0.5, 0.5], [1.0]), ([0.5, np.nan], [0.5, 0.5]),
                                    ([[0.5, 0.5]], [[0.5, 0.5]]), ([0.5, 0.5], [np.inf, 0.0])])
def test_compute_metrics_rejects_bad_input(ref, hw):
    with pytest.raises(ValueError):
        E.compute_metrics(ref, hw)


def test_percentiles_linear_interpolation_and_empty():
    p = E.percentiles(np.arange(101.0))
    assert p == {"p50": 50.0, "p90": 90.0, "p95": 95.0, "p99": 99.0, "p100": 100.0}
    assert E.percentiles([1.0, 2.0], (50,))["p50"] == 1.5
    assert all(np.isnan(v) for v in E.percentiles([]).values())


def test_order_violations():
    assert E.order_violations([0, 1, 2], [0.1, 0.2, 0.3]) == 0
    assert E.order_violations([0, 1, 2], [0.1, 0.1, 0.3]) == 0         # ties are allowed
    assert E.order_violations([0, 1, 2], [0.1, 0.3, 0.2]) == 1
    assert E.order_violations([2, 0, 1], [0.3, 0.1, 0.2]) == 0         # unsorted input
    assert E.order_violations([1, 1], [0.2, 0.3]) == 1                 # equal logits, different p


def test_md_table_escapes_pipes():
    t = E.md_table(["a|b"], [["x|y"]])
    assert "a\\|b" in t and "x\\|y" in t


# ======================= required inputs and boundary cases ===================
def test_required_inputs_present_exactly_as_specified():
    assert list(E.REQUIRED_CASES.values()) == [[2.0, 1.0, 0.0, -1.0], [1.0, 1.0, 1.0],
                                               [0.0, 0.0, 0.0], [10.0, 9.0, 8.0], [-2.0, -1.0, 0.0]]


def test_exact_reference_values_for_required_inputs():
    e = np.exp
    np.testing.assert_allclose(softmax_reference([2.0, 1.0, 0.0, -1.0]),
                               np.array([e(2), e(1), 1, e(-1)]) / (e(2) + e(1) + 1 + e(-1)), rtol=1e-14)
    np.testing.assert_allclose(softmax_reference([1.0, 1.0, 1.0]), [1 / 3] * 3, rtol=1e-14)
    np.testing.assert_allclose(softmax_reference([10.0, 9.0, 8.0]),
                               softmax_reference([2.0, 1.0, 0.0]), rtol=1e-14)


def test_hand_derived_float_and_fixed_values_for_zeros():
    """[0,0,0] derived by hand: y = 0, F = 3*d1, k = F/2, lnF = 0.6875*(k-1+1), ..."""
    cfg = WangConfig()
    F = 3 * cfg.d1                                   # 3.48 -> w = 1, k = 1.74
    ln_f = 0.6875 * ((F / 2 - 1) + 1)
    z = -ln_f * 1.4375
    u = math.floor(z)
    want = ((z - u) + cfg.d2) * 2.0 ** u
    p, _ = E.run_all([0.0, 0.0, 0.0])
    np.testing.assert_allclose(p["float"], [want] * 3, rtol=1e-12)
    assert want == pytest.approx(0.340723, abs=1e-6)
    # fixed 8-bit, by hand: d1 code 296, F = 888, w = 1, (k-1) = 188/256, log2 = 444, prod = 4884,
    # lnF code 305, z = floor(-305*23/16) = -439, u = -2, v = 73, d2 code 277 -> m = 350, out = 350 >> 2
    assert p["fixed8"].tolist() == [87 / 256] * 3
    np.testing.assert_allclose(p["exact"], [1 / 3] * 3, rtol=1e-14)


def test_all_required_and_boundary_cases_are_evaluated_and_consistent():
    d, res = quick()
    named = res["named"]
    assert set(E.REQUIRED_CASES) <= set(named) and set(E.BOUNDARY_CASES) <= set(named)
    for name, rec in named.items():
        r = rec["result"]
        n = len(rec["x"])
        for k in ("exact", "float", "fixed8", "fixed16"):
            assert r["p"][k].shape == (n,)
        assert r["p"]["exact"].sum() == pytest.approx(1.0, abs=1e-12)
        for key, m in r["metrics"].items():
            assert m["abs_err"].shape == (n,)
            assert m["max_abs"] == m["abs_err"].max()
            assert m["mean_abs"] == pytest.approx(m["abs_err"].mean(), rel=1e-12)
            assert m["rmse"] >= m["mean_abs"] - 1e-15            # RMSE >= mean abs (Jensen)
            assert m["max_abs"] >= m["rmse"] - 1e-15


def test_required_inputs_errors_are_small_and_argmax_agrees():
    # Regression bound 0.01: measured worst required-input error is 0.0074 (float, [1,1,1]);
    # the analysis (EXPU linear-fit + constants + LNU) predicts errors of this size, not larger.
    d, res = quick()
    for name in E.REQUIRED_CASES:
        r = res["named"][name]["result"]
        for impl in ("float", "fixed8", "fixed16"):
            m = r["metrics"][f"{impl}_vs_exact"]
            assert m["max_abs"] < 0.01, (name, impl)
            assert m["argmax_match"], (name, impl)
        assert r["env_violations"] == 0


def test_equal_inputs_give_equal_outputs_and_same_values_for_1_and_0():
    d, res = quick()
    a, b = (res["named"][n]["result"]["p"] for n in ("required_1_1_1", "required_0_0_0"))
    for k in a:
        assert np.all(a[k] == a[k][0])
        np.testing.assert_array_equal(a[k], b[k])


def test_8bit_resolution_cliff_is_visible_in_equal_logit_sweep():
    rows = {r["N"]: r for r in E.equal_logit_sweep([1, 16, 512, 1024])}
    assert rows[1]["p_exact"] == 1.0
    assert rows[512]["p_fixed8"] == 0.0 and rows[1024]["p_fixed8"] == 0.0 and rows[1024]["sum_fixed8"] == 0.0
    assert rows[16]["p_fixed8"] > 0.0
    assert rows[1024]["p_fixed16"] > 0.0           # 16-bit preset still resolves 1/1024
    assert rows[1024]["p_float"] == pytest.approx(1 / 1024, rel=0.1)


def test_single_element_error_is_the_documented_design_error():
    p, _ = E.run_all([3.0])
    assert p["exact"][0] == 1.0
    assert p["float"][0] == pytest.approx(0.96219, abs=1e-5)      # hand: (0.8419 + 1.0825)/2


# ============================ envelope / identity =============================
def test_envelope_contains_known_results_and_is_not_vacuous():
    lo, hi = E.wang_relative_error_envelope(1, 0.0)
    p, _ = E.run_all([3.0])
    assert lo < p["float"][0] - 1 < hi
    assert hi - lo < 0.5
    lo2, hi2 = E.wang_relative_error_envelope(1024, 30.0)
    assert lo2 < lo and hi2 > hi                         # wider with more elements / spread
    with pytest.raises(ValueError):
        E.wang_relative_error_envelope(4, 3.0, WangConfig(d1=0.9))
    with pytest.raises(ValueError):
        E.wang_relative_error_envelope(4, 3.0, WangConfig(pass1_ln_f=0.5))


def test_error_identity_holds_to_rounding():
    rng = np.random.default_rng(5)
    for x in ([2.0, 1.0, 0.0, -1.0], [0.0] * 7, list(rng.uniform(-8, 8, 40)), [0.0, -12.0]):
        d = E.decompose_wang(x)
        assert d["identity_resid"] < 1e-12
        np.testing.assert_allclose(d["ratio_actual"], d["ratio_pred"], rtol=1e-12)


# ============================== random tests ==================================
def test_random_grid_is_reproducible_and_seed_dependent():
    kw = dict(lengths=(4, 16), n_per_length={4: 3, 16: 3}, ranges=E.RANGES[:2])
    a = E.run_random(11, **kw)
    b = E.run_random(11, **kw)
    c = E.run_random(12, **kw)
    for k in a:
        for ra, rb in zip(a[k]["recs"], b[k]["recs"]):
            np.testing.assert_array_equal(ra["x"], rb["x"])
            for impl in ra["p"]:
                np.testing.assert_array_equal(ra["p"][impl], rb["p"][impl])
    assert not np.array_equal(a[(4, "u0.5")]["recs"][0]["x"], c[(4, "u0.5")]["recs"][0]["x"])


def test_cells_do_not_depend_on_run_order_or_other_cells():
    full = E.run_random(3, (4, 16), {4: 2, 16: 2}, E.RANGES[:3])
    only = E.run_random(3, (16,), {16: 2}, E.RANGES[:3])
    for lab in ("u0.5", "u2", "u5"):
        for ra, rb in zip(full[(16, lab)]["recs"], only[(16, lab)]["recs"]):
            np.testing.assert_array_equal(ra["x"], rb["x"])


def test_random_vectors_respect_declared_ranges_and_counts():
    cells = E.run_random(1, (8,), {8: 20}, E.RANGES)
    assert sum(c["n"] for c in cells.values()) == 5 * 20
    for (L, lab), c in cells.items():
        assert all(r["x"].shape == (8,) for r in c["recs"])
        if c["kind"] == "uniform":
            assert all(np.all(np.abs(r["x"]) <= c["scale"]) for r in c["recs"])


def test_cell_summary_statistics_are_consistent():
    cells = E.run_random(2, (16,), {16: 12}, E.RANGES[2:3])
    rows = E.summarise_cell(cells[(16, "u5")])
    assert [r["comparison"] for r in rows] == [f"{h}_vs_{r}" for h, r in E.COMPARISONS]
    for r in rows:
        assert r["n_vectors"] == 12
        assert r["max_abs_p50"] <= r["max_abs_p90"] <= r["max_abs_p95"] <= r["max_abs_p99"] <= r["max_abs_p100"]
        assert r["per_output_abs_p50"] <= r["per_output_abs_p100"]
        assert r["mean_abs_mean"] <= r["max_abs_p100"] and 0.0 <= r["argmax_match_rate"] <= 1.0
        assert r["sum_hw_min"] <= r["sum_hw_mean"] <= r["sum_hw_max"]


def test_worst_fixed_vector_diagnosis_is_consistent():
    cells = E.run_random(8, (64, 256), {64: 6, 256: 6}, E.RANGES[4:5])          # normal(0, 3)
    for impl, w in E.worst_fixed_vectors(cells).items():
        best = max(r["metrics"][f"{impl}_vs_float"]["max_abs"] for c in cells.values() for r in c["recs"])
        assert w["max_abs_vs_float"] == best
        assert 0 <= w["pass1_outputs_truncated_to_zero"] <= w["pass1_outputs"]
        assert w["mechanism"] and w["F_relative_deviation"] >= 0
        if w["pass1_outputs_truncated_to_zero"]:                    # truncation can only LOWER F
            assert w["F_fixed"] <= w["F_float_same_quantized_input"] + 1e-12
    w = E.worst_float_vector(cells)
    assert w["dominant"] in w["log_factors"] and w["max_abs"] > 0
    prod = w["factor_pass2_1_plus_e2"] * w["factor_lnu_exp_minus_dL"] / w["factor_pass1_F_over_S"]
    assert prod == pytest.approx(w["ratio_p_float_over_p_exact"], rel=1e-12)


def test_integer_boundary_jump_explains_float_vs_quantized_input():
    """z = -0.99984 (u=-1) vs -1.00093 (u=-2): same input to 1/256, but the output jumps by
    2*d2/(1+d2) (d2 = 1.0825) because the linear fit 2**v ~ v + d2 is discontinuous for d2 > 1."""
    cfg = WangConfig()
    from expu import expu
    hi = expu(np.array([-0.99984 / 1.4375]), 0.0, cfg.d2, cfg).out[0]
    lo = expu(np.array([-1.00093 / 1.4375]), 0.0, cfg.d2, cfg).out[0]
    assert hi / lo == pytest.approx(2 * cfg.d2 / (1 + cfg.d2), rel=2e-3)


def test_argmax_attribution_counts_add_up():
    cells = E.run_random(4, (8, 16), {8: 30, 16: 30}, E.RANGES[:2])
    for impl, a in E.argmax_attribution(cells).items():
        assert a["mismatches"] == a["mismatch_gap_below_input_lsb"] + a["mismatch_other_ties"] \
            + a["genuine_inversions"]
        assert a["genuine_inversions"] == 0
        assert a["mismatches"] <= a["vectors_with_gap_below_input_lsb"] + a["mismatch_other_ties"]


# ============================== precision sweep ===============================
def test_uniform_fraction_configs_are_valid():
    for f in (2, 3, 8, 16, 20):
        for mode in ("trunc", "round_half_up"):
            fx = E.uniform_fraction_config(f, mode)
            assert fx.fmt_t.frac_bits == fx.fmt_exp_out.frac_bits == fx.fmt_acc.frac_bits == f
            assert fx.fmt_t.int_bits == 4 and fx.fmt_a.int_bits == 7 and fx.fmt_exp_out.int_bits == 1


def test_error_decreases_with_precision_against_quantized_float_model():
    rows, floor = E.precision_sweep(7, (4, 8, 12), modes=("trunc",), n_vec=20)
    e = [r["vs_float_quantized_input_mean_abs"] for r in rows]
    assert e[0] > 4 * e[1] > 0 and e[1] > 4 * e[2]            # ~2x per bit; asserts >4x per 4 bits (observed ~8x)
    vs_exact = [r["vs_exact_mean_abs"] for r in rows]
    assert vs_exact[0] > vs_exact[2]
    assert vs_exact[2] == pytest.approx(floor["float_vs_exact_mean_abs"], rel=0.2)   # converges to float floor


# ====================== checks, outputs, determinism ==========================
def test_all_evaluation_checks_pass_in_quick_run():
    d, res = quick()
    assert res["checks"] and all(c["status"] == "PASS" for c in res["checks"]), \
        [c for c in res["checks"] if c["status"] != "PASS"]


def test_checks_are_not_vacuous_they_can_fail():
    d, res = quick()
    named, cells = copy.deepcopy(res["named"]), copy.deepcopy(res["cells"])
    some = next(iter(cells.values()))["recs"][0]
    some["identity_resid"] = 1.0
    some["env_violations"] = 3
    some["order_viol"]["fixed8"] = 2
    some["metrics"]["float_vs_exact"]["argmax_match"] = False
    some["unique_max"] = True
    status = {c["name"]: c["status"] for c in E.run_checks(named, cells)}
    assert status["float_error_identity"] == "FAIL"
    assert status["float_within_a_priori_envelope"] == "FAIL"
    assert status["fixed8_monotone_in_logits"] == "FAIL"
    assert status["float_argmax_equals_exact"] == "FAIL"
    assert status["fixed16_monotone_in_logits"] == "PASS"


def test_output_files_are_organised_and_nonempty():
    d, res = quick()
    for rel in ("report.md", "data/named_cases.json", "data/random_summary.csv", "data/random_vectors.csv",
                "data/precision_sweep.csv", "data/equal_logit_sweep.csv", "data/checks.json",
                "data/analysis.json", "plots/required_inputs.png", "plots/error_distribution.png",
                "plots/error_by_length_and_range.png", "plots/output_scatter.png",
                "plots/error_vs_precision.png", "plots/equal_logit_sweep.png"):
        assert (d / rel).stat().st_size > 500, rel
    for p in (d / "plots").glob("*.png"):
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    data = json.loads((d / "data" / "named_cases.json").read_text())
    c = data["required_2_1_0_-1"]
    assert c["x"] == [2.0, 1.0, 0.0, -1.0] and len(c["probabilities"]["fixed8"]) == 4
    assert len(c["comparisons"]["float_vs_exact"]["abs_err"]) == 4
    txt = (d / "report.md").read_text()
    assert "MEASURED" in txt and "seed" in txt and "PROVISIONAL" in txt and "EXPERIMENTAL" in txt
    assert "required_10_9_8" in txt and "Checks (pass/fail)" in txt


def test_evaluation_is_deterministic_for_a_fixed_seed():
    d1, d2 = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    E.run_evaluation(d1, seed=5, quick=True, make_plots=False, verbose=False)
    E.run_evaluation(d2, seed=5, quick=True, make_plots=False, verbose=False)
    for rel in ("report.md", "data/random_vectors.csv", "data/named_cases.json", "data/precision_sweep.csv"):
        assert (d1 / rel).read_text() == (d2 / rel).read_text(), rel


def test_cli_runs_from_project_root_and_writes_results_dir():
    out = Path(tempfile.mkdtemp()) / "res"
    r = subprocess.run([sys.executable, "evaluate.py", "--quick", "--no-plots", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "report.md").exists() and (out / "data" / "checks.json").exists()
    assert not (out / "plots" / "required_inputs.png").exists()
    assert E.DEFAULT_OUT == ROOT / "results"


def test_main_returns_nonzero_when_a_check_fails():
    original = E.run_checks
    try:
        E.run_checks = lambda named, cells: [{"name": "forced", "status": "FAIL", "detail": "x"}]
        out = Path(tempfile.mkdtemp())
        assert E.main(["--quick", "--no-plots", "--out", str(out)]) == 1
    finally:
        E.run_checks = original
