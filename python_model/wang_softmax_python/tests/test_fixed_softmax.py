"""Stage 7: fixed-point EXPU / LNU / accumulator / full Wang softmax pipeline."""
import dataclasses
import math
from fractions import Fraction
import numpy as np
import pytest
import config as CFG
import fixed_point as FP
from config import FixedConfig, WangConfig
from expu import expu_fixed, expu_fixed_pass1, expu_fixed_pass2
from lnu import lnu, lnu_fixed
from adder_tree import accumulate_chunks, accumulate_chunks_fixed, balanced_tree_sum_int
from wang_softmax import WangSoftmax, wang_softmax, WangTrace, FixedWangTrace
from softmax_reference import softmax_reference
from compare_fixed import compare_fixed, error_metrics

D11 = dict(d1=1.0, d2=1.0)
HAND = [2.0, 1.0, 0.0, -1.0]
MODES = ("trunc", "toward_zero", "round_half_up", "round_half_even")


def fcfg(fixed=None, **kw):
    return WangConfig(fixed=fixed if fixed is not None else FixedConfig(), **kw)


# =============== independent integer model (no code shared with modules) ======
def _rfrac(f: Fraction, mode):
    return {"trunc": lambda: math.floor(f), "toward_zero": lambda: int(f),
            "round_half_up": lambda: math.floor(f + Fraction(1, 2)),
            "round_half_even": lambda: round(f)}[mode]()


def _rs(x, n, mode):
    return x << -n if n <= 0 else _rfrac(Fraction(x, 1 << n), mode)


def _fit(c, width, signed, ovf):
    lo, hi = (-(1 << (width - 1)), (1 << (width - 1)) - 1) if signed else (0, (1 << width) - 1)
    if ovf == "saturate":
        return max(lo, min(hi, c))
    c %= 1 << width
    return c - (1 << width) if signed and c > hi else c


def ref_pipeline(x, fx, d1, d2, lanes=16):
    md, ov, nin = fx.rounding, fx.overflow, fx.input_frac_bits
    m = max(x)
    t = [_fit(_rfrac(Fraction(-(xi - m)) * (1 << nin), md), fx.data_width, False, ov) for xi in x]
    y = [-(c << (fx.a_frac_bits - nin)) for c in t]
    L, N = 23, 11                                   # 1.0111b and 0.1011b, 4 fraction bits

    def expu(ys, ln_f, d):
        dc = _fit(_rfrac(Fraction(d) * (1 << fx.z_frac_bits), md), fx.z_frac_bits + 2, False, "saturate")
        out = []
        for yc in ys:
            a = _fit(yc - ln_f, fx.a_width, True, ov)
            z = _fit(_rs(a * L, fx.a_frac_bits + 4 - fx.z_frac_bits, md), fx.z_width, True, ov)
            u, v = z >> fx.z_frac_bits, z & ((1 << fx.z_frac_bits) - 1)
            sh = u + fx.exp_out_frac_bits - fx.z_frac_bits
            out.append(_fit(_rs(v + dc, -sh, md), fx.exp_out_width, False, ov))
        return out

    e1 = expu(y, 0, d1)
    lane = [_rs(c, fx.exp_out_frac_bits - fx.acc_frac_bits, md) for c in e1]
    lane += [0] * ((-len(lane)) % lanes)
    F = 0
    for i in range(0, len(lane), lanes):
        F = _fit(F + sum(lane[i:i + lanes]), fx.acc_width, False, ov)
    if F == 0:
        return None
    msb = F.bit_length() - 1
    w = msb - fx.acc_frac_bits
    kf = _rfrac(Fraction(F - (1 << msb), 1 << msb) * (1 << fx.lnu_frac_bits), md)
    prod = ((w << fx.lnu_frac_bits) + kf) * N
    ln_f = _fit(_rs(prod, fx.lnu_frac_bits + 4 - fx.a_frac_bits, md), fx.a_width, True, ov)
    return dict(t=t, y=y, e1=e1, F=F, ln_f=ln_f, p=expu(y, ln_f, d2))


# small widths so that a, z, output and accumulator all overflow on ordinary inputs
STRESS = dict(data_width=8, input_frac_bits=4, exp_out_frac_bits=8, exp_out_width=9,
              acc_width=12, acc_frac_bits=8, a_width=13, a_frac_bits=8,
              z_width=12, z_frac_bits=8, lnu_frac_bits=6)


@pytest.mark.parametrize("overflow", ["saturate", "wrap"])
@pytest.mark.parametrize("rounding", MODES)
@pytest.mark.parametrize("name,base", [("default", {}), ("stress", STRESS)])
def test_bit_exact_against_independent_integer_model(name, base, rounding, overflow):
    fx = FixedConfig(rounding=rounding, overflow=overflow, **base)
    rng = np.random.default_rng(7)
    vectors = [HAND, [0.0], [1.0] * 20, list(rng.uniform(-12, 12, 30)), list(rng.uniform(-3, 3, 41)),
               list(rng.uniform(-1, 1, 5)), [0.0, -100.0], [5.5, 5.5, -0.03125, -7.96875]]
    vectors += [list(rng.uniform(-8, 8, int(rng.integers(1, 50)))) for _ in range(25)]
    checked = overflowed = 0
    for x in vectors:
        x = [float(v) for v in x]
        want = ref_pipeline(x, fx, 1.16, 1.0825)
        cfg = WangConfig(fixed=fx)
        if want is None:                              # wrapped accumulator hit exactly 0
            with pytest.raises(ValueError):
                WangSoftmax(cfg).run(x)
            continue
        tr = WangSoftmax(cfg).run(x, debug=True)
        assert tr.t_code.tolist() == want["t"]
        assert tr.y_code.tolist() == want["y"]
        assert tr.pass1.out_code.tolist() == want["e1"]
        assert tr.F_code == want["F"]
        assert tr.lnu.ln_f_code == want["ln_f"]
        assert tr.p_code.tolist() == want["p"]
        checked += 1
        overflowed += tr.any_overflow
    assert checked >= 25
    if name == "stress":
        assert overflowed >= 5                       # the stress config really overflows


# ===================== pinned hand-derived example ============================
def test_hand_example_every_code_default_8bit_d_equal_one():
    """Derived by hand (not by running the code) for x=[2,1,0,-1], d1=d2=1."""
    tr = WangSoftmax(fcfg(**D11)).run(HAND, debug=True)
    assert tr.t_code.tolist() == [0, 16, 32, 48]                 # t = 0,1,2,3 in UQ4.4
    assert tr.y_code.tolist() == [0, -256, -512, -768]           # SQ7.8
    assert tr.pass1.out_code.tolist() == [256, 100, 36, 13]      # 1.0, .390625, .140625, .0508 (trunc)
    assert tr.F_code == 405 and tr.F == 405 / 256
    assert (tr.lnu.w, tr.lnu.kf_code, tr.lnu.log2_code) == (0, 149, 149)
    assert tr.lnu.prod_code == 149 * 11 == 1639
    assert tr.lnu.ln_f_code == 102                               # floor(1639/16)
    assert tr.pass2.a_code.tolist() == [-102, -358, -614, -870]
    assert tr.pass2.z_code.tolist() == [-147, -515, -883, -1251]  # floor(a*23/16)
    assert tr.pass2.u.tolist() == [-1, -3, -4, -5]
    assert tr.pass2.v_code.tolist() == [109, 253, 141, 29]
    assert tr.pass2.m_code.tolist() == [365, 509, 397, 285]
    assert tr.p_code.tolist() == [182, 63, 24, 8]
    np.testing.assert_array_equal(tr.p_raw, np.array([182, 63, 24, 8]) / 256)


def test_negative_z_uses_floor_split_in_fixed_point():
    cfg = fcfg(**D11)
    t = expu_fixed(np.array([-256]), 0, 1.0, cfg)        # y = -1: z = -1.4375
    assert t.z_code.tolist() == [-368] and t.u.tolist() == [-2] and t.v_code.tolist() == [144]
    assert t.v[0] == 0.5625 and t.z[0] == -1.4375         # u = floor(z), v in [0,1)
    assert t.mantissa[0] == 1.5625 and t.out[0] == 1.5625 / 4


# ===================== rounding boundary inside the datapath ==================
@pytest.mark.parametrize("a_code,expected", [
    (-512, dict(trunc=-12, toward_zero=-11, round_half_up=-11, round_half_even=-12)),
    (512, dict(trunc=11, toward_zero=11, round_half_up=12, round_half_even=12)),
])
def test_z_rounding_at_exact_half_lsb(a_code, expected):
    # a = +-2.0 (code +-512 at 8 frac bits), z_frac_bits = 2: a*23/16 = +-2.875 = +-11.5 lsb
    for mode, want in expected.items():
        fx = FixedConfig(z_frac_bits=2, rounding=mode)
        t = expu_fixed(np.array([a_code]), 0, 1.0, WangConfig(fixed=fx))
        assert t.z_code.tolist() == [want], mode


def test_output_shift_rounding_at_exact_half_lsb():
    # Default formats, d = 1.0, y = -3 (a code -768): z = -4.3125, u = -5, v = .6875 (176),
    # m = 432; out = 432 * 2**-5 = 13.5 LSB of UQ1.8  -> exactly half an LSB.
    want = dict(trunc=13, toward_zero=13, round_half_up=14, round_half_even=14)   # 13 odd -> even 14
    for mode, code in want.items():
        t = expu_fixed(np.array([-768]), 0, 1.0, fcfg(FixedConfig(rounding=mode)))
        assert (int(t.u[0]), int(t.v_code[0]), int(t.m_code[0])) == (-5, 176, 432)
        assert t.out_code.tolist() == [code], mode
    # y = -0.5 (a code -128), UQ1.5 output: m = 328, shift -4 -> 20.5 LSB (20 even)
    want = dict(trunc=20, toward_zero=20, round_half_up=21, round_half_even=20)
    for mode, code in want.items():
        fx = FixedConfig(rounding=mode, exp_out_frac_bits=5, exp_out_width=8)
        t = expu_fixed(np.array([-128]), 0, 1.0, fcfg(fx))
        assert (int(t.u[0]), int(t.m_code[0])) == (-1, 328)
        assert t.out_code.tolist() == [code], mode


# ===================== overflow: saturation and wraparound ====================
def test_input_saturates_vs_wraps_for_very_negative_logit():
    sat = WangSoftmax(fcfg()).run([0.0, -100.0], debug=True)
    wrp = WangSoftmax(fcfg(FixedConfig(overflow="wrap"))).run([0.0, -100.0], debug=True)
    assert sat.t_code.tolist() == [0, 255] and sat.y_q.tolist() == [0.0, -15.9375]
    assert sat.overflow["input"] and wrp.overflow["input"]
    assert wrp.t_code.tolist() == [0, 64] and wrp.y_q.tolist() == [0.0, -4.0]   # 1600 mod 256
    assert wrp.p_raw[1] > 10 * sat.p_raw[1]            # wrap makes a hopeless logit look plausible
    assert sat.p_raw[1] < 1e-3


def test_no_overflow_flag_on_ordinary_input():
    assert not WangSoftmax(fcfg()).run(HAND, debug=True).any_overflow


def test_accumulator_saturation_and_wraparound():
    x = [0.0] * 16                                       # 16 equal logits: F ~ 16 * d1
    base = dict(acc_width=10, acc_frac_bits=8)           # UQ2.8: max 3.996
    sat = WangSoftmax(fcfg(FixedConfig(**base))).run(x, debug=True)
    wrp = WangSoftmax(fcfg(FixedConfig(overflow="wrap", **base))).run(x, debug=True)
    exact_F = int(sat.pass1.out_code.sum())               # 16 * 296 = 4736
    assert exact_F == 16 * 296 and sat.acc["overflow"] and wrp.acc["overflow"]
    assert sat.F_code == 1023
    assert wrp.F_code == exact_F % 1024 == 640
    assert sat.F == 1023 / 256 and wrp.F == 2.5
    with pytest.raises(ValueError):                       # d1 = 1.0: wraps to exactly 0 -> LOD undefined
        WangSoftmax(fcfg(FixedConfig(overflow="wrap", **base), d1=1.0)).run(x)


def test_accumulator_register_saturates_per_cycle_not_just_at_end():
    fx = FixedConfig(acc_width=10, acc_frac_bits=8)          # register max 1023
    codes = np.array([511] * 4)
    two = accumulate_chunks_fixed(codes, 2, fx, return_trace=True)     # cycles: 1022, +1022
    assert two["running"] == [1022, 1023] and two["F"] == 1023 and two["overflow"]
    one = accumulate_chunks_fixed(codes, 1, fx, return_trace=True)     # 511, 1022, sat, sat
    assert one["running"] == [511, 1022, 1023, 1023]
    fx_w = FixedConfig(acc_width=10, acc_frac_bits=8, overflow="wrap")
    assert accumulate_chunks_fixed(codes, 1, fx_w, return_trace=True)["running"] == \
        [511, 1022, (1533) % 1024, (2044) % 1024]                      # 509, 1020
    assert accumulate_chunks_fixed(codes, 4, fx) == 1023               # one cycle of 2044

def test_expu_output_saturation_and_wraparound():
    # a = +2.0: z = 2.875 -> u = 2, v = 0.875, m = 1.875 + (d-1)... out = m*4 > UQ1.8 max
    y = np.array([512])
    sat = expu_fixed(y, 0, 1.0, fcfg())
    wrp = expu_fixed(y, 0, 1.0, fcfg(FixedConfig(overflow="wrap")))
    assert sat.m_code.tolist() == [480] and sat.u.tolist() == [2]
    assert sat.out_code.tolist() == [511] and sat.overflow["out"]
    assert wrp.out_code.tolist() == [(480 * 4) % 512] == [384]
    assert wrp.overflow["out"]


def test_a_format_overflow_saturation_and_wraparound():
    y = np.array([-32768])                                # a-format minimum
    sat = expu_fixed(y, 100, 1.0, fcfg())                 # a = -32868 -> below the minimum
    wrp = expu_fixed(y, 100, 1.0, fcfg(FixedConfig(overflow="wrap")))
    assert sat.a_code.tolist() == [-32768] and sat.overflow["a"]
    assert wrp.a_code.tolist() == [-32868 + 65536] and wrp.overflow["a"]


def test_z_format_overflow_saturation_and_wraparound():
    fx_kw = dict(z_width=10, z_frac_bits=8)               # z in [-2, 1.996]
    y = np.array([-512])                                  # z = -2.875: below range
    sat = expu_fixed(y, 0, 1.0, fcfg(FixedConfig(**fx_kw)))
    wrp = expu_fixed(y, 0, 1.0, fcfg(FixedConfig(overflow="wrap", **fx_kw)))
    assert sat.z_code.tolist() == [-512] and sat.overflow["z"]
    assert wrp.z_code.tolist() == [-736 + 1024] and wrp.overflow["z"]


def test_lnf_overflow_flag():
    fx = FixedConfig(a_width=13, a_frac_bits=8)           # a range +-16; ln of a big F fits...
    tr = lnu_fixed(fx.fmt_acc.max_code, fcfg(fx))
    assert not tr.overflow
    big = FixedConfig(a_width=10, a_frac_bits=8, data_width=8, input_frac_bits=7)   # a in [-2, 2)
    tr = lnu_fixed(big.fmt_acc.max_code, fcfg(big))        # ln F ~ 7.6 > 1.99
    assert tr.overflow and tr.ln_f_code == big.fmt_a.max_code


# ===================== LNU fixed ==============================================
def test_lnu_fixed_negative_w_and_rounding_of_negative_ln():
    # F = 101/256 = 0.3945: w = -2, k = 101/64, k-1 = 37/64 = 148/256
    modes = dict(trunc=-251, toward_zero=-250, round_half_up=-250, round_half_even=-250)
    for mode, want in modes.items():
        tr = lnu_fixed(101, fcfg(FixedConfig(rounding=mode)))
        assert (tr.w, tr.kf_code, tr.log2_code, tr.prod_code) == (-2, 148, -364, -4004)
        assert tr.ln_f_code == want, mode            # -4004/16 = -250.25
    tr = lnu_fixed(100, fcfg())                       # k - 1 + w = -1.4375 exactly
    assert tr.log2_approx == -1.4375 and tr.ln_f_code == -253 and tr.ln_f == -253 / 256


def test_lnu_fixed_rounding_carry_propagates_into_w():
    fx = FixedConfig(lnu_frac_bits=2, rounding="round_half_up")
    tr = lnu_fixed(486, fcfg(fx))                     # k-1 = 230/256 = .898 -> 3.59/4 -> rounds to 4/4
    assert (tr.w, tr.kf_code, tr.log2_code) == (0, 4, 4)       # (k-1) became 1.0
    assert tr.ln_f_code == 176                        # 0.6875 * 1.0 in 8 frac bits


def test_lnu_fixed_rejects_zero_and_bad_codes():
    for bad in (0, -1, 1 << 24, 1.0, True):
        with pytest.raises(ValueError):
            lnu_fixed(bad, fcfg())


def test_lnu_fixed_matches_float_lnu_on_wide_format():
    cfg_w = fcfg(CFG.FIXED_WIDE_REFERENCE)
    for code in (1, 3, 255, 256, 257, 1 << 20, (1 << 30) + 12345, 5 << 30, (1 << 40) + 999):
        F = math.ldexp(code, -30)
        assert lnu_fixed(code, cfg_w).ln_f == pytest.approx(lnu(F, WangConfig()).ln_f, abs=2e-7)


# ===================== accumulator ============================================
def test_fixed_tree_is_exact_and_independent_of_lane_count():
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 512, 77)
    fx = FixedConfig()
    ref = int(codes.sum())
    assert balanced_tree_sum_int(codes) == ref
    for lanes in (1, 2, 5, 8, 16, 100):
        assert accumulate_chunks_fixed(codes, lanes, fx) == ref


def test_tree_width_grows_one_bit_per_level():
    fx = FixedConfig(acc_width=32)
    tr = accumulate_chunks_fixed(np.full(16, 511), 16, fx, return_trace=True)
    assert tr["tree_bits"] == 9 + 4 and tr["cycles"] == 1 and not tr["overflow"]


def test_accumulator_alignment_left_and_right():
    up = FixedConfig(exp_out_frac_bits=4, exp_out_width=6, acc_frac_bits=8)      # acc finer: exact
    assert accumulate_chunks_fixed(np.array([3]), 1, up) == 48
    down = dict(exp_out_frac_bits=4, exp_out_width=6, acc_frac_bits=2)           # drop 2 bits per lane
    assert accumulate_chunks_fixed(np.array([6]), 1, FixedConfig(**down)) == 1   # trunc 6/4
    assert accumulate_chunks_fixed(np.array([6]), 1, FixedConfig(rounding="round_half_up", **down)) == 2
    assert accumulate_chunks_fixed(np.array([6]), 1, FixedConfig(rounding="round_half_even", **down)) == 2
    assert accumulate_chunks_fixed(np.array([2]), 1, FixedConfig(rounding="round_half_even", **down)) == 0


def test_accumulator_input_validation():
    fx = FixedConfig()
    for bad in (np.array([]), np.array([[1, 2]]), np.array([1.5]), np.array([512]), np.array([-1])):
        with pytest.raises(ValueError):
            accumulate_chunks_fixed(bad, 16, fx)
    for lanes in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            accumulate_chunks_fixed(np.array([1]), lanes, fx)


# ===================== separate biases, assumptions, no renormalisation =======
def test_d1_and_d2_remain_separate_in_fixed_point():
    x = np.random.default_rng(0).uniform(-4, 4, 20)
    base = WangSoftmax(fcfg(d1=1.0, d2=1.0)).run(x, debug=True)
    only_d2 = WangSoftmax(fcfg(d1=1.0, d2=1.25)).run(x, debug=True)
    only_d1 = WangSoftmax(fcfg(d1=1.25, d2=1.0)).run(x, debug=True)
    assert only_d2.F_code == base.F_code                       # pass 1 does not see d2
    assert only_d2.pass1.d_code == base.pass1.d_code == 256
    assert only_d2.pass2.d_code == 320 and only_d2.pass2.d == 1.25
    assert not np.array_equal(only_d2.p_code, base.p_code)
    assert only_d1.pass1.d_code == 320 and only_d1.pass2.d_code == 256
    assert only_d1.F_code != base.F_code                        # pass 2 does not feed back into F
    assert (WangConfig().d1, WangConfig().d2) != (1.0, 1.0)
    dflt = WangSoftmax(fcfg()).run(x, debug=True)
    assert dflt.pass1.d_code == FP.quantize(WangConfig().d1, dflt.fx.fmt_m)
    assert dflt.pass2.d_code == FP.quantize(WangConfig().d2, dflt.fx.fmt_m)
    assert dflt.pass1.d_code != dflt.pass2.d_code


def test_pass1_ln_f_assumption_is_a_config_value_in_fixed_point():
    x = np.array(HAND)
    assert WangSoftmax(fcfg()).run(x, debug=True).pass1.ln_f_code == 0
    t = WangSoftmax(fcfg(pass1_ln_f=0.5)).run(x, debug=True)
    assert t.pass1.ln_f_code == 128 and t.pass2.ln_f_code == t.lnu.ln_f_code


def test_outputs_are_not_renormalised_and_flag_is_refused():
    tr = WangSoftmax(fcfg(**D11)).run(HAND, debug=True)      # codes 182+63+24+8 (hand example)
    np.testing.assert_array_equal(tr.p, tr.p_raw)
    assert tr.p_raw.sum() == 277 / 256                 # raw sum of the codes, left as is
    assert abs(tr.p_raw.sum() - 1.0) > 1e-3
    with pytest.raises(ValueError):
        WangSoftmax(fcfg(renormalize_output=True)).run(HAND)


def test_fixed_probabilities_are_exact_multiples_of_lsb_and_nonnegative():
    fx = FixedConfig()
    for seed in range(5):
        x = np.random.default_rng(seed).uniform(-6, 6, 33)
        p = wang_softmax(x, fcfg(fx))
        codes = p * 2 ** fx.exp_out_frac_bits
        np.testing.assert_array_equal(codes, np.round(codes))
        assert np.all(p >= 0) and np.all(p <= fx.fmt_exp_out.max_value)


# ===================== float model not broken =================================
def test_float_model_unchanged_by_stage7():
    x = [0.3, -1.2, 2.2, 0.0, 1.1]
    p = wang_softmax(x)
    tr = wang_softmax(x, debug=True)
    assert isinstance(tr, WangTrace) and WangConfig().fixed is None
    np.testing.assert_array_equal(p, tr.p_raw)
    np.testing.assert_array_equal(p, wang_softmax(x, dataclasses.replace(fcfg(), fixed=None)))
    assert isinstance(WangSoftmax(fcfg()).run(x, debug=True), FixedWangTrace)
    with pytest.raises(ValueError):
        expu_fixed(np.array([0]), 0, 1.0, WangConfig())      # float config rejected


def test_paper_constants_never_rounded_silently():
    with pytest.raises(ValueError):
        WangSoftmax(WangConfig(log2e_const=1.44, fixed=FixedConfig())).run(HAND)
    with pytest.raises(ValueError):
        WangSoftmax(WangConfig(ln2_const=0.693, fixed=FixedConfig())).run(HAND)
    for bad_d in (-0.1, 2.0, float("nan")):
        with pytest.raises(ValueError):
            WangSoftmax(WangConfig(d1=bad_d, fixed=FixedConfig())).run(HAND)


def test_fixedconfig_validation():
    for kw in (dict(rounding="nearest"), dict(overflow="clip"), dict(a_frac_bits=2),
               dict(a_width=10, a_frac_bits=8), dict(lnu_frac_bits=0), dict(data_width=0),
               dict(acc_frac_bits=30), dict(z_frac_bits=20)):
        with pytest.raises(ValueError):
            FixedConfig(**kw)
    FixedConfig()                                           # defaults valid
    for name, fx in CFG.PRESETS.items():
        assert fx.fmt_t.width == fx.data_width


# ===================== documented formats are true ============================
@pytest.mark.parametrize("name", ["8bit", "16bit", "wide"])
@pytest.mark.parametrize("overflow", ["saturate", "wrap"])
def test_every_intermediate_fits_its_documented_format(name, overflow):
    fx = dataclasses.replace(CFG.PRESETS[name], overflow=overflow)
    rng = np.random.default_rng(11)
    for _ in range(15):
        x = rng.uniform(-9, 9, int(rng.integers(2, 60)))
        try:
            tr = WangSoftmax(WangConfig(fixed=fx)).run(x, debug=True)
        except ValueError:
            assert overflow == "wrap"                       # only a wrapped F == 0 may raise
            continue
        for tt in (tr.pass1, tr.pass2):
            assert not np.any(FP.would_overflow(tt.y_code, fx.fmt_a))
            assert not np.any(FP.would_overflow(tt.a_code, fx.fmt_a))
            assert not np.any(FP.would_overflow(tt.z_code, fx.fmt_z))
            assert not np.any(FP.would_overflow(tt.v_code, fx.fmt_v))
            assert not np.any(FP.would_overflow(tt.m_code, fx.fmt_m))
            assert not np.any(FP.would_overflow(tt.out_code, fx.fmt_exp_out))
            assert not np.any(FP.would_overflow(tt.u, FP.QFormat(fx.z_width - fx.z_frac_bits, 0, True)))
            assert 0 <= tt.d_code <= fx.fmt_m.max_code
        assert not np.any(FP.would_overflow(tr.t_code, fx.fmt_t))
        assert not FP.would_overflow(tr.F_code, fx.fmt_acc)
        assert not FP.would_overflow(tr.lnu.log2_code, fx.fmt_log2)
        assert not FP.would_overflow(tr.lnu.ln_f_code, fx.fmt_a)


def test_format_table_documents_every_stage():
    fx = FixedConfig()
    rows = {r[0]: r for r in fx.format_table()}
    assert rows["t = -y (input magnitude)"][1].name == "UQ4.4"
    assert rows["a = y - lnF"][1].name == "SQ7.8" and rows["a = y - lnF"][2] == 16
    assert rows["v = z & mask (fraction, [0,1))"][1].name == "UQ0.8"
    assert rows["m = v + d"][1].name == "UQ2.8"
    assert rows["EXPU out = m * 2^u"][1].name == "UQ1.8"
    assert rows["F (accumulator)"][1].name == "UQ16.8"
    text = fx.describe()
    assert "ALL EXPERIMENTAL" in text and all(r[0] in text for r in fx.format_table())


# ===================== accuracy: fixed vs float Wang vs exact softmax =========
def _vectors(n, length, r, seed):
    rng = np.random.default_rng(seed)
    return [rng.uniform(-r, r, length) for _ in range(n)]


def test_comparison_metrics_are_computed_correctly():
    r = compare_fixed(HAND, fcfg())
    p, ref = r["p_fixed"], softmax_reference(HAND)
    e = np.abs(p - ref)
    m = r["fixed_vs_exact"]
    assert m["max_abs"] == e.max() and m["mean_abs"] == pytest.approx(e.mean(), rel=1e-12)
    assert m["rmse"] == pytest.approx(np.sqrt(np.mean(e ** 2)), rel=1e-12)
    assert m["sum_p"] == p.sum() and m["argmax_match"]
    np.testing.assert_array_equal(r["p_float"], wang_softmax(HAND))
    np.testing.assert_array_equal(r["p_fixed"], r["trace"].p_raw)
    assert error_metrics([1, 0], [1, 0])["max_abs"] == 0.0
    with pytest.raises(ValueError):
        compare_fixed(HAND, WangConfig())


def test_8bit_fixed_outputs_close_to_float_wang_and_exact_softmax():
    cfg = fcfg(CFG.FIXED_8BIT)
    worst_q = worst_exact = 0.0
    for x in _vectors(100, 16, 5.0, 1):
        r = compare_fixed(x, cfg)
        worst_q = max(worst_q, r["fixed_vs_float_quantized_input"]["max_abs"])
        worst_exact = max(worst_exact, r["fixed_vs_exact"]["max_abs"])
        assert not r["trace"].any_overflow
        assert r["fixed_vs_exact"]["sum_p"] < 1.2         # not renormalised, but sane
    # Regression bounds, MEASURED (worst 0.0230 / 0.0484 for this seed); the experimental
    # 8-bit "trunc" format is coarse: many small pass-1 terms truncate to 0, biasing F and lnF.
    assert worst_q < 0.03           # arithmetic error alone (same quantized input)
    assert worst_exact < 0.06


def test_round_to_nearest_reduces_8bit_arithmetic_error_versus_truncation():
    xs = _vectors(100, 16, 5.0, 1)
    err = {}
    for mode in ("trunc", "round_half_up"):
        cfg = fcfg(dataclasses.replace(CFG.FIXED_8BIT, rounding=mode))
        rs = [compare_fixed(x, cfg)["fixed_vs_float_quantized_input"] for x in xs]
        err[mode] = (np.mean([r["mean_abs"] for r in rs]), max(r["max_abs"] for r in rs))
    assert err["round_half_up"][0] < err["trunc"][0]
    assert err["round_half_up"][1] < err["trunc"][1]


def test_16bit_fixed_tracks_float_wang_closely_and_inherits_its_exact_error():
    cfg = fcfg(CFG.FIXED_16BIT)
    for x in _vectors(60, 32, 5.0, 2):
        r = compare_fixed(x, cfg)
        assert r["fixed_vs_float_quantized_input"]["max_abs"] < 1e-3
        # fixed-vs-exact is the float Wang error plus at most a small extra
        assert r["fixed_vs_exact"]["max_abs"] <= r["float_vs_exact"]["max_abs"] + 6e-3


def test_error_shrinks_as_precision_grows():
    xs = _vectors(40, 24, 4.0, 5)
    mean_err = {}
    for name in ("8bit", "16bit", "wide"):
        cfg = fcfg(CFG.PRESETS[name])
        mean_err[name] = np.mean([compare_fixed(x, cfg)["fixed_vs_float_quantized_input"]["mean_abs"]
                                  for x in xs])
    assert mean_err["8bit"] > 10 * mean_err["16bit"] > 0
    assert mean_err["16bit"] > mean_err["wide"]
    assert mean_err["wide"] < 1e-7


def test_wide_format_converges_to_float_wang_model():
    # logits that are exact multiples of 2**-8, d1 = d2 = 1: only the finite widths differ
    rng = np.random.default_rng(9)
    cfg_w, cfg_f = fcfg(CFG.FIXED_WIDE_REFERENCE, **D11), WangConfig(**D11)
    for _ in range(20):
        x = rng.integers(-8 * 256, 8 * 256, 25) / 256.0
        np.testing.assert_allclose(wang_softmax(x, cfg_w), wang_softmax(x, cfg_f), atol=2e-7)
    # same with the provisional (non dyadic) d1, d2 quantized to 24 fraction bits
    x = rng.integers(-8 * 256, 8 * 256, 25) / 256.0
    np.testing.assert_allclose(wang_softmax(x, fcfg(CFG.FIXED_WIDE_REFERENCE)), wang_softmax(x), atol=2e-7)


def test_fixed_vs_exact_reference_on_hand_example():
    p = wang_softmax(HAND, fcfg())
    np.testing.assert_allclose(p, softmax_reference(HAND), atol=0.01)
    assert int(np.argmax(p)) == 0


def test_input_quantization_is_a_separate_error_source():
    x = [0.3, -0.2, 0.31, 1.7]                                # not multiples of 1/16
    r = compare_fixed(x, fcfg(CFG.FIXED_8BIT))
    assert not np.array_equal(r["trace"].y_q, r["trace"].y)
    assert np.all(r["trace"].y_q >= r["trace"].y)             # trunc of |y| moves y toward 0
    assert r["fixed_vs_float"]["max_abs"] != r["fixed_vs_float_quantized_input"]["max_abs"]


# ===================== edge cases =============================================
def test_edge_vectors_single_equal_and_large():
    for fx in (CFG.FIXED_8BIT, CFG.FIXED_16BIT):
        cfg = fcfg(fx)
        one = WangSoftmax(cfg).run([3.0], debug=True)
        assert one.t_code.tolist() == [0] and one.p_raw.shape == (1,)
        eq = wang_softmax([1.0] * 8, cfg)
        assert np.all(eq == eq[0])
    big = np.random.default_rng(4).uniform(-5, 5, 2048)         # many lanes cycles
    tr = WangSoftmax(fcfg()).run(big, debug=True)
    assert tr.acc["cycles"] == 128 and not tr.acc["overflow"]
    assert tr.p_raw.shape == (2048,) and np.all(tr.p_raw >= 0)


def test_invalid_inputs_rejected_in_fixed_mode():
    for bad in ([], [np.nan], [np.inf, 1.0], [[1.0, 2.0]], ["a"]):
        with pytest.raises(ValueError):
            WangSoftmax(fcfg()).run(bad)
