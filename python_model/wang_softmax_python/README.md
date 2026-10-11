# wang_softmax_python
Python reference model of the Wang et al. softmax architecture
(EXPU -> adder tree -> LNU -> EXPU). Status: Stage 8 (evaluation and comparison) complete.

Data flow:  x -> preprocess (y=x-max) -> EXPU pass1 (d1) -> adder tree/accumulator -> F
            -> LNU (lnF) -> EXPU pass2 (d2, input y-lnF) -> p

Provenance policy: every constant is tagged PAPER / DERIVED / PROVISIONAL in
coefficients.py. d1, d2 and all fixed-point widths are PROVISIONAL.

## Stage 7 - fixed-point model (all formats EXPERIMENTAL, none from the paper)

`WangConfig(fixed=FixedConfig(...))` runs the same dataflow on integer codes
(bit-exact, shifts/adds/one constant multiply only); `fixed=None` (default) is the
unchanged float model. Outputs are never renormalised (`renormalize_output=True` is
refused in fixed mode). d1 and d2 stay separate (each quantized to UQ2.z_frac).

Paper facts used: 8/16-bit input/output supported (Sec. V-A); constants 1.0111b and
0.1011b (must be exactly representable - they are never rounded silently).
Everything else (widths, fraction bits, rounding, overflow) is a configurable guess.

Default `FIXED_8BIT` (`python compare_fixed.py --formats` prints every intermediate):

| value | format | note |
|---|---|---|
| t = -y | UQ4.4 (8 b) | input magnitude, y <= 0 so no sign bit |
| a = y - lnF, lnF | SQ7.8 (16 b) | adder #1 |
| z = a*1.0111b | SQ7.8 (16 b) | exact product has 12 frac bits, then rounded |
| u, v | SQ7.0, UQ0.8 | floor split via arithmetic shift / bit mask |
| d1, d2, m = v+d | UQ2.8 (10 b) | 0 <= d < 2 |
| EXPU out, p | UQ1.8 (9 b) | |
| F accumulator | UQ16.8 (24 b) | tree exact, register saturates/wraps per cycle |
| k-1+w | SQ5.8 (14 b) | LNU, then x 0.1011b -> a-format |

Rounding: `trunc` (=floor, default), `toward_zero`, `round_half_up`, `round_half_even`.
Overflow: `saturate` (default) or `wrap`. Presets: `8bit`, `16bit`, `wide` (convergence check only).

    python compare_fixed.py --x 2 1 0 -1 --preset 8bit
    python compare_fixed.py --random 200 --len 64 --preset 16bit --rounding round_half_up
    pytest -q

## Stage 8 - evaluation and comparison (all numbers MEASURED by running the Python models)

Compares (1) exact softmax, (2) float Wang model, (3) fixed-point Wang model
(`fixed8` = `FIXED_8BIT`, `fixed16` = `FIXED_16BIT`; both EXPERIMENTAL, trunc + saturate).
No hardware (area/power/timing) metric is produced. d1, d2 are PROVISIONAL.

Run everything from the project root:

    pip install -r requirements.txt
    python evaluate.py                       # full run, seed 2026 -> results/   (~1 min)
    python evaluate.py --quick               # small smoke run
    python evaluate.py --seed 7 --out results_seed7
    python evaluate.py --no-plots
    pytest -q                                # all tests (incl. tests/test_evaluate.py)
    pytest -q tests/test_evaluate.py         # Stage 8 tests only

`evaluate.py` exits 1 (after writing all outputs) if any evaluation check fails.

Outputs in `results/`:

| path | content |
|---|---|
| `report.md` | all tables, checks, error analysis (start here) |
| `data/named_cases.json` | the 5 required inputs + 20 boundary cases: probabilities of all implementations, per-output abs/rel error, metrics |
| `data/random_summary.csv` | per (length, distribution, comparison): percentiles of max abs error, per-output percentiles, sum range, argmax rate, relative error |
| `data/random_vectors.csv` | one row per random vector and comparison |
| `data/precision_sweep.csv`, `data/equal_logit_sweep.csv` | error vs fraction bits; equal logits N = 1..1024 |
| `data/checks.json`, `data/analysis.json` | pass/fail of every check; argmax-mismatch attribution, overflow counts, worst-case diagnoses |
| `plots/*.png` | `required_inputs`, `error_distribution`, `error_by_length_and_range`, `output_scatter`, `error_vs_precision`, `equal_logit_sweep` |

Test set: the 5 required inputs, 20 boundary cases (z integer boundaries, 8-bit range limits,
dominant logit, near ties, large offsets, N = 1, 256/1024 equal logits, ...), and a seeded random grid:
lengths N in {4, 8, 16, 64, 256, 1024} x {uniform[-0.5,0.5], uniform[-2,2], uniform[-5,5], uniform[-10,10],
normal(0,3)} with 200/200/200/100/50/20 vectors per cell = **3850 vectors**. Each cell has its own
generator `SeedSequence([seed, N, distribution_index])`. Relative error is reported only where the
exact p >= 0.01. Argmax "match" = first-index argmax is a reference maximum (ties reported separately).

Snapshot of the run with seed 2026 (per-vector max abs error, 3850 vectors; see `report.md` for everything):

| comparison | p50 | p95 | p99 | max | mean abs (avg) | RMSE (avg) | argmax match |
|---|---|---|---|---|---|---|---|
| float Wang vs exact | 0.0099 | 0.0364 | 0.0467 | 0.0583 | 0.00316 | 0.00538 | 100.0% |
| fixed8 vs exact | 0.0111 | 0.0362 | 0.0512 | 0.1237 | 0.00365 | 0.00586 | 84.2% |
| fixed16 vs exact | 0.0098 | 0.0365 | 0.0470 | 0.0587 | 0.00315 | 0.00536 | 97.1% |
| fixed8 vs float Wang | 0.0053 | 0.0164 | 0.0268 | 0.1173 | 0.00218 | 0.00300 | 84.2% |
| fixed16 vs float Wang | 0.0003 | 0.0009 | 0.0028 | 0.0209 | 0.00011 | 0.00017 | 97.1% |

Evaluation checks (13, all pass in the run above): exact error identity of the float model, float
error inside an a-priori worst-case envelope, float strictly monotone/argmax-exact, fixed monotone with
only tie-type argmax disagreements, outputs within format, shift/permutation invariances.
Findings are explained in `report.md` (8-bit output resolution cliff for N >= 384 equal logits,
pass-1 truncation bias, quantization across the d > 1 discontinuity, argmax ties).
