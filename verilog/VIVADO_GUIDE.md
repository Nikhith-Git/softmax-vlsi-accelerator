# Vivado run guide (softmax_top, 16-bit preset)

Layout: `rtl/` (5 files), `tb/tb_softmax_top.v`, `scripts/`, `docs/`, `vectors/` (generated).
The Python model folder `wang_softmax_python/` must be reachable (see below).

## 1. Golden vectors
    python3 scripts/gen_vectors.py --model-dir <path>/wang_softmax_python
Options: `--d1/--d2` (floats), `--rounding trunc|toward_zero|round_half_up|round_half_even`,
`--overflow saturate|wrap`, `--max-n`. It prints the matching RTL parameters
(`ROUNDING OVERFLOW D1_CODE D2_CODE MAX_N`, also in `vectors/params.txt`).
Defaults match the RTL defaults (D1=38010, D2=35471, trunc, saturate, MAX_N=1024).

## 2. Simulate (xsim, run from the project root)
    scripts/run_xsim.sh            # or the three commands inside it
    xvlog rtl/arith_common.v rtl/expu.v rtl/adder_tree.v rtl/lnu.v rtl/softmax_top.v tb/tb_softmax_top.v
    xelab -debug typical tb_softmax_top -s tb_sim
    xsim tb_sim -runall
Pass criterion: the log ends with `TB_RESULT: PASS`. Any `ERROR` line names the run
(0 = no stalls, 1 = random back-pressure), the case index (see `vectors/cases.txt`), and
the first mismatching lane. Non-default arithmetic: pass the same values to both sides, e.g.
`python3 scripts/gen_vectors.py --rounding round_half_up` and
`xelab ... -generic_top ROUNDING=2` (OVERFLOW=1 for wrap, D1_CODE/D2_CODE/MAX_N likewise).

## 3. Synthesize (out-of-context)
    PART=<fpga-part> CLK_NS=10.0 vivado -mode batch -source scripts/synth.tcl
Reports go to `reports/` (utilization, timing, DRC). The part and clock are your choice; record
Vivado version, part and clock with every report. For post-implementation timing add
`opt_design / place_design / route_design` after `synth_design`.

## Notes
- Interface: `start`+`n_elems` (accepted when `busy=0`), 256-bit input beats of 16 x UQ8.8 `t`
  (`in_valid/in_ready`, ceil(n/16) beats), 256-bit output beats of 16 x UQ1.15
  (`out_valid/out_ready/out_last/out_keep`), sticky `status[7:0]` (bit map in `softmax_top.v`).
- `status[3]` (accumulator), `[4]` (LNU) and `[2]` (EXPU output) overflow are unreachable with the
  default `d1` and valid inputs, so the testbench only checks that they stay 0; `[0]`,`[1]`,`[5]`,`[6]`,`[7]`
  are exercised by directed cases.
