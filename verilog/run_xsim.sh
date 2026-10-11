#!/bin/sh
# Generate vectors, compile, elaborate and run tb_softmax_top with Vivado xsim.
# Run from anywhere; requires Vivado's bin directory on PATH and python3 + numpy.
# Optional: WANG_PY=<path to wang_softmax_python>, extra gen_vectors.py args after "--".
set -e
cd "$(dirname "$0")/.."
python3 scripts/gen_vectors.py ${WANG_PY:+--model-dir "$WANG_PY"} "$@"
xvlog rtl/arith_common.v rtl/expu.v rtl/adder_tree.v rtl/lnu.v rtl/softmax_top.v tb/tb_softmax_top.v
xelab -debug typical tb_softmax_top -s tb_sim
xsim tb_sim -runall | tee xsim_run.log
grep -q "TB_RESULT: PASS" xsim_run.log
