"""Adder tree + accumulator register (paper Fig. 1).

Numerical summation vs hardware adder tree
------------------------------------------
* numpy_sum: whatever order/blocking numpy chooses; a mathematical reference.
* balanced_tree_sum: a FIXED order of additions: level 0 holds the N inputs,
  each next level adds neighbouring pairs, depth = ceil(log2 N) adders in
  series.  An odd leftover is passed up unchanged (hardware: its partner is 0).
* accumulate_chunks: the structure of Fig. 1.  `lanes` EXPU outputs arrive per
  cycle (16 for 8-bit data, paper Sec. V-A); a tree adds them, then an adder
  and the register F accumulate across cycles.

In floating point the three orders differ only by rounding (~1e-16 relative).
In integer fixed point, addition is exactly associative until overflow, so the
orders agree bit-for-bit and only per-node width/rounding choices matter.
The finite-width accumulator model (Stage 7) is `accumulate_chunks_fixed`:
integer codes, exact tree (one extra bit per level), accumulator register of
acc_width bits with the configured saturate/wrap.  The float function below
refuses a fixed-point cfg rather than silently using an unbounded accumulator.
"""
import numpy as np
import fixed_point as FP


def _validate(values) -> np.ndarray:
    arr = np.array(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("values must be 1-D")
    if arr.size == 0:
        raise ValueError("values must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite")
    return arr


def numpy_sum(values) -> float:
    """Reference: plain numerical summation."""
    return float(np.sum(_validate(values)))


def balanced_tree_sum(values, return_levels: bool = False):
    """Balanced binary adder-tree model.  Returns the total, and with
    return_levels=True also the list of arrays, leaf level first."""
    level = _validate(values)
    levels = [level]
    while level.size > 1:
        n = level.size
        paired = level[: n - n % 2].reshape(-1, 2).sum(axis=1)    # one adder level
        level = np.concatenate([paired, level[n - 1:]]) if n % 2 else paired
        levels.append(level)
    total = float(level[0])
    return (total, levels) if return_levels else total


def accumulate_chunks(values, lanes: int = 16, cfg=None, return_trace: bool = False):
    """Fig. 1 structure: per cycle a `lanes`-wide tree sum feeds the register F.

    The last chunk is zero-padded (idle lanes add 0).  Returns F, or with
    return_trace=True a dict with the per-cycle tree sums and running F.
    """
    if cfg is not None and getattr(cfg, "fixed", None) is not None:
        raise ValueError("cfg.fixed is set: float accumulate_chunks takes floats; "
                         "use accumulate_chunks_fixed on integer codes")
    if not isinstance(lanes, (int, np.integer)) or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    arr = _validate(values)
    pad = (-arr.size) % lanes
    chunks = np.concatenate([arr, np.zeros(pad)]).reshape(-1, lanes)
    chunk_sums = np.array([balanced_tree_sum(c) for c in chunks])
    F = 0.0
    running = []
    for s in chunk_sums:                       # accumulator register, one add / cycle
        F = F + float(s)
        running.append(F)
    if return_trace:
        return {"F": F, "chunk_sums": chunk_sums, "running": np.array(running),
                "cycles": len(chunk_sums)}
    return F


# ------------------------- Stage 7: integer accumulator ----------------------
def balanced_tree_sum_int(codes) -> int:
    """Exact integer adder tree (same pairing as balanced_tree_sum).  Python
    ints: no overflow; in hardware each level is one bit wider."""
    level = [int(c) for c in codes]
    if not level:
        raise ValueError("codes must not be empty")
    while len(level) > 1:
        paired = [level[i] + level[i + 1] for i in range(0, len(level) - 1, 2)]
        level = paired + ([level[-1]] if len(level) % 2 else [])
    return level[0]


def accumulate_chunks_fixed(codes, lanes: int, fx, return_trace: bool = False):
    """Fixed-point Fig. 1.  codes: 1-D integer EXPU outputs in fx.fmt_exp_out.

      lane  = code aligned to acc_frac_bits (exact left shift, or right shift
              with fx.rounding if acc_frac < exp_out_frac)
      tree  = exact sum of `lanes` lane values (no overflow: width grows)
      F     = fit(F + tree, UQ acc_width.acc_frac)   once per cycle, using
              fx.overflow (saturate or wrap); starts at 0.
    Returns the F code (int), or with return_trace a dict that also reports
    `overflow` (any accumulator step left the range) and `tree_bits`.
    """
    if not isinstance(lanes, (int, np.integer)) or isinstance(lanes, bool) or lanes < 1:
        raise ValueError("lanes must be a positive integer")
    arr = np.asarray(codes)
    if arr.ndim != 1 or arr.size == 0 or arr.dtype.kind not in "iu":
        raise ValueError("codes must be a non-empty 1-D integer array")
    if np.any(FP.would_overflow(arr, fx.fmt_exp_out)):
        raise ValueError("codes outside the EXPU output format")
    shift = fx.exp_out_frac_bits - fx.acc_frac_bits
    lane_vals = [FP.rshift_round(int(c), shift, fx.rounding) for c in arr]
    lane_vals += [0] * ((-len(lane_vals)) % lanes)
    F, running, chunk_sums, overflow, tree_bits = 0, [], [], False, 0
    for i in range(0, len(lane_vals), lanes):
        s = balanced_tree_sum_int(lane_vals[i:i + lanes])
        chunk_sums.append(s)
        tree_bits = max(tree_bits, s.bit_length())
        exact = F + s
        overflow |= bool(FP.would_overflow(exact, fx.fmt_acc))
        F = int(FP.saturate_or_wrap(exact, fx.fmt_acc, fx.overflow))
        running.append(F)
    if return_trace:
        return {"F": F, "chunk_sums": chunk_sums, "running": running,
                "cycles": len(chunk_sums), "overflow": overflow, "tree_bits": tree_bits}
    return F
