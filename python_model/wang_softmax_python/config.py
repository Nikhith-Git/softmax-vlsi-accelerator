"""Central configuration. Every field says where its default comes from."""
from dataclasses import dataclass, field
from typing import Optional
import coefficients as C
import fixed_point as FP


@dataclass(frozen=True)
class FixedConfig:
    """Fixed-point settings (Stage 7).

    PAPER facts (only these): 8-bit and 16-bit input/output are supported
    (Sec. V-A); y = x - max is computed in software and the hardware receives
    fixed-point y plus N_in; constants 1.0111b and 0.1011b (Sec. III-C, IV-B).
    EVERYTHING ELSE below is an EXPERIMENTAL choice, not taken from the paper.

    Datapath formats (see format_table() for the full list of intermediates):
      t   = -y             UQ(data_width-input_frac_bits).input_frac_bits
                           (y <= 0, so the magnitude needs no sign bit)
      a, lnF               SQ.a_frac_bits      width a_width
      z = a * log2e        SQ.z_frac_bits      width z_width
      v, d, m = v + d      unsigned, z_frac_bits fraction bits
      EXPU out / p         UQ.exp_out_frac_bits, width exp_out_width
      F (accumulator)      UQ.acc_frac_bits,   width acc_width
      log2 approx, lnF*..  lnu_frac_bits fraction bits
    One global `rounding` and `overflow` policy applies to every stage.
    """
    data_width: int = 8                 # PAPER: 8 or 16 (input t width)
    input_frac_bits: int = 4            # EXPERIMENTAL (paper: N_in, unspecified)
    exp_out_frac_bits: int = 8          # EXPERIMENTAL
    acc_width: int = 24                 # EXPERIMENTAL (paper: unspecified)
    acc_frac_bits: int = 8              # EXPERIMENTAL
    rounding: str = "trunc"             # EXPERIMENTAL; "trunc" = floor (see fixed_point.py)
    overflow: str = "saturate"          # EXPERIMENTAL; paper unspecified
    exp_out_width: int = 9              # EXPERIMENTAL: UQ1.8 with the default frac bits
    a_width: int = 16                   # EXPERIMENTAL
    a_frac_bits: int = 8                # EXPERIMENTAL
    z_width: int = 16                   # EXPERIMENTAL
    z_frac_bits: int = 8                # EXPERIMENTAL
    lnu_frac_bits: int = 8              # EXPERIMENTAL: bits of (k-1) kept by the LNU
    const_frac_bits: int = 4            # width of the fraction of the PAPER constants
                                        # (1.0111b and 0.1011b both have 4)

    def __post_init__(self):
        for name in ("data_width", "input_frac_bits", "exp_out_frac_bits", "acc_width",
                     "acc_frac_bits", "exp_out_width", "a_width", "a_frac_bits",
                     "z_width", "z_frac_bits", "lnu_frac_bits", "const_frac_bits"):
            v = getattr(self, name)
            if not FP._is_int(v):
                raise ValueError(f"{name} must be an integer, got {v!r}")
        FP._canon_rounding(self.rounding)
        FP._check_overflow_mode(self.overflow)
        if self.lnu_frac_bits < 1 or self.const_frac_bits < 0:
            raise ValueError("lnu_frac_bits >= 1 and const_frac_bits >= 0 required")
        # Constructing the formats validates widths/fractions (QFormat raises).
        t, a = self.fmt_t, self.fmt_a
        if self.a_frac_bits < self.input_frac_bits:
            raise ValueError("a_frac_bits must be >= input_frac_bits (t is aligned "
                             "to the a-format without discarding bits)")
        # The interface must hold the whole input range: no silent clipping at t -> a.
        if a.int_bits < t.int_bits:
            raise ValueError("a-format has too few integer bits for the t input range")
        if self.acc_width < 2:
            raise ValueError("acc_width must be >= 2 (LOD needs width >= 2)")
        self.fmt_z, self.fmt_m, self.fmt_exp_out, self.fmt_acc  # validate

    # ---- formats -----------------------------------------------------------
    @property
    def fmt_t(self) -> FP.QFormat:
        return FP.QFormat(self.data_width, self.input_frac_bits, False)

    @property
    def fmt_a(self) -> FP.QFormat:
        return FP.QFormat(self.a_width, self.a_frac_bits, True)

    @property
    def fmt_z(self) -> FP.QFormat:
        return FP.QFormat(self.z_width, self.z_frac_bits, True)

    @property
    def fmt_v(self) -> FP.QFormat:
        return FP.QFormat(self.z_frac_bits, self.z_frac_bits, False)       # UQ0.zf

    @property
    def fmt_m(self) -> FP.QFormat:
        """d and m = v + d: 2 integer bits (needs 0 <= d < 2; then m < 3)."""
        return FP.QFormat(self.z_frac_bits + 2, self.z_frac_bits, False)

    @property
    def fmt_exp_out(self) -> FP.QFormat:
        return FP.QFormat(self.exp_out_width, self.exp_out_frac_bits, False)

    @property
    def fmt_acc(self) -> FP.QFormat:
        return FP.QFormat(self.acc_width, self.acc_frac_bits, False)

    @property
    def log2_width(self) -> int:
        """Signed width of (k - 1 + w) with lnu_frac_bits fraction bits.
        w in [-acc_frac_bits, acc_width-acc_frac_bits-1]; +1 for a rounding carry."""
        wmax = max(self.acc_frac_bits, self.acc_width - self.acc_frac_bits - 1) + 1
        return wmax.bit_length() + 1 + self.lnu_frac_bits

    @property
    def fmt_log2(self) -> FP.QFormat:
        return FP.QFormat(self.log2_width, self.lnu_frac_bits, True)

    def format_table(self):
        """Every intermediate: (name, QFormat or None, width, frac_bits, signed, note).
        Rows with fmt None are EXACT products/sums (no rounding, no overflow)."""
        cb = self.const_frac_bits
        mult_bits = (1 << (cb + 1)).bit_length()           # constants < 2.0 -> cb+1 bits
        rows = [
            ("t = -y (input magnitude)", self.fmt_t, "exact after quantize; rounding+overflow at input"),
            ("a = y - lnF", self.fmt_a, "adder #1; overflow policy applies"),
            ("lnF", self.fmt_a, "LNU output requantized to a-format"),
            ("z_exact = a*log2e_code", None, f"width {self.a_width + mult_bits}, frac {self.a_frac_bits + cb}; exact"),
            ("z = a*log2e (rounded)", self.fmt_z, "rounding + overflow here"),
            ("u = z >> z_frac (integer part)", FP.QFormat(self.z_width - self.z_frac_bits, 0, True), "arithmetic shift = floor"),
            ("v = z & mask (fraction, [0,1))", self.fmt_v, "exact bit slice"),
            ("d1, d2 (bias codes)", self.fmt_m, "quantized from float; rounding applies"),
            ("m = v + d", self.fmt_m, "adder #2; cannot overflow if 0 <= d < 2"),
            ("EXPU out = m * 2^u", self.fmt_exp_out, "shift with rounding + overflow"),
            ("lane value (acc-aligned)", None, f"frac {self.acc_frac_bits}; exact if acc_frac >= exp_out_frac"),
            ("adder-tree node", None, "exact; +1 bit per tree level, no overflow by construction"),
            ("F (accumulator)", self.fmt_acc, "overflow policy applies each cycle"),
            ("log2 approx = w*2^f + (k-1)", self.fmt_log2, "k-1 rounded to lnu_frac_bits"),
            ("ln2_code * log2 (exact)", None, f"frac {self.lnu_frac_bits + cb}; exact"),
            ("p (pass-2 output)", self.fmt_exp_out, "same format as pass-1 EXPU out"),
        ]
        out = []
        for name, fmt, note in rows:
            if fmt is None:
                out.append((name, None, None, None, None, note))
            else:
                out.append((name, fmt, fmt.width, fmt.frac_bits, fmt.signed, note))
        return out

    def describe(self) -> str:
        lines = [f"rounding={self.rounding}  overflow={self.overflow}   (ALL EXPERIMENTAL)",
                 f"{'intermediate':34s} {'format':8s} {'width':>5s} {'frac':>4s}  {'lsb':>10s}  note"]
        for name, fmt, w, f, s, note in self.format_table():
            if fmt is None:
                lines.append(f"{name:34s} {'exact':8s} {'-':>5s} {'-':>4s}  {'-':>10s}  {note}")
            else:
                lines.append(f"{name:34s} {fmt.name:8s} {w:5d} {f:4d}  {fmt.lsb:10.3g}  {note}")
        return "\n".join(lines)


@dataclass(frozen=True)
class WangConfig:
    # Bias values: PROVISIONAL (not in paper). Separate by design.
    d1: float = C.D1.value
    d2: float = C.D2.value
    # Architecture constants: PAPER values by default.
    log2e_const: float = C.LOG2E_ARCH.value
    ln2_const: float = C.LN2_ARCH.value
    # Pass-1 lnF input: ASSUMED 0 (Fig. 2 shows lnF entering, paper silent).
    pass1_ln_f: float = 0.0
    # Parallel lanes (PAPER Sec. V-A: 16 EXPUs for 8-bit, 8 for 16-bit).
    lanes: int = 16
    # Optional extra renormalisation: OFF; NOT in paper. Experiment only.
    renormalize_output: bool = False
    # None -> floating-point architecture model; FixedConfig -> Stage 7.
    fixed: Optional[FixedConfig] = None


# ---- Named fixed-point presets: ALL EXPERIMENTAL, none comes from the paper ----
FIXED_8BIT = FixedConfig()                                   # 8-bit input t (UQ4.4)
FIXED_16BIT = FixedConfig(data_width=16, input_frac_bits=8, exp_out_frac_bits=15,
                          exp_out_width=16, acc_width=32, acc_frac_bits=15,
                          a_width=24, a_frac_bits=15, z_width=24, z_frac_bits=15,
                          lnu_frac_bits=15)                  # 16-bit input t (UQ8.8)
# Very wide: used only to show the fixed model converges to the float model.
FIXED_WIDE_REFERENCE = FixedConfig(data_width=16, input_frac_bits=8, exp_out_frac_bits=30,
                                   exp_out_width=32, acc_width=48, acc_frac_bits=30,
                                   a_width=40, a_frac_bits=24, z_width=40, z_frac_bits=24,
                                   lnu_frac_bits=24)
PRESETS = {"8bit": FIXED_8BIT, "16bit": FIXED_16BIT, "wide": FIXED_WIDE_REFERENCE}
