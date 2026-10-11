`timescale 1ns/1ps
// =============================================================================
// arith_common.v : common fixed-point helpers (Verilog-2001, Vivado compatible)
//
// ROUNDING codes (global policy, same meaning as fixed_point.py):
//   0 trunc/floor (arithmetic shift)   1 toward_zero
//   2 round_half_up                    3 round_half_even
// OVERFLOW codes: 0 saturate, 1 wrap (keep low bits)
//
// All values are two's-complement integer codes; binary-point handling is done
// by the instantiating module. Purely combinational.
// =============================================================================

// -----------------------------------------------------------------------------
// rshift_round : constant right shift by SHIFT with rounding.
//   x, y are IN_W bits wide, interpreted signed (IN_SIGNED=1) or unsigned.
//   Requires 0 <= SHIFT <= IN_W. Result always fits IN_W bits (SHIFT>=1 halves
//   the magnitude, so a rounding carry cannot overflow). SHIFT=0 is a wire.
// -----------------------------------------------------------------------------
module rshift_round #(
    parameter IN_W      = 16,
    parameter SHIFT     = 1,
    parameter IN_SIGNED = 1,
    parameter ROUNDING  = 0
)(
    input  [IN_W-1:0] x,
    output [IN_W-1:0] y
);
generate
if (SHIFT == 0) begin : g_id
    assign y = x;
end else begin : g_sh
    localparam EW = IN_W + 1;
    wire signed [EW-1:0] xe = IN_SIGNED ? {x[IN_W-1], x} : {1'b0, x};
    wire signed [EW-1:0] q  = xe >>> SHIFT;                 // floor
    wire half   = xe[SHIFT-1];                              // first discarded bit
    wire sticky = |(xe & ({EW{1'b1}} >> (EW - (SHIFT - 1)))); // bits below 'half'
    wire nz_low = |(xe & ({EW{1'b1}} >> (EW - SHIFT)));     // any discarded bit
    wire neg    = IN_SIGNED & x[IN_W-1];
    reg  inc;
    always @* begin
        case (ROUNDING)
            1:       inc = neg & nz_low;                    // toward zero
            2:       inc = half;                            // half up
            3:       inc = half & (sticky | q[0]);          // half even
            default: inc = 1'b0;                            // trunc
        endcase
    end
    wire [EW-1:0] r = q + {{(EW-1){1'b0}}, inc};
    assign y = r[IN_W-1:0];
end
endgenerate
endmodule

// -----------------------------------------------------------------------------
// rshift_round_var : UNSIGNED right shift by a runtime amount with exact
//   mathematical rounding for every mode (ROUNDING 1 == 0 for unsigned).
//   There is NO clamp at the data width: sh == W can round up to 1
//   (half_up/half_even). The result is 0 for every mode once sh >= W+1,
//   which falls out of the shifts below. y is W bits (rounding carry cannot
//   overflow because sh>=1 halves the magnitude).
// -----------------------------------------------------------------------------
module rshift_round_var #(
    parameter W        = 17,
    parameter SH_W     = 10,
    parameter ROUNDING = 0
)(
    input  [W-1:0]    x,
    input  [SH_W-1:0] sh,
    output [W-1:0]    y
);
    wire [W:0]      ext      = {x, 1'b0};
    wire [W:0]      t        = ext >> sh;                   // t[0]=half bit, t[W:1]=floor(x/2^sh)
    wire [W-1:0]    q        = t[W:1];
    wire            half     = t[0];
    wire [SH_W-1:0] shm1     = sh - 1'b1;
    wire [W-1:0]    lowmask  = ~({W{1'b1}} << shm1);        // bits below the half bit
    wire            sticky   = |(x & lowmask);
    reg inc;
    always @* begin
        case (ROUNDING)
            2:       inc = half;
            3:       inc = half & (sticky | q[0]);
            default: inc = 1'b0;
        endcase
    end
    assign y = (sh == {SH_W{1'b0}}) ? x : (q + {{(W-1){1'b0}}, inc});
endmodule

// -----------------------------------------------------------------------------
// fit_sat_wrap : bring a wide SIGNED two's-complement value x (IN_W bits) into
//   an OUT_W-bit format (signed if OUT_SIGNED, else unsigned).
//   Requires IN_W > OUT_W. ovf=1 when x is outside the output range (flag is
//   reported for both saturate and wrap). Unsigned output: negative x
//   saturates to 0.
// -----------------------------------------------------------------------------
module fit_sat_wrap #(
    parameter IN_W       = 25,
    parameter OUT_W      = 24,
    parameter OUT_SIGNED = 1,
    parameter OVERFLOW   = 0
)(
    input  [IN_W-1:0]  x,
    output [OUT_W-1:0] y,
    output             ovf
);
    localparam HI_W = IN_W - OUT_W + (OUT_SIGNED ? 1 : 0);
    wire [HI_W-1:0] hi   = x[IN_W-1 -: HI_W];
    wire            neg  = x[IN_W-1];
    wire            fits = OUT_SIGNED ? ((&hi) | ~(|hi)) : ~(|hi);
    wire [OUT_W-1:0] satv = OUT_SIGNED ?
                            (neg ? {1'b1, {(OUT_W-1){1'b0}}} : {1'b0, {(OUT_W-1){1'b1}}}) :
                            (neg ? {OUT_W{1'b0}} : {OUT_W{1'b1}});
    assign ovf = ~fits;
    assign y   = (fits | (OVERFLOW != 0)) ? x[OUT_W-1:0] : satv;
endmodule
