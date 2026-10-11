`timescale 1ns/1ps
// =============================================================================
// adder_tree.v : 16-lane adder tree + accumulator register F (paper Fig. 1).
//
//   lane value = aligned EXPU output (UQ.LANE_F -> UQ.ACC_F, rshift_round when
//                LANE_F > ACC_F; LANE_F >= ACC_F required), forced to 0 when
//                lane_keep=0 (zero-padding of the last beat)
//   tree       = exact sum of 16 lane values (+1 bit per level, LANE_W+4 bits)
//   F          = fit( F + tree ) once per valid beat, UQ.ACC_F in ACC_W bits
//
// Pipeline: [levels 1-2] reg -> [levels 3-4] reg -> [accumulate] reg.
// A beat presented at in_valid is included in f_code 3 clocks later;
// acc_done pulses in that same clock when that beat had in_last=1.
// acc_ovf pulses when that accumulation left the ACC_W-bit range (before
// saturate/wrap). acc_clear (synchronous) zeroes F; it has priority over
// accumulation and must not coincide with an in-flight beat.
// No back-pressure: the tree always accepts a beat. Reset is synchronous,
// active high; data registers other than F are not reset (valid-qualified).
// LANES is fixed at 16 by the explicit tree structure.
// =============================================================================
module adder_tree #(
    parameter LANE_W   = 16,
    parameter LANE_F   = 15,
    parameter ACC_W    = 32,
    parameter ACC_F    = 15,
    parameter ROUNDING = 0,
    parameter OVERFLOW = 0
)(
    input                      clk,
    input                      rst,
    input                      acc_clear,
    input                      in_valid,
    input                      in_last,
    input  [16*LANE_W-1:0]     lane_in,      // 16 x UQ.LANE_F, lane i at [i*LANE_W +: LANE_W]
    input  [15:0]              lane_keep,
    output reg [ACC_W-1:0]     f_code,       // UQ.ACC_F
    output reg                 acc_done,
    output reg                 acc_ovf
);
    localparam LANES      = 16;
    localparam LANE_SHIFT = LANE_F - ACC_F;
    localparam W1 = LANE_W + 1;
    localparam W2 = LANE_W + 2;
    localparam W3 = LANE_W + 3;
    localparam W4 = LANE_W + 4;
    localparam SUM_W = ACC_W + 2;            // signed (zero-extended) sum for fit

    // ---- lane alignment + masking ----
    wire [LANES*LANE_W-1:0] lv;
    genvar gi;
    generate
        for (gi = 0; gi < LANES; gi = gi + 1) begin : g_lane
            wire [LANE_W-1:0] al;
            rshift_round #(.IN_W(LANE_W), .SHIFT(LANE_SHIFT), .IN_SIGNED(0), .ROUNDING(ROUNDING))
                u_al (.x(lane_in[gi*LANE_W +: LANE_W]), .y(al));
            assign lv[gi*LANE_W +: LANE_W] = lane_keep[gi] ? al : {LANE_W{1'b0}};
        end
    endgenerate

    // ---- levels 1 and 2 ----
    wire [8*W1-1:0] l1;
    wire [4*W2-1:0] l2;
    generate
        for (gi = 0; gi < 8; gi = gi + 1) begin : g_l1
            assign l1[gi*W1 +: W1] = {1'b0, lv[(2*gi)*LANE_W +: LANE_W]}
                                   + {1'b0, lv[(2*gi+1)*LANE_W +: LANE_W]};
        end
        for (gi = 0; gi < 4; gi = gi + 1) begin : g_l2
            assign l2[gi*W2 +: W2] = {1'b0, l1[(2*gi)*W1 +: W1]}
                                   + {1'b0, l1[(2*gi+1)*W1 +: W1]};
        end
    endgenerate

    reg [4*W2-1:0] r2;
    reg            r2_valid, r2_last;

    // ---- levels 3 and 4 ----
    wire [2*W3-1:0] l3;
    wire [W4-1:0]   l4;
    generate
        for (gi = 0; gi < 2; gi = gi + 1) begin : g_l3
            assign l3[gi*W3 +: W3] = {1'b0, r2[(2*gi)*W2 +: W2]}
                                   + {1'b0, r2[(2*gi+1)*W2 +: W2]};
        end
    endgenerate
    assign l4 = {1'b0, l3[0 +: W3]} + {1'b0, l3[W3 +: W3]};

    reg [W4-1:0] tsum;
    reg          t_valid, t_last;

    // ---- accumulate: F = fit(F + tsum) ----
    wire [SUM_W-1:0] acc_exact = {2'b00, f_code} + {{(SUM_W-W4){1'b0}}, tsum};
    wire [ACC_W-1:0] acc_fit;
    wire             acc_o;
    fit_sat_wrap #(.IN_W(SUM_W), .OUT_W(ACC_W), .OUT_SIGNED(0), .OVERFLOW(OVERFLOW))
        u_fit_acc (.x(acc_exact), .y(acc_fit), .ovf(acc_o));

    always @(posedge clk) begin
        if (rst) begin
            r2_valid <= 1'b0; r2_last <= 1'b0;
            t_valid  <= 1'b0; t_last  <= 1'b0;
            f_code   <= {ACC_W{1'b0}};
            acc_done <= 1'b0; acc_ovf <= 1'b0;
        end else begin
            r2_valid <= in_valid;
            r2_last  <= in_last;
            t_valid  <= r2_valid;
            t_last   <= r2_last;
            acc_done <= t_valid & t_last & ~acc_clear;
            acc_ovf  <= t_valid & acc_o & ~acc_clear;
            if (acc_clear)
                f_code <= {ACC_W{1'b0}};
            else if (t_valid)
                f_code <= acc_fit;
        end
    end

    always @(posedge clk) begin
        r2   <= l2;
        tsum <= l4;
    end
endmodule
