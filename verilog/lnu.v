`timescale 1ns/1ps
// =============================================================================
// lnu.v : natural-log unit (paper Fig. 4, Eqs. 5-8), 3-stage pipeline,
//         latency = 3 clocks, no back-pressure.
//
//   F = 2^w * k, k in [1,2): LOD gives msb; w = msb - ACC_F;
//   kf = bits of F below the leading one, normalised (ACC_W-1 bits) = k-1
//   kf_r   = rshift_round(kf, ACC_W-1-LNU_F)            (rounding carry allowed)
//   log2c  = (w << LNU_F) + kf_r                        SQ.LNU_F, WW+LNU_F bits
//   prod   = log2c * 0.1011b = log2c*11 = (c<<3)+(c<<1)+c   exact, +4 bits
//   ln_f   = fit( rshift_round(prod, LNU_F+4-A_F) )     SQ.A_F in A_W bits
//
// F == 0 : f_zero=1, ln_f forced to 0, ovf=0 (the Python model raises here).
// Default 16-bit preset: F UQ17.15 (32b), log2c SQ5.15 (21b), prod 25b,
// ln_f SQ8.15 (24b). Requires LNU_F+4 >= A_F and ACC_W-1 >= LNU_F.
// Reset is synchronous, active high; clears valid bits only.
// =============================================================================
module lnu #(
    parameter ACC_W    = 32,
    parameter ACC_F    = 15,
    parameter LNU_F    = 15,
    parameter A_W      = 24,
    parameter A_F      = 15,
    parameter ROUNDING = 0,
    parameter OVERFLOW = 0
)(
    input                       clk,
    input                       rst,
    input                       in_valid,
    input      [ACC_W-1:0]      f_code,      // UQ.ACC_F
    output reg                  out_valid,
    output reg [A_W-1:0]        ln_f_code,   // SQ.A_F
    output reg                  ovf,         // ln_f left the A-format before sat/wrap
    output reg                  f_zero       // F was zero
);
    localparam MSB_W  = $clog2(ACC_W);
    localparam WW     = MSB_W + 1;                 // signed width of w
    localparam KF_W   = ACC_W - 1;
    localparam LOG2_W = WW + LNU_F;
    localparam PROD_W = LOG2_W + 4;
    localparam P_SHIFT = LNU_F + 4 - A_F;

    // ---------------- stage 1: LOD + normalise ----------------
    reg [MSB_W-1:0] msb;
    integer i;
    always @* begin
        msb = {MSB_W{1'b0}};
        for (i = 0; i < ACC_W; i = i + 1)
            if (f_code[i]) msb = i;
    end
    wire [ACC_W-1:0] norm = f_code << ((ACC_W - 1) - msb);
    wire signed [WW-1:0] w_n = $signed({1'b0, msb}) - ACC_F;

    reg              s1_valid, s1_zero;
    reg signed [WW-1:0] s1_w;
    reg [KF_W-1:0]   s1_kf;

    // ---------------- stage 2: kf rounding, log2c, *11 ----------------
    wire [KF_W-1:0] kf_r;
    rshift_round #(.IN_W(KF_W), .SHIFT(KF_W - LNU_F), .IN_SIGNED(0), .ROUNDING(ROUNDING))
        u_kf (.x(s1_kf), .y(kf_r));
    wire signed [LOG2_W-1:0] w_ext  = {{(LOG2_W-WW){s1_w[WW-1]}}, s1_w};
    wire signed [LOG2_W-1:0] w_sh   = w_ext <<< LNU_F;
    wire signed [LOG2_W-1:0] kf_ext = {{(LOG2_W-LNU_F-1){1'b0}}, kf_r[LNU_F:0]};
    wire signed [LOG2_W-1:0] log2c  = w_sh + kf_ext;
    wire signed [PROD_W-1:0] c_x    = {{(PROD_W-LOG2_W){log2c[LOG2_W-1]}}, log2c};
    wire signed [PROD_W-1:0] prod_n = (c_x <<< 3) + (c_x <<< 1) + c_x;

    reg              s2_valid, s2_zero;
    reg signed [PROD_W-1:0] s2_prod;

    // ---------------- stage 3: >> P_SHIFT, fit ----------------
    wire [PROD_W-1:0] p_r;
    rshift_round #(.IN_W(PROD_W), .SHIFT(P_SHIFT), .IN_SIGNED(1), .ROUNDING(ROUNDING))
        u_p (.x(s2_prod), .y(p_r));
    wire [A_W-1:0] ln_fit;
    wire           ln_ovf;
    fit_sat_wrap #(.IN_W(PROD_W), .OUT_W(A_W), .OUT_SIGNED(1), .OVERFLOW(OVERFLOW))
        u_fit (.x(p_r), .y(ln_fit), .ovf(ln_ovf));

    always @(posedge clk) begin
        if (rst) begin
            s1_valid <= 1'b0; s2_valid <= 1'b0; out_valid <= 1'b0;
        end else begin
            s1_valid  <= in_valid;
            s2_valid  <= s1_valid;
            out_valid <= s2_valid;
        end
    end

    always @(posedge clk) begin
        s1_zero <= ~(|f_code);
        s1_w    <= w_n;
        s1_kf   <= norm[KF_W-1:0];

        s2_zero <= s1_zero;
        s2_prod <= prod_n;

        f_zero    <= s2_zero;
        ln_f_code <= s2_zero ? {A_W{1'b0}} : ln_fit;
        ovf       <= s2_zero ? 1'b0        : ln_ovf;
    end
endmodule
