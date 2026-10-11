`timescale 1ns/1ps
// =============================================================================
// expu.v : one EXPU lane (paper Fig. 2), 4-stage pipeline, latency = 4 clocks.
//
//   a = y - lnF                 SQ(A_W-A_F-1).A_F, fit (sat/wrap)
//   z = a * 1.0111b             1.0111b = 23/16 -> a*23 = (a<<4)+(a<<2)+(a<<1)+a
//                               exact (A_W+5 bits), >> (A_F+4-Z_F) with rounding, fit
//   u = floor(z) = z[Z_W-1:Z_F] (signed), v = z[Z_F-1:0] (UQ0.Z_F)
//   m = v + d                   UQ2.Z_F, requires d < 2.0 (d_code <= 2^(Z_F+1)-1)
//   out = m * 2^u               shift by u+(OUT_F-Z_F): left saturates/wraps,
//                               right rounds exactly (no clamp at the data width)
//
// Default 16-bit preset: A SQ8.15 (24b), Z SQ8.15 (24b), d/m UQ2.15 (17b),
// u SQ8.0 (9b), OUT UQ1.15 (16b).
//
// en=0 freezes every pipeline register (global back-pressure). Reset is
// synchronous, active high, and clears only the valid bit; data registers are
// not reset and must be qualified by out_valid. ovf_* are registered with the
// output beat (overflow before saturate/wrap, per stage).
// in_sb/out_sb: opaque side-band bits carried with the beat.
// =============================================================================
module expu #(
    parameter A_W      = 24,
    parameter A_F      = 15,
    parameter Z_W      = 24,
    parameter Z_F      = 15,
    parameter OUT_W    = 16,
    parameter OUT_F    = 15,
    parameter ROUNDING = 0,
    parameter OVERFLOW = 0,
    parameter SB_W     = 1
)(
    input                       clk,
    input                       rst,
    input                       en,
    input                       in_valid,
    input      [SB_W-1:0]       in_sb,
    input  signed [A_W-1:0]     y_code,      // SQ.A_F
    input  signed [A_W-1:0]     ln_f_code,   // SQ.A_F
    input      [Z_F+1:0]        d_code,      // UQ2.Z_F, d < 2.0
    output reg                  out_valid,
    output reg [SB_W-1:0]       out_sb,
    output reg [OUT_W-1:0]      out_code,    // UQ.OUT_F
    output reg                  ovf_a,
    output reg                  ovf_z,
    output reg                  ovf_out
);
    localparam D_W     = Z_F + 2;
    localparam Z_EX_W  = A_W + 5;            // a * 23 (23 needs 5 bits)
    localparam Z_SHIFT = A_F + 4 - Z_F;      // must be >= 0
    localparam U_W     = Z_W - Z_F;
    localparam SH_W    = U_W + 2;
    localparam LSW     = SH_W - 1;
    localparam LW      = D_W + OUT_W;

    // ---------------- stage 1: a = y - lnF ----------------
    wire signed [A_W:0] a_exact = y_code - ln_f_code;
    wire [A_W-1:0] a_fit;
    wire           a_ovf;
    fit_sat_wrap #(.IN_W(A_W+1), .OUT_W(A_W), .OUT_SIGNED(1), .OVERFLOW(OVERFLOW))
        u_fit_a (.x(a_exact), .y(a_fit), .ovf(a_ovf));

    reg              s1_valid;
    reg [SB_W-1:0]   s1_sb;
    reg [A_W-1:0]    s1_a;
    reg              s1_ova;
    reg [D_W-1:0]    s1_d;

    // ---------------- stage 2: z = a * 1.0111b ----------------
    wire signed [Z_EX_W-1:0] a_x = {{(Z_EX_W-A_W){s1_a[A_W-1]}}, s1_a};
    wire signed [Z_EX_W-1:0] z_exact = (a_x <<< 4) + (a_x <<< 2) + (a_x <<< 1) + a_x;
    wire [Z_EX_W-1:0] z_r;
    rshift_round #(.IN_W(Z_EX_W), .SHIFT(Z_SHIFT), .IN_SIGNED(1), .ROUNDING(ROUNDING))
        u_rz (.x(z_exact), .y(z_r));
    wire [Z_W-1:0] z_fit;
    wire           z_ovf;
    fit_sat_wrap #(.IN_W(Z_EX_W), .OUT_W(Z_W), .OUT_SIGNED(1), .OVERFLOW(OVERFLOW))
        u_fit_z (.x(z_r), .y(z_fit), .ovf(z_ovf));

    reg              s2_valid;
    reg [SB_W-1:0]   s2_sb;
    reg [Z_W-1:0]    s2_z;
    reg              s2_ova, s2_ovz;
    reg [D_W-1:0]    s2_d;

    // ---------------- stage 3: split + m = v + d ----------------
    wire [D_W-1:0] m_w = {{(D_W-Z_F){1'b0}}, s2_z[Z_F-1:0]} + s2_d;   // < 3.0, no overflow

    reg              s3_valid;
    reg [SB_W-1:0]   s3_sb;
    reg [D_W-1:0]    s3_m;
    reg [U_W-1:0]    s3_u;                   // signed integer part (floor)
    reg              s3_ova, s3_ovz;

    // ---------------- stage 4: out = m * 2^(u + OUT_F - Z_F) ----------------
    wire signed [SH_W-1:0] sh_s =
        $signed({{(SH_W-U_W){s3_u[U_W-1]}}, s3_u}) + (OUT_F - Z_F);
    wire is_left = ~sh_s[SH_W-1];

    // left shift (sh >= 0): exact for sh < OUT_W, otherwise overflow if m != 0
    wire [LSW-1:0] lsh   = sh_s[LSW-1:0];
    wire           big   = (lsh >= OUT_W);
    wire [LSW-1:0] lsh_c = big ? {LSW{1'b0}} : lsh;
    wire [LW-1:0]  exact_l = {{(LW-D_W){1'b0}}, s3_m} << lsh_c;
    wire [OUT_W-1:0] yl_fit;
    wire             ol_fit;
    fit_sat_wrap #(.IN_W(LW+1), .OUT_W(OUT_W), .OUT_SIGNED(0), .OVERFLOW(OVERFLOW))
        u_fit_l (.x({1'b0, exact_l}), .y(yl_fit), .ovf(ol_fit));
    wire m_nz = |s3_m;
    wire [OUT_W-1:0] y_l = big ? ((m_nz && (OVERFLOW == 0)) ? {OUT_W{1'b1}} : {OUT_W{1'b0}})
                               : yl_fit;
    wire             o_l = big ? m_nz : ol_fit;

    // right shift (sh < 0): exact rounding, amount = -sh
    wire [SH_W-1:0] rs_amt = {SH_W{1'b0}} - sh_s;
    wire [D_W-1:0]  q_r;
    rshift_round_var #(.W(D_W), .SH_W(SH_W), .ROUNDING(ROUNDING))
        u_rr (.x(s3_m), .sh(rs_amt), .y(q_r));
    wire [OUT_W-1:0] yr_fit;
    wire             or_fit;
    fit_sat_wrap #(.IN_W(D_W+1), .OUT_W(OUT_W), .OUT_SIGNED(0), .OVERFLOW(OVERFLOW))
        u_fit_r (.x({1'b0, q_r}), .y(yr_fit), .ovf(or_fit));

    wire [OUT_W-1:0] out_n = is_left ? y_l : yr_fit;
    wire             ovf_n = is_left ? o_l : or_fit;

    // ---------------- valid chain (reset) ----------------
    always @(posedge clk) begin
        if (rst) begin
            s1_valid <= 1'b0; s2_valid <= 1'b0; s3_valid <= 1'b0; out_valid <= 1'b0;
        end else if (en) begin
            s1_valid  <= in_valid;
            s2_valid  <= s1_valid;
            s3_valid  <= s2_valid;
            out_valid <= s3_valid;
        end
    end

    // ---------------- data chain (no reset) ----------------
    always @(posedge clk) begin
        if (en) begin
            s1_sb  <= in_sb;       s1_a <= a_fit;   s1_ova <= a_ovf;  s1_d <= d_code;
            s2_sb  <= s1_sb;       s2_z <= z_fit;   s2_ova <= s1_ova; s2_ovz <= z_ovf;  s2_d <= s1_d;
            s3_sb  <= s2_sb;       s3_m <= m_w;     s3_u <= s2_z[Z_W-1:Z_F];
            s3_ova <= s2_ova;      s3_ovz <= s2_ovz;
            out_sb   <= s3_sb;     out_code <= out_n;
            ovf_a    <= s3_ova;    ovf_z    <= s3_ovz;  ovf_out <= ovf_n;
        end
    end
endmodule
