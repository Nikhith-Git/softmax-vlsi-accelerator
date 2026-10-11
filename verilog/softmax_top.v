`timescale 1ns/1ps
// =============================================================================
// softmax_top.v : Wang-inspired softmax, 16-bit preset, 16 shared EXPU lanes.
//
// Flow per vector (one vector at a time, no overlap):
//   start+n_elems -> PASS1: stream t beats in, EXPU(d1, lnF=0) -> adder tree -> F
//                    (beats also stored in the t buffer)
//                 -> LNU(F) -> lnF
//                 -> PASS2: replay t buffer, EXPU(d2, lnF) -> output stream
//
// Data formats (16-bit preset)
//   in_data   : 16 lanes x UQ8.8 (16b), t = -(x - xmax) >= 0, lane i at [16i+:16].
//               Software contract: at least one element of the vector is 0.
//               Elements beyond n_elems in the last beat are ignored (any value).
//   out_data  : 16 lanes x UQ1.15 (16b), not renormalised (sum != 1 is expected).
//               Lanes beyond n_elems in the last beat are 0 (out_keep=0).
//   y = -t is formed as SQ8.15: y_code = -(t << 7).
//   d1, d2    : UQ2.15 codes, PROVISIONAL (not from the paper), must be < 2.0
//               (values above 65535 are clamped to 65535).
//
// Control
//   start      : pulse, honoured only while busy=0. n_elems sampled on that clock.
//                n_elems==0 or > MAX_N_EFF is rejected (status[6]), stays idle.
//   in_ready   : high while PASS1 accepts beats; exactly ceil(n/16) beats are
//                taken (no in_last needed).
//   out_*      : valid/ready; out_last marks the final beat; busy drops after it
//                is accepted.
//   rst        : synchronous, active high. Resets FSM, counters and valid bits.
//
// MAX_N_EFF = min(MAX_N, 65535, floor((2^32-1)/TERM_MAX)) where TERM_MAX is the
// largest pass-1 term (D1 code if D1>=1.0, else 65535), so F cannot overflow.
//
// status (sticky, cleared on each start):
//   [0] EXPU a ovf   [1] EXPU z ovf   [2] EXPU out ovf   [3] accumulator ovf
//   [4] LNU ovf      [5] F==0 (lnF forced 0)   [6] invalid n_elems
//   [7] no zero-t element found (software contract violated)
// =============================================================================
module softmax_top #(
    parameter MAX_N    = 1024,
    parameter [16:0] D1_CODE = 17'd38010,    // UQ2.15, PROVISIONAL
    parameter [16:0] D2_CODE = 17'd35471,    // UQ2.15, PROVISIONAL
    parameter ROUNDING = 0,                  // 0 trunc,1 toward_zero,2 half_up,3 half_even
    parameter OVERFLOW = 0                   // 0 saturate, 1 wrap
)(
    input              clk,
    input              rst,

    input              start,
    input      [15:0]  n_elems,
    output             busy,

    input  [255:0]     in_data,
    input              in_valid,
    output             in_ready,

    output [255:0]     out_data,
    output [15:0]      out_keep,
    output             out_valid,
    input              out_ready,
    output             out_last,

    output reg [7:0]   status
);
    // ---------------- formats ----------------
    localparam LANES = 16;
    localparam T_W   = 16;               // UQ8.8
    localparam T_F   = 8;
    localparam A_W   = 24;               // SQ8.15
    localparam A_F   = 15;
    localparam ALIGN = A_F - T_F;        // 7
    localparam OUT_W = 16;               // UQ1.15
    localparam OUT_F = 15;
    localparam D_W   = 17;               // UQ2.15
    localparam ACC_W = 32;               // UQ17.15
    localparam ACC_F = 15;
    localparam SB_W  = 3;                // {pass, last, keep}

    localparam [16:0] D1_C = (D1_CODE > 17'd65535) ? 17'd65535 : D1_CODE;
    localparam [16:0] D2_C = (D2_CODE > 17'd65535) ? 17'd65535 : D2_CODE;

    // ---------------- maximum vector length ----------------
    localparam [63:0] ACC_MAXV   = 64'd4294967295;
    localparam [63:0] D1_64      = {47'd0, D1_C};
    localparam [63:0] TERM_MAX   = (D1_64 >= 64'd32768) ? D1_64 : 64'd65535;
    localparam [63:0] ACC_LIMIT  = ACC_MAXV / TERM_MAX;
    localparam [63:0] MAXN_P     = MAX_N;
    localparam [63:0] MAXN_B     = (MAXN_P < 64'd65535) ? MAXN_P : 64'd65535;
    localparam [63:0] MAXN_E     = (MAXN_B < ACC_LIMIT) ? MAXN_B : ACC_LIMIT;
    localparam [15:0] MAX_N_EFF  = MAXN_E;
    localparam integer MAX_K     = (MAX_N_EFF + 15) / 16;
    localparam integer AW        = (MAX_K > 1) ? $clog2(MAX_K) : 1;
    localparam KC_W              = 13;   // beat counters (n <= 65535 -> <= 4096 beats)

    // ---------------- FSM ----------------
    localparam [2:0] S_IDLE   = 3'd0,
                     S_P1     = 3'd1,
                     S_DRAIN1 = 3'd2,
                     S_LNU    = 3'd3,
                     S_P2     = 3'd4,
                     S_DRAIN2 = 3'd5;
    reg [2:0] state;
    assign busy = (state != S_IDLE);

    reg [15:0]     n_reg;
    reg [KC_W-1:0] k_last;               // beats - 1
    reg [KC_W-1:0] wr_cnt, rd_cnt;
    reg [15:0]     rem;                  // elements not yet issued
    reg            zero_seen;
    reg [A_W-1:0]  lnf_reg;

    // output register / global advance
    reg                   out_valid_q;
    reg                   out_last_q;
    reg [LANES*OUT_W-1:0] out_data_q;
    reg [LANES-1:0]       out_keep_q;
    wire adv = ~out_valid_q | out_ready;          // global clock-enable of the datapath

    // start handling
    wire n_ok      = (n_elems != 16'd0) && (n_elems <= MAX_N_EFF);
    wire start_go  = (state == S_IDLE) && start;
    wire start_acc = start_go && n_ok;
    wire [16:0] n_plus = {1'b0, n_elems} + 17'd15;
    wire [KC_W-1:0] k_tot_m1 = n_plus[16:4] - 1'b1;

    // issue control
    assign in_ready = (state == S_P1) && adv;
    wire load_p1 = in_valid && in_ready;
    wire load_p2 = (state == S_P2) && adv;
    wire load_any = load_p1 | load_p2;

    wire [15:0] mask_now = (rem >= 16'd16) ? 16'hFFFF : ((16'h0001 << rem[3:0]) - 16'h0001);
    wire [15:0] rem_next = (rem >= 16'd16) ? (rem - 16'd16) : 16'd0;

    // ---------------- t buffer ----------------
    reg [LANES*T_W-1:0] tbuf [0:MAX_K-1];
    reg [LANES*T_W-1:0] rd_t;
    reg [LANES*T_W-1:0] iss_t;
    always @(posedge clk) begin
        if (load_p1) tbuf[wr_cnt[AW-1:0]] <= in_data;
        if (load_p2) rd_t <= tbuf[rd_cnt[AW-1:0]];
        if (load_p1) iss_t <= in_data;
    end

    // issue-stage meta
    reg            iss_valid, iss_pass, iss_last;
    reg [15:0]     iss_mask;
    always @(posedge clk) begin
        if (rst) iss_valid <= 1'b0;
        else if (adv) iss_valid <= load_any;
    end
    always @(posedge clk) begin
        if (adv && load_any) begin
            iss_pass <= load_p2;
            iss_last <= load_p2 ? (rd_cnt == k_last) : (wr_cnt == k_last);
            iss_mask <= mask_now;
        end
    end

    // zero-element contract check (pass 1)
    wire [LANES-1:0] lane_zero;
    genvar gz;
    generate
        for (gz = 0; gz < LANES; gz = gz + 1) begin : g_zero
            assign lane_zero[gz] = (in_data[gz*T_W +: T_W] == {T_W{1'b0}});
        end
    endgenerate
    wire zero_beat  = |(lane_zero & mask_now);
    wire ev_nozero  = load_p1 && (wr_cnt == k_last) && !(zero_seen | zero_beat);

    // ---------------- shared EXPU array ----------------
    wire [A_W-1:0] ln_in = iss_pass ? lnf_reg : {A_W{1'b0}};   // pass-1 lnF = 0 (assumed)
    wire [D_W-1:0] d_in  = iss_pass ? D2_C : D1_C;

    wire [LANES-1:0]       ex_valid;
    wire [LANES*SB_W-1:0]  ex_sb;
    wire [LANES*OUT_W-1:0] ex_out;
    wire [LANES-1:0]       ex_ova, ex_ovz, ex_ovo, ex_keep;

    genvar gi;
    generate
        for (gi = 0; gi < LANES; gi = gi + 1) begin : g_lane
            wire [T_W-1:0] t_i = iss_pass ? rd_t[gi*T_W +: T_W] : iss_t[gi*T_W +: T_W];
            wire [A_W-1:0] y_i = {A_W{1'b0}} -
                                 {{(A_W-T_W-ALIGN){1'b0}}, t_i, {ALIGN{1'b0}}};   // -(t<<7)
            expu #(.A_W(A_W), .A_F(A_F), .Z_W(A_W), .Z_F(A_F), .OUT_W(OUT_W), .OUT_F(OUT_F),
                   .ROUNDING(ROUNDING), .OVERFLOW(OVERFLOW), .SB_W(SB_W))
            u_expu (
                .clk(clk), .rst(rst), .en(adv),
                .in_valid(iss_valid),
                .in_sb({iss_pass, iss_last, iss_mask[gi]}),
                .y_code(y_i), .ln_f_code(ln_in), .d_code(d_in),
                .out_valid(ex_valid[gi]),
                .out_sb(ex_sb[gi*SB_W +: SB_W]),
                .out_code(ex_out[gi*OUT_W +: OUT_W]),
                .ovf_a(ex_ova[gi]), .ovf_z(ex_ovz[gi]), .ovf_out(ex_ovo[gi])
            );
            assign ex_keep[gi] = ex_sb[gi*SB_W];
        end
    endgenerate

    wire ex_v0    = ex_valid[0];
    wire ex_last0 = ex_sb[1];
    wire ex_pass0 = ex_sb[2];

    // ---------------- adder tree / accumulator (pass 1 only) ----------------
    wire [ACC_W-1:0] F_code;
    wire             acc_done, acc_ovf;
    adder_tree #(.LANE_W(OUT_W), .LANE_F(OUT_F), .ACC_W(ACC_W), .ACC_F(ACC_F),
                 .ROUNDING(ROUNDING), .OVERFLOW(OVERFLOW))
    u_tree (
        .clk(clk), .rst(rst), .acc_clear(start_acc),
        .in_valid(ex_v0 & ~ex_pass0), .in_last(ex_last0),
        .lane_in(ex_out), .lane_keep(ex_keep),
        .f_code(F_code), .acc_done(acc_done), .acc_ovf(acc_ovf)
    );

    // ---------------- LNU ----------------
    wire             lnu_in_valid = (state == S_DRAIN1) && acc_done;
    wire             lnu_valid, lnu_ovf, lnu_fzero;
    wire [A_W-1:0]   lnu_ln_f;
    lnu #(.ACC_W(ACC_W), .ACC_F(ACC_F), .LNU_F(15), .A_W(A_W), .A_F(A_F),
          .ROUNDING(ROUNDING), .OVERFLOW(OVERFLOW))
    u_lnu (
        .clk(clk), .rst(rst), .in_valid(lnu_in_valid), .f_code(F_code),
        .out_valid(lnu_valid), .ln_f_code(lnu_ln_f), .ovf(lnu_ovf), .f_zero(lnu_fzero)
    );

    // ---------------- output stage (pass 2 only) ----------------
    wire [LANES*OUT_W-1:0] ex_out_masked;
    generate
        for (gi = 0; gi < LANES; gi = gi + 1) begin : g_mask
            assign ex_out_masked[gi*OUT_W +: OUT_W] =
                ex_keep[gi] ? ex_out[gi*OUT_W +: OUT_W] : {OUT_W{1'b0}};
        end
    endgenerate

    always @(posedge clk) begin
        if (rst) out_valid_q <= 1'b0;
        else if (adv) out_valid_q <= ex_v0 & ex_pass0;
    end
    always @(posedge clk) begin
        if (adv) begin
            out_data_q <= ex_out_masked;
            out_keep_q <= ex_keep;
            out_last_q <= ex_last0;
        end
    end
    assign out_data  = out_data_q;
    assign out_keep  = out_keep_q;
    assign out_valid = out_valid_q;
    assign out_last  = out_last_q;

    // ---------------- sticky status ----------------
    wire ev_a = |(ex_valid & ex_keep & ex_ova);
    wire ev_z = |(ex_valid & ex_keep & ex_ovz);
    wire ev_o = |(ex_valid & ex_keep & ex_ovo);
    wire [7:0] ev = {ev_nozero, 1'b0, lnu_valid & lnu_fzero, lnu_valid & lnu_ovf,
                     acc_ovf, ev_o, ev_z, ev_a};

    always @(posedge clk) begin
        if (rst)           status <= 8'h00;
        else if (start_go) status <= n_ok ? 8'h00 : 8'h40;
        else               status <= status | ev;
    end

    // ---------------- FSM / counters ----------------
    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
        end else begin
            case (state)
                S_IDLE:   if (start_acc) state <= S_P1;
                S_P1:     if (load_p1 && (wr_cnt == k_last)) state <= S_DRAIN1;
                S_DRAIN1: if (acc_done) state <= S_LNU;
                S_LNU:    if (lnu_valid) state <= S_P2;
                S_P2:     if (load_p2 && (rd_cnt == k_last)) state <= S_DRAIN2;
                S_DRAIN2: if (out_valid_q && out_ready && out_last_q) state <= S_IDLE;
                default:  state <= S_IDLE;
            endcase
        end
    end

    always @(posedge clk) begin
        if (start_acc) begin
            n_reg     <= n_elems;
            k_last    <= k_tot_m1;
            rem       <= n_elems;
            wr_cnt    <= {KC_W{1'b0}};
            rd_cnt    <= {KC_W{1'b0}};
            zero_seen <= 1'b0;
        end else begin
            if (load_p1) begin
                wr_cnt    <= wr_cnt + 1'b1;
                rem       <= rem_next;
                zero_seen <= zero_seen | zero_beat;
            end
            if (state == S_LNU && lnu_valid) begin
                lnf_reg <= lnu_ln_f;
                rem     <= n_reg;
                rd_cnt  <= {KC_W{1'b0}};
            end
            if (load_p2) begin
                rd_cnt <= rd_cnt + 1'b1;
                rem    <= rem_next;
            end
        end
    end
endmodule
