// Wang-inspired softmax, 16-bit fixed point (FIXED_16BIT of the Python model).
// Preprocessing (t = x_max - x) stays in software. Rounding = trunc, overflow = saturate.
//
// Formats (value = code * 2^-frac):
//   t UQ8.8 (16b, input)   a,z SQ8.15 (24b)   d1,d2,m UQ2.15 (17b)
//   EXPU out / p UQ1.15 (16b)   tree sum 20b exact   F UQ17.15 (32b)   lnF UQ4.15 (19b)
//
// Architecture: shared EXPU array (LANES lanes) used for both passes:
//   load beats -> [pass 1: EXPU(d1, lnF=0) -> adder tree -> F] -> LNU -> [pass 2: EXPU(d2, y-lnF) -> p]
// Beat = LANES elements, element i of the vector = beat i/LANES, lane i%LANES (bits [16*l +: 16]).
// The last beat's in_cnt (1..LANES) gives its number of valid lanes; unused lanes are
// zero-padded AFTER the EXPU (as in Python), and flagged 0 in out_keep.
// One vector at a time: in_ready is low from the last input beat until the last output beat.
//
// Supported range: at least one t == 0 per vector (software x_max subtraction), which
// gives F >= d1 >= 1.0. Anything else is flagged, never silently computed:
//   out_status[0] err_len : more than MAX_BEATS beats; extra beats dropped
//   out_status[1] err_cnt : in_cnt of last beat is 0 or > LANES (> LANES is clamped)
//   out_status[2] err_f   : F < 1.0 (e.g. no t == 0, or F == 0)
//   out_status[3] acc_sat : F saturated at 2^32-1 (N ~> 113k); outputs remain valid per Python
// With err_len/err_cnt/err_f set, all p of the vector are 0 (out_keep still marks the lanes).
module softmax_top #(
  parameter int          LANES     = 16,         // power of two; Python preset uses 16
  parameter int          MAX_BEATS = 256,        // buffer depth: max N = LANES*MAX_BEATS
  parameter logic [16:0] D1_CODE   = 17'd38010,  // PROVISIONAL d1=1.16   = floor(1.16*2^15)
  parameter logic [16:0] D2_CODE   = 17'd35471   // PROVISIONAL d2=1.0825 = floor(1.0825*2^15)
) (
  input  logic                        clk,
  input  logic                        rst,        // synchronous, active high
  // input stream
  input  logic                        in_valid,
  output logic                        in_ready,
  input  logic [LANES*16-1:0]         in_data,    // t codes, UQ8.8
  input  logic                        in_last,
  input  logic [$clog2(LANES+1)-1:0]  in_cnt,     // valid lanes of the last beat (1..LANES)
  // output stream
  output logic                        out_valid,
  input  logic                        out_ready,
  output logic [LANES*16-1:0]         out_data,   // p codes, UQ1.15
  output logic [LANES-1:0]            out_keep,
  output logic                        out_last,
  output logic [3:0]                  out_status
);
  localparam int CNT_W  = $clog2(LANES+1);
  localparam int AW     = $clog2(MAX_BEATS);
  localparam int BW     = $clog2(MAX_BEATS+1);
  localparam int TREE_W = 16 + $clog2(LANES);

  generate
    if (D1_CODE < 17'd32768) begin : g_chk_d1
      $error("softmax_top: D1_CODE must be >= 2^15 (d1 >= 1) so F >= 1.0 whenever a t==0 exists");
    end
    if (D2_CODE > 17'd65535 || D1_CODE > 17'd65535) begin : g_chk_d
      $error("softmax_top: D1/D2 codes must be < 2^16 (d < 2)");
    end
    if (MAX_BEATS < 2) begin : g_chk_mb
      $error("softmax_top: MAX_BEATS >= 2");
    end
  endgenerate

  typedef enum logic [2:0] {S_LOAD, S_P1, S_P1_WAIT, S_LNU, S_P2, S_P2_DRAIN} state_t;
  state_t state;

  // ---------------- storage and control ----------------
  (* ram_style = "block" *) logic [LANES*16-1:0] mem [0:MAX_BEATS-1];

  logic [BW-1:0]     wcnt, raddr;
  logic [CNT_W-1:0]  last_cnt;
  logic              err_len, err_cnt, err_f;
  logic [18:0]       ln_f_q;

  logic [31:0]       acc_F;
  logic              acc_done, acc_sat;
  logic              lnu_valid;
  logic [18:0]       lnu_ln_f;

  logic              out_v_i;
  wire               en        = ~(out_v_i & ~out_ready);   // global pipeline enable
  wire               vec_err   = err_len | err_cnt | err_f;
  wire               issuing   = (state == S_P1) | (state == S_P2);
  wire               last_beat = (raddr + 1'b1 == wcnt);

  assign in_ready = (state == S_LOAD);

  always_ff @(posedge clk) begin
    if (in_valid && in_ready && wcnt < BW'(MAX_BEATS)) mem[wcnt[AW-1:0]] <= in_data;
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      state    <= S_LOAD;
      wcnt     <= '0;
      raddr    <= '0;
      last_cnt <= CNT_W'(LANES);
      err_len  <= 1'b0;
      err_cnt  <= 1'b0;
      err_f    <= 1'b0;
      ln_f_q   <= '0;
    end else begin
      case (state)
        S_LOAD: if (in_valid) begin
          if (wcnt < BW'(MAX_BEATS)) wcnt <= wcnt + 1'b1;
          else                       err_len <= 1'b1;
          if (in_last) begin
            if (in_cnt == '0 || in_cnt > CNT_W'(LANES)) err_cnt <= 1'b1;
            if (wcnt < BW'(MAX_BEATS))
              last_cnt <= (in_cnt > CNT_W'(LANES)) ? CNT_W'(LANES) : in_cnt;
            state <= S_P1;
          end
        end
        S_P1: if (en) begin
          if (last_beat) begin raddr <= '0; state <= S_P1_WAIT; end
          else           raddr <= raddr + 1'b1;
        end
        S_P1_WAIT: if (acc_done) state <= S_LNU;
        S_LNU: if (lnu_valid) begin
          ln_f_q <= lnu_ln_f;
          err_f  <= (acc_F < 32'd32768);
          state  <= S_P2;
        end
        S_P2: if (en) begin
          if (last_beat) begin raddr <= '0; state <= S_P2_DRAIN; end
          else           raddr <= raddr + 1'b1;
        end
        S_P2_DRAIN: if (out_v_i && out_ready && out_last) begin
          state    <= S_LOAD;
          wcnt     <= '0;
          raddr    <= '0;
          last_cnt <= CNT_W'(LANES);
          err_len  <= 1'b0;
          err_cnt  <= 1'b0;
          err_f    <= 1'b0;
          ln_f_q   <= '0;
        end
        default: state <= S_LOAD;
      endcase
    end
  end

  // ---------------- issue stage (BRAM read + side info) ----------------
  logic [LANES-1:0] mask_c;
  always_comb
    for (int l = 0; l < LANES; l++) mask_c[l] = ~last_beat | (CNT_W'(l) < last_cnt);

  logic [LANES*16-1:0] rd_data;
  logic                rd_valid, rd_pass2, rd_last;
  logic [LANES-1:0]    rd_mask;
  always_ff @(posedge clk) if (en) rd_data <= mem[raddr[AW-1:0]];
  always_ff @(posedge clk) begin
    if (rst) begin
      rd_valid <= 1'b0;
    end else if (en) begin
      rd_valid <= issuing;
    end
  end
  always_ff @(posedge clk) if (en) begin
    rd_pass2 <= (state == S_P2);
    rd_last  <= last_beat;
    rd_mask  <= mask_c;
  end

  // ---------------- EXPU array (3 stages) + side pipeline ----------------
  logic [LANES*16-1:0] exp_out;
  genvar l;
  generate
    for (l = 0; l < LANES; l++) begin : g_lane
      expu #(.D1_CODE(D1_CODE), .D2_CODE(D2_CODE)) u_expu (
        .clk(clk), .en(en),
        .t(rd_data[l*16 +: 16]),
        .ln_f(ln_f_q),                       // 0 during pass 1 (cleared per vector)
        .sel_d2(rd_pass2),
        .keep(rd_mask[l] & ~(rd_pass2 & vec_err)),
        .out(exp_out[l*16 +: 16])
      );
    end
  endgenerate

  logic [2:0]       sv_valid, sv_pass2, sv_last;
  logic [LANES-1:0] sv_mask [3];
  always_ff @(posedge clk) begin
    if (rst) sv_valid <= '0;
    else if (en) sv_valid <= {sv_valid[1:0], rd_valid};
  end
  always_ff @(posedge clk) if (en) begin
    sv_pass2 <= {sv_pass2[1:0], rd_pass2};
    sv_last  <= {sv_last[1:0],  rd_last};
    sv_mask[0] <= rd_mask;
    sv_mask[1] <= sv_mask[0];
    sv_mask[2] <= sv_mask[1];
  end

  // ---------------- pass 1 sink: adder tree -> F -> LNU ----------------
  logic [TREE_W-1:0] tree_sum;
  logic              tree_valid, tree_last;
  adder_tree #(.LANES(LANES), .IN_W(16)) u_tree (
    .clk(clk), .rst(rst), .en(en),
    .lane_in(exp_out),
    .in_valid(sv_valid[2] & ~sv_pass2[2]),
    .in_last(sv_last[2]),
    .sum(tree_sum), .out_valid(tree_valid), .out_last(tree_last)
  );

  accumulator #(.IN_W(TREE_W), .ACC_W(32)) u_acc (
    .clk(clk), .rst(rst), .en(en),
    .clear(state == S_LOAD),
    .add_valid(tree_valid), .add_val(tree_sum), .add_last(tree_last),
    .F(acc_F), .done(acc_done), .sat(acc_sat)
  );

  lnu u_lnu (.clk(clk), .in_valid(acc_done), .F(acc_F), .out_valid(lnu_valid), .ln_f(lnu_ln_f));

  // ---------------- pass 2 sink: output stream ----------------
  assign out_v_i    = sv_valid[2] & sv_pass2[2];
  assign out_valid  = out_v_i;
  assign out_data   = exp_out;
  assign out_keep   = sv_mask[2];
  assign out_last   = sv_last[2];
  assign out_status = {acc_sat, err_f, err_cnt, err_len};
endmodule
