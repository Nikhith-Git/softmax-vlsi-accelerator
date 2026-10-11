// =============================================================================
// tb_softmax_top.v : self-checking testbench for softmax_top (Verilog-2001).
//
// Vectors come from scripts/gen_vectors.py (vectors/meta.hex, in.hex, exp.hex);
// run xsim from the project root so the relative paths resolve.
// Every case is run RUNS times: run 0 = no stalls, run 1 = random input gaps and
// random out_ready de-assertion (back-pressure).
// Checks per case: reject behaviour, beat count, out_data (bit exact), out_keep,
// out_last, final sticky status, busy/out_valid back to 0.
// Prints "TB_RESULT: PASS" only if zero errors were counted over >0 cases.
// The DUT parameters below must match vectors/params.txt.
// =============================================================================
`timescale 1ns/1ps
module tb_softmax_top;
    parameter MAX_N          = 1024;
    parameter [16:0] D1_CODE = 17'd38010;
    parameter [16:0] D2_CODE = 17'd35471;
    parameter ROUNDING       = 0;
    parameter OVERFLOW       = 0;
    parameter RUNS           = 2;
    parameter IN_DEPTH       = 16384;
    parameter MAX_CASES      = 1024;
    parameter CASE_TIMEOUT   = 20000;                 // clocks
    parameter META_FILE      = "vectors/meta.hex";
    parameter IN_FILE        = "vectors/in.hex";
    parameter EXP_FILE       = "vectors/exp.hex";

    localparam MAXK = (MAX_N + 15) / 16 + 4;

    reg          clk = 1'b0;
    reg          rst;
    reg          start;
    reg  [15:0]  n_elems;
    reg  [255:0] in_data;
    reg          in_valid;
    reg          out_ready;
    wire         busy, in_ready, out_valid, out_last;
    wire [255:0] out_data;
    wire [15:0]  out_keep;
    wire [7:0]   status;

    softmax_top #(.MAX_N(MAX_N), .D1_CODE(D1_CODE), .D2_CODE(D2_CODE),
                  .ROUNDING(ROUNDING), .OVERFLOW(OVERFLOW)) dut (
        .clk(clk), .rst(rst), .start(start), .n_elems(n_elems), .busy(busy),
        .in_data(in_data), .in_valid(in_valid), .in_ready(in_ready),
        .out_data(out_data), .out_keep(out_keep), .out_valid(out_valid),
        .out_ready(out_ready), .out_last(out_last), .status(status));

    always #5 clk = ~clk;

    // ---------------- vector memories ----------------
    reg [159:0] meta   [0:MAX_CASES];
    reg [255:0] in_mem  [0:IN_DEPTH-1];
    reg [255:0] exp_mem [0:IN_DEPTH-1];

    // ---------------- monitors (sample pre-edge values in the active region) ----------------
    reg          hs_in;
    integer      out_cnt;
    reg [255:0]  got_data [0:MAXK-1];
    reg [15:0]   got_keep [0:MAXK-1];
    reg          got_last [0:MAXK-1];
    reg          stress;
    reg [15:0]   lfsr_o;
    reg [15:0]   lfsr_i;

    always @(posedge clk) begin
        hs_in = in_valid & in_ready;
        if (out_valid & out_ready) begin
            if (out_cnt < MAXK) begin
                got_data[out_cnt] = out_data;
                got_keep[out_cnt] = out_keep;
                got_last[out_cnt] = out_last;
            end
            out_cnt = out_cnt + 1;
        end
    end

    always @(negedge clk) begin
        lfsr_o    = {lfsr_o[14:0], lfsr_o[15] ^ lfsr_o[13] ^ lfsr_o[12] ^ lfsr_o[10]};
        out_ready = stress ? (lfsr_o[0] | lfsr_o[3]) : 1'b1;
    end

    // ---------------- bookkeeping ----------------
    integer errors, cases_run, ncases;
    integer run, ci;

    task err;
        input [8*40-1:0] what;
        begin
            errors = errors + 1;
            if (errors <= 30) $display("ERROR run=%0d case=%0d: %0s", run, ci, what);
        end
    endtask

    task wait_neg;
        input integer cycles;
        integer j;
        begin
            for (j = 0; j < cycles; j = j + 1) @(negedge clk);
        end
    endtask

    // send one input beat, honouring in_ready (handshake seen by the monitor)
    task send_beat;
        input [255:0] d;
        integer tmo;
        begin
            in_data  = d;
            in_valid = 1'b1;
            tmo = 0;
            @(negedge clk);
            while (!hs_in && tmo < CASE_TIMEOUT) begin
                @(negedge clk);
                tmo = tmo + 1;
            end
            if (tmo >= CASE_TIMEOUT) err("timeout waiting for in_ready");
            in_valid = 1'b0;
        end
    endtask

    task run_case;
        input integer idx;
        reg [159:0] mw;
        integer n, in_off, out_off, k, b, l, g, tmo, rem;
        reg [7:0]  exp_st;
        reg        rej;
        reg [15:0] exp_keep;
        reg [255:0] got, expd;
        begin
            mw      = meta[idx];
            n       = mw[31:0];
            in_off  = mw[63:32];
            out_off = mw[95:64];
            exp_st  = mw[103:96];
            rej     = mw[128];
            k       = (n + 15) / 16;

            // wait for idle, then pulse start with n_elems
            tmo = 0;
            while (busy && tmo < CASE_TIMEOUT) begin @(negedge clk); tmo = tmo + 1; end
            if (busy) err("busy stuck before start");
            @(negedge clk);
            out_cnt = 0;
            n_elems = n[15:0];
            start   = 1'b1;
            @(negedge clk);
            start   = 1'b0;
            n_elems = 16'd0;

            if (rej) begin
                // rejected: stays idle, status == 0x40, no handshakes
                if (busy)            err("busy after rejected start");
                if (status !== 8'h40) begin
                    err("status != 0x40 after invalid n");
                    $display("       got status=0x%02h", status);
                end
                wait_neg(3);
                if (busy || in_ready || out_valid) err("activity after rejected start");
                if (status !== 8'h40) err("status changed after reject");
            end else begin
                if (!busy) err("busy not set after start");
                // input beats (optional random gaps in stress runs)
                for (b = 0; b < k; b = b + 1) begin
                    if (stress) begin
                        lfsr_i = {lfsr_i[14:0], lfsr_i[15] ^ lfsr_i[13] ^ lfsr_i[12] ^ lfsr_i[10]};
                        g = lfsr_i[1:0] % 3;
                        wait_neg(g);
                    end
                    send_beat(in_mem[in_off + b]);
                end
                // wait until the vector is finished (last output beat accepted)
                tmo = 0;
                while (busy && tmo < CASE_TIMEOUT) begin @(negedge clk); tmo = tmo + 1; end
                if (busy) err("timeout: busy never fell");
                // output checks
                if (out_cnt != k) begin
                    err("wrong number of output beats");
                    $display("       got %0d beats, expected %0d (n=%0d)", out_cnt, k, n);
                end
                for (b = 0; b < k && b < MAXK && b < out_cnt; b = b + 1) begin
                    got  = got_data[b];
                    expd = exp_mem[out_off + b];
                    if (got !== expd) begin
                        err("out_data mismatch");
                        for (l = 0; l < 16; l = l + 1)
                            if (got[16*l +: 16] !== expd[16*l +: 16])
                                $display("       beat %0d lane %0d got=0x%04h exp=0x%04h",
                                         b, l, got[16*l +: 16], expd[16*l +: 16]);
                    end
                    rem = n - 16 * b;
                    exp_keep = (rem >= 16) ? 16'hFFFF : ((16'h0001 << rem) - 16'h0001);
                    if (got_keep[b] !== exp_keep) err("out_keep mismatch");
                    if (got_last[b] !== (b == k - 1)) err("out_last mismatch");
                end
                wait_neg(2);
                if (status !== exp_st) begin
                    err("status mismatch");
                    $display("       got status=0x%02h exp=0x%02h", status, exp_st);
                end
                if (busy || out_valid || in_ready) err("not idle at end of case");
            end
            cases_run = cases_run + 1;
        end
    endtask

    initial begin
        errors = 0; cases_run = 0; out_cnt = 0; stress = 1'b0;
        lfsr_o = 16'hACE1; lfsr_i = 16'h1D2B;
        rst = 1'b1; start = 1'b0; n_elems = 16'd0; in_data = 256'd0;
        in_valid = 1'b0; out_ready = 1'b1;
        $readmemh(META_FILE, meta);
        $readmemh(IN_FILE,   in_mem);
        $readmemh(EXP_FILE,  exp_mem);
        begin : rd_meta
            reg [159:0] m0;
            m0 = meta[0];
            ncases = m0[31:0];
        end
        if (ncases < 1 || ncases > MAX_CASES) begin
            $display("TB_RESULT: FAIL (no cases loaded; run from project root after gen_vectors.py)");
            $finish;
        end
        wait_neg(5);
        rst = 1'b0;
        wait_neg(3);
        for (run = 0; run < RUNS; run = run + 1) begin
            stress = (run == 1);
            for (ci = 1; ci <= ncases; ci = ci + 1)
                run_case(ci);
            $display("run %0d done: cases=%0d errors so far=%0d", run, ncases, errors);
        end
        if (errors == 0 && cases_run > 0)
            $display("TB_RESULT: PASS (%0d case executions, 0 errors)", cases_run);
        else
            $display("TB_RESULT: FAIL (%0d errors, %0d case executions)", errors, cases_run);
        $finish;
    end

    initial begin
        #400_000_000;
        $display("TB_RESULT: FAIL (global watchdog)");
        $finish;
    end
endmodule
