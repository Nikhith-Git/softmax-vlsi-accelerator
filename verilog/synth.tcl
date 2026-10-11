# Out-of-context synthesis + reports.  Usage (from project root):
#   PART=<your-fpga-part> vivado -mode batch -source scripts/synth.tcl
# PART is not defined by the project; set it explicitly. CLK_NS defaults to 10.0.
if {![info exists ::env(PART)]} { error "set the PART environment variable" }
set clk_ns [expr {[info exists ::env(CLK_NS)] ? $::env(CLK_NS) : 10.0}]
file mkdir reports
read_verilog [list rtl/arith_common.v rtl/expu.v rtl/adder_tree.v rtl/lnu.v rtl/softmax_top.v]
synth_design -top softmax_top -part $::env(PART) -mode out_of_context
create_clock -name clk -period $clk_ns [get_ports clk]
report_utilization -hierarchical -file reports/utilization.rpt
report_timing_summary -file reports/timing_summary.rpt
report_timing -max_paths 10 -file reports/timing_paths.rpt
report_drc -file reports/drc.rpt
puts "Vivado [version -short], part $::env(PART), clock ${clk_ns} ns (post-synthesis, no placement)"
