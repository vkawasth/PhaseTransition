include("ReesBlowupHybrid.jl")
using .ReesBlowupHybrid

B = run_blowup!(".")
print_events(B)
print_mittag_leffler_summary(B)
write_blowup_table(B, "blowup_table.tsv")
plot_mittag_leffler(B)
plot_generators(B)
