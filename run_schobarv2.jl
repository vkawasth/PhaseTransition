include("SchobarNavigatorV2.jl")
using .SchoberNavigatorV2

files = filter(f -> occursin(r"ainf_export_(?:\w+_)?\d+(?:\.\d+)?\.json", basename(f)), readdir(folder))
sort!(files)
S = run_schober_path(files)
summarize_path(S)
write_chamber_table(S, "chambers.tsv")
write_wall_table(S, "walls.tsv")
