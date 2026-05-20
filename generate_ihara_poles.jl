"""
generate_ihara_poles.jl
=======================
Generates ihara_poles.json from connectome_graphs.json + ainf_export_*.json.

The poles of the Ihara zeta are 1/λ where λ are eigenvalues of B_Ihara.
Each snapshot gets the same structural poles but with A∞-weighted spectral
radius from the ihara_radius field in the JSON.

Usage:
    julia generate_ihara_poles.jl [folder] [graph_type]

Output:
    ihara_poles.json
"""

using JSON3, LinearAlgebra, Printf

folder     = length(ARGS) >= 1 ? ARGS[1] : "."
graph_type = length(ARGS) >= 2 ? ARGS[2] : "Q_7P"

# ── Load B_Ihara from connectome_graphs.json ──────────────────────────────────

function load_B_ihara(graph_type)
    gpath = joinpath(folder, "connectome_graphs.json")
    !isfile(gpath) && error("connectome_graphs.json not found in $folder")
    gdata = JSON3.read(read(gpath, String))
    g = get(gdata, Symbol(graph_type), nothing)
    isnothing(g) && error("Graph type $graph_type not found in connectome_graphs.json")
    
    n_arr = Int(g[:n_arr])
    arrows = g[:arrows]
    
    # Build reversal map
    rev = zeros(Int, n_arr)
    for a in arrows
        ri = Int(a["rev_idx"])
        ri > 0 && (rev[Int(a["idx"])] = ri)
    end
    
    # Build B_Ihara
    B = zeros(Float64, n_arr, n_arr)
    for ai in arrows, aj in arrows
        i = Int(ai["idx"]); j = Int(aj["idx"])
        if Int(ai["v_tgt"]) == Int(aj["v_src"]) &&
           !(Bool(ai["sym"]) && rev[i] == j)
            B[i, j] = 1.0
        end
    end
    
    # Ramanujan bound: √(q_max) where q = max vertex degree
    n_v = Int(g[:n_v])
    deg = zeros(Int, n_v)
    for a in arrows
        v = Int(a["v_src"])
        v <= n_v && (deg[v] += 1)
    end
    q_max = maximum(deg)
    ramanujan_bound = sqrt(Float64(q_max))
    
    return B, n_arr, ramanujan_bound, q_max
end

println("Loading B_Ihara for $graph_type...")
B, n_arr, ramanujan_bound, q_max = load_B_ihara(graph_type)

# Compute eigenvalues of B_Ihara (same for all snapshots — structural)
eigs_B = eigvals(B)
println("B_Ihara: $(n_arr)×$(n_arr), ρ=$(round(maximum(abs.(eigs_B)), digits=4))")
println("Ramanujan bound √q = $(round(ramanujan_bound, digits=4))  (q_max=$q_max)")

# ── Load all ainf_export JSON files ──────────────────────────────────────────

files = sort(filter(f -> endswith(f, ".json") && startswith(f, "ainf_export"),
                    readdir(folder)))
n_snapshots = length(files)
println("Found $n_snapshots ainf_export JSON files")
isempty(files) && error("No ainf_export JSON files found in $folder")

n_poles_per_snap = n_arr   # one pole per B_Ihara eigenvalue

# ── Build poles array ─────────────────────────────────────────────────────────

poles            = Dict{String,Any}[]
spectral_radii   = Float64[]
ramanujan_bounds = Float64[]

for (s, f) in enumerate(files)
    snap = JSON3.read(read(joinpath(folder, f), String))
    
    # Get A∞-weighted spectral radius from snapshot if available
    # Otherwise use unweighted B_Ihara spectral radius
    ainf_rho = get(snap, "ihara_radius", maximum(abs.(eigs_B)))
    ainf_rho = Float64(ainf_rho)
    
    # Scale factor: how much A∞ shifts the spectral radius
    # bridge_b_ratio = ainf_rho / unweighted_rho
    unweighted_rho = maximum(abs.(eigs_B))
    scale = unweighted_rho > 1e-10 ? ainf_rho / unweighted_rho : 1.0
    
    for λ in eigs_B
        # Scale eigenvalue by A∞ factor
        λ_scaled = λ * scale
        push!(poles, Dict(
            "snapshot"        => s,
            "radius"          => abs(λ_scaled),
            "ramanujan_bound" => ramanujan_bound,
            "re"              => real(λ_scaled),
            "im"              => imag(λ_scaled),
            "unweighted_r"    => abs(λ),
            "ainf_scale"      => scale,
        ))
    end
    
    push!(spectral_radii, ainf_rho)
    push!(ramanujan_bounds, ramanujan_bound)
    s % 100 == 0 && @printf("  %d/%d  rho=%.4f  scale=%.4f\n",
                             s, n_snapshots, ainf_rho, scale)
end

# ── Save ihara_poles.json ─────────────────────────────────────────────────────



out = Dict(
    "poles"            => poles,
    "spectral_radii"   => spectral_radii,
    "ramanujan_bounds" => ramanujan_bounds,
    "n_snapshots"      => n_snapshots,
    "n_poles_per_snap" => n_poles_per_snap,
    "graph_type"       => graph_type,
    "ramanujan_bound"  => ramanujan_bound,
    "q_max"            => q_max,
    "rho_B_ihara"      => maximum(abs.(eigs_B)),
)

outpath = joinpath(folder, "ihara_poles.json")
open(outpath, "w") do io
    JSON3.write(io, out)
end

println()
println("Saved: $outpath")
@printf("  n_snapshots      = %d\n", n_snapshots)
@printf("  n_poles_per_snap = %d\n", n_poles_per_snap)
@printf("  total poles      = %d\n", length(poles))
@printf("  ramanujan_bound  = %.4f\n", ramanujan_bound)
println()
println("Now run: julia run_test_klein.jl ./")
