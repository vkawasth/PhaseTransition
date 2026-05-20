# run_bridge_b_phase2_standalone.jl
# Standalone Phase 2 Bridge B.
# No Glob. No external deps beyond JSON3.
# Arrow format: {idx, src, tgt, v_src, v_tgt, weight, sym, rev_idx}
#
# Usage:
#   julia run_bridge_b_phase2_standalone.jl ./ Q_7P connectome_graphs.json

using JSON3, LinearAlgebra, Printf, Statistics

folder      = length(ARGS) >= 1 ? ARGS[1] : "./"
graph_type  = length(ARGS) >= 2 ? ARGS[2] : "Q_7P"
graphs_json = length(ARGS) >= 3 ? ARGS[3] : "connectome_graphs.json"

println("=== Bridge B Phase 2 (standalone) ===")
println("Folder:  $folder")
println("Graph:   $graph_type")
println()

# ── Load graph data ────────────────────────────────────────────────────────
gdata = JSON3.read(read(graphs_json, String))
ginfo = gdata[graph_type]

n_v   = Int(ginfo["n_v"])
n_arr = Int(ginfo["n_arr"])

# b1 and q_max hardcoded per graph (not in JSON)
b1, q_max = if graph_type == "Q_7P"; (2, 5)
            elseif graph_type == "Q_8";  (2, 5)
            elseif graph_type == "Q_7L"; (1, 4)
            else;                         (1, 4)  # Q_6
            end

println("n_v=$n_v  n_arr=$n_arr  b1=$b1  q_max=$q_max")

# ── Build B_Ihara from arrows ──────────────────────────────────────────────
# Arrow format: {idx, src, tgt, v_src, v_tgt, weight, sym, rev_idx}
# idx is 1-based. B[j,i]=1 if tgt[i]==src[j] and j != rev_idx[i]
arrows_raw = ginfo["arrows"]
n = length(arrows_raw)
println("Building B_Ihara ($n × $n)...")

# Extract src vertex index, tgt vertex index, rev_idx for each arrow
# Use v_src/v_tgt (integer vertex indices) directly
srcs    = Int[]  # source vertex (1-based after +1)
tgts    = Int[]  # target vertex
rev_ids = Int[]  # reverse arrow idx (1-based)

for a in arrows_raw
    push!(srcs,    Int(a["v_src"]))
    push!(tgts,    Int(a["v_tgt"]))
    push!(rev_ids, Int(a["rev_idx"]))
end

println("Arrow 1: $(srcs[1])→$(tgts[1])  rev=$(rev_ids[1])")
println("Arrow 2: $(srcs[2])→$(tgts[2])  rev=$(rev_ids[2])")

B = zeros(Float64, n, n)
for i in 1:n
    for j in 1:n
        if tgts[i] == srcs[j] && j != rev_ids[i]
            B[j, i] = 1.0
        end
    end
end

evals = eigvals(B)
rho   = maximum(abs.(evals))
@printf("ρ(B_Ihara) = %.4f\n", rho)
@printf("ρ/√q_max   = %.4f\n\n", rho / sqrt(Float64(q_max)))

# ── Load Plücker phases ────────────────────────────────────────────────────
plucker_path = joinpath(folder, "plucker_phase.json")
pp_raw = JSON3.read(read(plucker_path, String))
phases = if haskey(pp_raw, :phase)
    Float64.(pp_raw[:phase])
elseif haskey(pp_raw, "phase")
    Float64.(pp_raw["phase"])
else
    Float64.(pp_raw)
end
println("Loaded $(length(phases)) Plücker phases")
@printf("Phase range: [%.4f, %.4f]  std=%.4f\n\n",
        minimum(phases), maximum(phases), std(phases))

# ── Load snapshot files ────────────────────────────────────────────────────
files = sort(filter(f -> startswith(basename(f), "ainf_export_") &&
                         endswith(f, ".json"),
              readdir(folder, join=true)))
println("Found $(length(files)) ainf_export JSON files")
n_snap = min(length(files), length(phases))

# ── Detect wall crossings ──────────────────────────────────────────────────
threshold  = 0.01
wall_steps = Int[]
wall_dphis = Float64[]

let prev = phases[1]
    for i in 2:n_snap
        Δφ = phases[i] - prev
        if abs(Δφ) > threshold
            push!(wall_steps, i)
            push!(wall_dphis, Δφ)
        end
        prev = phases[i]
        i % 200 == 0 && @printf("  %d/%d  walls=%d\n", i, n_snap, length(wall_steps))
    end
end

println("Wall crossings:  $(length(wall_steps))")
println()

if isempty(wall_steps)
    println("No wall crossings detected.")
else
    # ── Accumulate Phi_KS ─────────────────────────────────────────────────
    Phi = let acc = Matrix{Float64}(I, b1, b1)
        for Δφ in wall_dphis
            c, s_val = cos(Δφ), sin(Δφ)
            R = b1 == 2 ? [c -s_val; s_val c] : reshape([c], 1, 1)
            acc = R * acc
        end
        acc
    end

    tr_phi  = tr(Phi)
    det_phi = det(Phi)

    println("Φ_KS matrix:")
    for row in eachrow(Phi)
        println("  ", join([@sprintf("%+.6f", x) for x in row], "  "))
    end
    @printf("tr(Φ_KS)  = %.6f\n", tr_phi)
    @printf("det(Φ_KS) = %.6f\n\n", det_phi)

    a1 = -tr_phi
    a2 =  det_phi
    @printf("det(I-u·Φ_KS)      = 1 %+.6f·u %+.6f·u²\n", a1, a2)

    lam1, lam2 = 1.7898, -0.8021
    ih_a1 = -(lam1 + lam2)
    ih_a2 =  lam1 * lam2
    @printf("ζ_Ihara^{-1}|_{H1} = 1 %+.6f·u %+.6f·u²\n\n", ih_a1, ih_a2)

    println("=== Bridge B Assessment ===")
    if abs(a1 - ih_a1) < 0.05 && abs(a2 - ih_a2) < 0.05
        println("✓  BRIDGE B CONFIRMED")
    elseif abs(tr_phi) > 2.001
        println("~  Φ_KS HYPERBOLIC (|tr|=$(round(abs(tr_phi),digits=6)) > 2)")
        @printf("   u coeff:  got=%+.4f  target=%+.4f\n", a1, ih_a1)
        @printf("   u² coeff: got=%+.4f  target=%+.4f\n", a2, ih_a2)
    elseif abs(tr_phi - 2.0) < 0.001
        println("~  Φ_KS PARABOLIC (tr=2.000 exactly) — boundary Dehn twist")
    elseif abs(tr_phi + 2.0) < 0.001
        println("~  Φ_KS PARABOLIC (tr=-2.000) — boundary Dehn twist")
    else
        @printf("~  Φ_KS ELLIPTIC (|tr|=%.6f < 2)\n", abs(tr_phi))
    end

    # ── Write bridge_b_result.txt ──────────────────────────────────────────
    open("bridge_b_result.txt", "w") do io
        println(io, "Bridge B Result — $graph_type (Phase 2)")
        println(io, "="^50)
        println(io, "Graph:         $graph_type")
        println(io, "n_v=$n_v  n_arr=$n_arr  b1=$b1")
        println(io, "Surface:       $(b1==1 ? "cylinder" : "trinion")")
        println(io, "Snapshots:     $n_snap")
        println(io, "Wall crossings: $(length(wall_steps))")
        println(io, "")
        @printf(io, "ρ(B_Ihara) = %.4f\n", rho)
        @printf(io, "ρ/√q_max   = %.4f\n", rho/sqrt(Float64(q_max)))
        println(io, "")
        println(io, "Phase trajectory (first 30 crossings):")
        let prev2 = phases[1]
            for i in wall_steps[1:min(30,end)]
                Δφ = phases[i] - prev2
                @printf(io, "  Step %3d: %.4f → %.4f  Δφ=%+.4f\n",
                        i, prev2, phases[i], Δφ)
                prev2 = phases[i]
            end
        end
        println(io, "")
        @printf(io, "det(I-u·Φ_KS)      = 1 %+.6f·u %+.6f·u²\n", a1, a2)
        @printf(io, "ζ_Ihara^{-1}|_H1   = 1 %+.6f·u %+.6f·u²\n", ih_a1, ih_a2)
    end
    println("\nWrote bridge_b_result.txt")
end
