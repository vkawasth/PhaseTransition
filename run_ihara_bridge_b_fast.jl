"""
run_ihara_bridge_b_fast.jl
==========================
Fast Bridge B pipeline. Reads ainf_export JSON snapshots,
extracts monodromy phase, detects wall crossings, computes
Phi_KS, and checks Bridge B against the Ihara zeta.

Usage:
  julia run_ihara_bridge_b_fast.jl <json_folder> <graph_type> [graphs_json]

  graph_type : Q_6 | Q_7P | Q_7L | Q_8
  graphs_json: path to connectome_graphs.json (optional,
               defaults to ./connectome_graphs.json)

What it reads from JSON snapshots:
  prime_paths weights → monodromy phase scalar
  score               → one changing field

What it reads from connectome_graphs.json:
  arrows, rev map, weights → B_Ihara matrix
  b1                       → H1 dimension
  n_v, n_arr               → matrix sizes

What it computes:
  1. Monodromy phase per snapshot (fast scalar)
  2. Wall crossings (phase jumps > 0.01)
  3. H1 restriction matrix per wall crossing
  4. Phi_KS = product of wall-crossing H1 matrices
  5. det(I - u*Phi_KS) as polynomial
  6. B_Ihara from graph data
  7. Ihara zeta via det(I - u*B_Ihara)
  8. Bridge B comparison

Output:
  blowup_table_v2.tsv  — per-snapshot data with H1 entries
  bridge_b_result.txt  — Bridge B verification summary
"""

using JSON3
using Printf
using LinearAlgebra

# ── ARGS ──────────────────────────────────────────────────────────────────────

if length(ARGS) < 2
    println("Usage: julia run_ihara_bridge_b_fast.jl <folder> <graph_type> [graphs_json]")
    println("  graph_type: Q_6 | Q_7P | Q_7L | Q_8")
    exit(1)
end

folder      = ARGS[1]
graph_type  = ARGS[2]
graphs_json = length(ARGS) >= 3 ? ARGS[3] : joinpath(dirname(@__FILE__), "connectome_graphs.json")

println("=== Bridge B Fast Pipeline ===")
println("Folder:      $folder")
println("Graph:       $graph_type")
println("Graphs JSON: $graphs_json")
println()

# ── Load graph data from connectome_graphs.json ───────────────────────────────

println("Loading graph data...")
gdata_all = JSON3.read(read(graphs_json, String))

if !haskey(gdata_all, Symbol(graph_type))
    println("ERROR: graph type '$graph_type' not found in $graphs_json")
    println("Available: $(keys(gdata_all))")
    exit(1)
end

gdata   = gdata_all[Symbol(graph_type)]
n_v     = Int(gdata[:n_v])
arrows  = gdata[:arrows]
n_arr   = Int(gdata[:n_arr])

# Build reversal map: rev[i] = j if arrow j reverses arrow i, else 0
rev_map = zeros(Int, n_arr)
for a in arrows
    if Int(a["rev_idx"]) > 0
        rev_map[Int(a["idx"])] = Int(a["rev_idx"])
    end
end

# b1 = first Betti number
n_sym = sum(1 for a in arrows if Bool(a["sym"]) && Int(a["idx"]) < Int(a["rev_idx"]))
b1 = n_sym - n_v + 1

println("  n_v=$n_v  n_arr=$n_arr  n_sym_pairs=$n_sym  b1=$b1")
println()

# ── Build B_Ihara from graph data ─────────────────────────────────────────────

println("Building B_Ihara ($n_arr × $n_arr)...")

B_Ihara = zeros(Float64, n_arr, n_arr)
for ai in arrows
    for aj in arrows
        i = Int(ai["idx"]); j = Int(aj["idx"])
        v_tgt_i = Int(ai["v_tgt"])
        v_src_j = Int(aj["v_src"])
        # Composable and not backtracking
        if v_tgt_i == v_src_j && !(Bool(ai["sym"]) && rev_map[i] == j)
            B_Ihara[i, j] = 1.0
        end
    end
end

# Spectral radius of B_Ihara
eigs_B = eigvals(B_Ihara)
rho_B  = maximum(abs.(eigs_B))
# q_max from nonbacktracking out-degrees
q_vals = [sum(B_Ihara[i,:]) for i in 1:n_arr]
q_max  = maximum(q_vals)
println("  ρ(B_Ihara) = $(round(rho_B, digits=4))")
println("  ρ/√q_max   = $(round(rho_B/sqrt(q_max), digits=4))")
println()

# Ihara zeta on H1:
# For b1=1 (cylinder): det(I - u*B_Ihara|_{H1}) = 1 - λ_1*u
# For b1=2 (trinion):  det(I - u*B_Ihara|_{H1}) = 1 - tr(M)*u + det(M)*u^2
# The nontrivial eigenvalues of B_Ihara are those not from (1-u^2)^{|E|-|V|}
# Separate trivial (±1) from nontrivial eigenvalues
eigs_sorted = sort(abs.(eigs_B), rev=true)
trivial_threshold = 1.05  # eigenvalues near ±1 are trivial
nontrivial_eigs = [e for e in eigs_B if abs(abs(e) - 1.0) > 0.05 && abs(e) > 0.01]

println("B_Ihara eigenvalue summary:")
println("  All eigenvalues (sorted by magnitude):")
for e in sort(eigs_B, by=abs, rev=true)[1:min(10,n_arr)]
    println("    $(round(real(e),digits=4)) + $(round(imag(e),digits=4))i  |λ|=$(round(abs(e),digits=4))")
end
println()

# ── Load JSON snapshots ───────────────────────────────────────────────────────

files = sort(filter(f -> endswith(f,".json") && startswith(f,"ainf_export"),
                    readdir(folder)))
println("Found $(length(files)) ainf_export JSON files")
isempty(files) && (println("ERROR: no JSON files found in $folder"); exit(1))

# ── Per-snapshot processing ───────────────────────────────────────────────────

function compute_phase(snap)::Float64
    haskey(snap, :prime_paths) || return 0.0
    pp = snap[:prime_paths]; isempty(pp) && return 0.0
    total_lw = 0.0
    for p in pp
        haskey(p,:weight) || continue
        w = Float64(p[:weight])
        (w > 0 && isfinite(w)) && (total_lw += log10(w))
    end
    θ_geo  = π * min(total_lw/30.0, 0.5)
    θ_path = 0.05π * min(length(pp)/10.0, 0.2)
    return θ_geo + θ_path
end

function h1_matrix(Δφ::Float64, b1::Int)
    # Rotation matrix of angle Δφ restricted to H1
    # b1=1: scalar cos(Δφ) (real part only, H1=Z^1)
    # b1=2: 2×2 rotation matrix
    if b1 == 1
        return reshape([cos(Δφ)], 1, 1)
    else
        return [cos(Δφ) -sin(Δφ); sin(Δφ) cos(Δφ)]
    end
end

println("Processing snapshots...")
results    = []
wall_steps = Int[]

let prev_phase = NaN
    for (i, f) in enumerate(files)
        snap = JSON3.read(read(joinpath(folder,f), String))
        phase = compute_phase(snap)
        tw    = let pp=get(snap,:prime_paths,[]);
                    isempty(pp) ? 0.0 : maximum(Float64(get(p,:weight,0.0)) for p in pp)
                end
        np    = length(get(snap,:prime_paths,[]))
        sc    = Float64(get(snap,:score, 0.0))

        is_wall = !isnan(prev_phase) && abs(phase - prev_phase) > 0.01

        push!(results, (step=i, file=f, score=sc, n_paths=np,
                        top_w=tw, phase=phase, is_wall=is_wall))
        is_wall && push!(wall_steps, i)
        prev_phase = phase
        i % 200 == 0 && @printf("  %d/%d  walls=%d\n", i, length(files), length(wall_steps))
    end
end

println()
println("Total snapshots: $(length(results))")
println("Wall crossings:  $(length(wall_steps))")
println()

# ── Wall-crossing operators and Phi_KS ───────────────────────────────────────

println("=== Wall-Crossing Analysis ===")
println()

wc_data = []
for i in wall_steps
    Δφ = results[i].phase - results[i-1].phase
    M  = h1_matrix(Δφ, b1)
    push!(wc_data, (step=i, delta_phi=Δφ,
                    phase_before=results[i-1].phase,
                    phase_after=results[i].phase, M=M))
    if b1 == 1
        @printf("  T%d (step %4d): φ %+.4f → %+.4f  Δφ=%+.4f  M=[%.6f]\n",
                length(wc_data), i,
                results[i-1].phase, results[i].phase, Δφ, M[1,1])
    else
        @printf("  T%d (step %4d): φ %+.4f → %+.4f  Δφ=%+.4f\n",
                length(wc_data), i,
                results[i-1].phase, results[i].phase, Δφ)
        @printf("    M = [[%.4f, %.4f], [%.4f, %.4f]]\n",
                M[1,1], M[1,2], M[2,1], M[2,2])
    end
end

println()

# Phi_KS = product of wall-crossing matrices (ordered by step)
if isempty(wc_data)
    println("No wall crossings found — cannot compute Phi_KS")
else
    if b1 == 1
        Phi_KS = reduce(*, [d.M[1,1] for d in wc_data])
        println("Φ_KS (1×1): $Phi_KS")
        det_poly_coeffs = [1.0, -Phi_KS]  # 1 - Phi_KS*u
    else
        Phi_KS = reduce(*, [d.M for d in wc_data])
        println("Φ_KS (2×2):")
        @printf("  [[%.6f, %.6f],\n   [%.6f, %.6f]]\n",
                Phi_KS[1,1], Phi_KS[1,2], Phi_KS[2,1], Phi_KS[2,2])
        tr_Phi  = tr(Phi_KS)
        det_Phi = det(Phi_KS)
        det_poly_coeffs = [1.0, -tr_Phi, det_Phi]  # 1 - tr*u + det*u^2
    end

    println()

    # ── Bridge B comparison ───────────────────────────────────────────────────

    println("=== Bridge B Comparison ===")
    println()

    # det(I - u*Phi_KS) from wall-crossing data
    if b1 == 1
        println("det(I - u·Φ_KS)  = 1 - $(round(Phi_KS, digits=6))·u")
    else
        println("det(I - u·Φ_KS)  = 1 - $(round(tr_Phi,digits=6))·u + $(round(det_Phi,digits=6))·u²")
    end
    println()

    # Ihara zeta restricted to H1 from B_Ihara nontrivial eigenvalues
    # For b1=1: ζ^{-1}|_{H1} = 1 - λ_1·u  where λ_1 = dominant nontrivial eig
    # For b1=2: ζ^{-1}|_{H1} = (1-λ_1·u)(1-λ_2·u) = 1-(λ_1+λ_2)u+λ_1λ_2·u²
    # Use the largest b1 nontrivial eigenvalues of B_Ihara
    nt_eigs = sort([e for e in eigs_B if abs(e) > 0.1 && abs(abs(e)-1.0) > 0.05],
                   by=abs, rev=true)

    if b1 == 1 && !isempty(nt_eigs)
        λ1 = real(nt_eigs[1])
        println("ζ_Ihara^{-1}|_{H1} = 1 - $(round(λ1,digits=6))·u")
        println()
        match = abs(Phi_KS - λ1) / (abs(λ1) + 1e-10)
        println("Bridge B check (b1=1):")
        println("  Φ_KS              = $(round(Phi_KS, digits=6))")
        println("  λ_1(B_Ihara)|_{H1} = $(round(λ1, digits=6))")
        println("  Relative difference = $(round(match*100, digits=2))%")
        println()
        if match < 0.05
            println("  ✓ Bridge B CONSISTENT (< 5% difference)")
        elseif match < 0.20
            println("  ~ Bridge B APPROXIMATELY holds (< 20% difference)")
            println("    Note: H1 matrix uses U(1) approximation — full matrix needed")
        else
            println("  ✗ Bridge B NOT confirmed at this approximation level")
            println("    The U(1) phase approximation may be insufficient.")
            println("    Need full KS monodromy matrices from matched JSON+blowup data.")
        end
    elseif b1 == 2 && length(nt_eigs) >= 2
        λ1, λ2 = real(nt_eigs[1]), real(nt_eigs[2])
        ihara_tr  = λ1 + λ2
        ihara_det = λ1 * λ2
        println("ζ_Ihara^{-1}|_{H1} = 1 - $(round(ihara_tr,digits=6))·u + $(round(ihara_det,digits=6))·u²")
        println()
        diff_tr  = abs(tr_Phi  - ihara_tr)  / (abs(ihara_tr)  + 1e-10)
        diff_det = abs(det_Phi - ihara_det) / (abs(ihara_det) + 1e-10)
        println("Bridge B check (b1=2):")
        println("  tr(Φ_KS)   = $(round(tr_Phi,digits=6))    vs  λ₁+λ₂ = $(round(ihara_tr,digits=6))    diff=$(round(diff_tr*100,digits=2))%")
        println("  det(Φ_KS)  = $(round(det_Phi,digits=6))   vs  λ₁λ₂  = $(round(ihara_det,digits=6))   diff=$(round(diff_det*100,digits=2))%")
        println()
        if max(diff_tr, diff_det) < 0.05
            println("  ✓ Bridge B CONSISTENT (both < 5%)")
        elseif max(diff_tr, diff_det) < 0.20
            println("  ~ Bridge B APPROXIMATELY holds")
        else
            println("  ✗ Bridge B NOT confirmed at this approximation level")
        end
    end

    # ── Write blowup_table_v2.tsv ─────────────────────────────────────────────
    outfile = joinpath(folder, "blowup_table_v2.tsv")
    open(outfile, "w") do io
        hdr = "step\tfile\tscore\tn_paths\ttop_weight\tmonodromy_phase\twall_crossing"
        hdr *= b1==1 ? "\tH1_11" : "\tH1_11\tH1_12\tH1_21\tH1_22"
        println(io, hdr)
        for r in results
            # Find H1 matrix: use identity between wall crossings,
            # wall-crossing matrix at transition steps
            wc_idx = findfirst(d->d.step==r.step, wc_data)
            if wc_idx !== nothing
                M = wc_data[wc_idx].M
            else
                M = Matrix{Float64}(I, b1, b1)
            end
            if b1 == 1
                @printf(io, "%d\t%s\t%.6f\t%d\t%.6e\t%.6f\t%s\t%.8f\n",
                        r.step, r.file, r.score, r.n_paths, r.top_w,
                        r.phase, r.is_wall ? "true" : "false", M[1,1])
            else
                @printf(io, "%d\t%s\t%.6f\t%d\t%.6e\t%.6f\t%s\t%.8f\t%.8f\t%.8f\t%.8f\n",
                        r.step, r.file, r.score, r.n_paths, r.top_w,
                        r.phase, r.is_wall ? "true" : "false",
                        M[1,1], M[1,2], M[2,1], M[2,2])
            end
        end
    end
    println()
    println("Written: $outfile")

    # ── Write bridge_b_result.txt ─────────────────────────────────────────────
    resfile = joinpath(folder, "bridge_b_result.txt")
    open(resfile, "w") do io
        println(io, "Bridge B Result — $graph_type")
        println(io, "="^50)
        println(io, "Graph:         $graph_type")
        println(io, "n_v=$n_v  n_arr=$n_arr  b1=$b1")
        println(io, "Surface:       $(b1==1 ? "cylinder" : "trinion")")
        println(io, "Snapshots:     $(length(results))")
        println(io, "Wall crossings: $(length(wall_steps))")
        println(io, "")
        println(io, "ρ(B_Ihara) = $(round(rho_B,digits=4))")
        println(io, "ρ/√q_max   = $(round(rho_B/sqrt(q_max),digits=4))")
        println(io, "")
        println(io, "Phase trajectory:")
        for d in wc_data
            println(io, "  Step $(d.step): $(round(d.phase_before,digits=4)) → $(round(d.phase_after,digits=4))  Δφ=$(round(d.delta_phi,digits=4))")
        end
        println(io, "")
        if b1 == 1
            println(io, "det(I-u·Φ_KS)     = 1 - $(round(Phi_KS,digits=6))·u")
            !isempty(nt_eigs) &&
            println(io, "ζ_Ihara^{-1}|_{H1} = 1 - $(round(real(nt_eigs[1]),digits=6))·u")
        else
            println(io, "det(I-u·Φ_KS)     = 1 - $(round(tr_Phi,digits=6))·u + $(round(det_Phi,digits=6))·u²")
            length(nt_eigs)>=2 &&
            println(io, "ζ_Ihara^{-1}|_{H1} = 1 - $(round(real(nt_eigs[1])+real(nt_eigs[2]),digits=6))·u + $(round(real(nt_eigs[1])*real(nt_eigs[2]),digits=6))·u²")
        end
    end
    println("Written: $resfile")
end

println()
println("=== Done ===")

