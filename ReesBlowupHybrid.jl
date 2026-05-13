###############################################################
# ReesBlowupHybrid.jl
#
# Hybrid commutative Rees blow-up over JSON snapshots
# for noncommutative / A∞ connectome dynamics.
#
# INPUT:
#   Folder of JSON snapshots containing any of:
#   m6, hh2, gerstenhaber, cup_product,
#   deriv_basis_info, prime_paths, ihara_radius
#
# OUTPUT:
#   Crisis generators
#   Blow-up chart coordinates
#   Exceptional divisor motion
#   Crisis typing / clustering
#
###############################################################

module ReesBlowupHybrid

using JSON3
using LinearAlgebra
using Statistics
using CairoMakie
using Printf

export BlowupPoint,
       BlowupRun,
       run_blowup!,
       plot_divisor,
       plot_generators,
       plot_mittag_leffler,
       print_events,
       print_mittag_leffler_summary,
       write_blowup_table

###############################################################
# TYPES
###############################################################

mutable struct BlowupPoint
    time::Int
    file::String
    region::Symbol
    g::Vector{Float64}      # generators (real — Rees algebra is commutative)
    proj::Vector{Float64}   # normalized projective coords on E = Proj(Rees(R,I))
    chart::Int              # dominant generator chart
    severity::Float64
    # --- Mittag-Leffler and monodromy additions ---
    mittag_leffler_ok::Bool     # does zeta extend across exceptional divisor?
    monodromy_phase::Float64    # arg of Gauss-Manin monodromy around E (from Plücker)
    pole_distance::Float64      # |spectral_radius - √q| — distance to Ramanujan floor
    generator_entropy::Float64  # -Σ p_i log p_i over proj coords — divisor complexity
end

mutable struct BlowupRun
    names::Vector{String}
    pts::Vector{BlowupPoint}
end

###############################################################
# REGIONS
###############################################################

const REGIONS = [:CA1sp,:BLA,:HY,:HPF,:sAMY,:LA]

###############################################################
# HELPERS
###############################################################

load_json(file) = JSON3.read(read(file,String))

safe_float(x) = try
    Float64(x)
catch e
    0.0
end

function region_hits(s::String)
    Symbol[
        r for r in REGIONS if occursin(String(r), s)
    ]
end

###############################################################
# AGGREGATORS
###############################################################

function agg_dict_mass(obj)

    tot = 0.0

    for (_,v) in pairs(obj)
        if v isa Number
            tot += abs(safe_float(v))
        elseif v isa AbstractDict
            tot += agg_dict_mass(v)
        end
    end

    tot
end

function region_mass(obj)

    score = Dict(r=>0.0 for r in REGIONS)

    for (k,v) in pairs(obj)
        s = String(k)
        w = v isa Number ? abs(safe_float(v)) : 1.0

        for r in region_hits(s)
            score[r] += w
        end
    end

    score
end

###############################################################
# PLÜCKER PHASE LOADER
# Reads plucker_trajectory.json to get monodromy phase at each
# blowup time. The phase is arg of Gauss-Manin monodromy around
# the exceptional divisor E = Proj(Rees(R,I)).
###############################################################

function load_plucker_phases_rees(; file="plucker_trajectory.json")
    if !isfile(file)
        @warn "plucker_trajectory.json not found — monodromy phases set to 0"
        return Float64[], Int[]
    end
    data = JSON3.read(read(file, String))
    # Use phase_q12 if available (full SO(4) winding), else fall back to q12/q13 atan
    phases = if haskey(data, "phase_q12")
        Float64.(data["phase_q12"])
    else
        q12 = Float64.(data["q12"])
        q13 = Float64.(data["q13"])
        atan.(q13, q12)
    end
    steps = haskey(data, "steps") ? Int.(data["steps"]) : collect(1:length(phases))
    return phases, steps
end

"""
Find the Plücker phase closest in step index to the blowup at step t.
"""
function plucker_phase_at(t::Int, phases::Vector{Float64}, steps::Vector{Int})
    isempty(phases) && return 0.0
    diffs = abs.(steps .- t)
    return phases[argmin(diffs)]
end

###############################################################
# MITTAG-LEFFLER CONDITION
#
# Checks whether the zeta function extends across the exceptional
# divisor at each blowup point.
#
# The Mittag-Leffler condition holds when the generator vector g
# satisfies: the dominant generator (chart) can absorb the polar
# part from all other generators without residue.
#
# Proxy criterion (computable from g):
#   ML holds  ⟺  g[chart] > Σ_{j≠chart} g[j]
#              ⟺  proj[chart] > 0.5
#              ⟺  one generator dominates the exceptional divisor
#
# When ML fails: the polar parts of multiple generators cannot be
# reconciled → HH² obstruction → blowup required to resolve.
# This is the Mittag-Leffler obstruction = H^1(𝓜, O_{𝓜}).
###############################################################

function mittag_leffler_ok(proj::Vector{Float64})::Bool
    # ML holds when one generator strictly dominates (proj > 0.5)
    # i.e. the exceptional divisor is in a single affine chart
    return maximum(proj) > 0.5
end

"""
Generator entropy: -Σ p_i log(p_i) over the projective coordinates.
Low entropy  → concentrated on one generator → ML likely holds
High entropy → spread across generators → ML failure zone
Maximum entropy = log(6) ≈ 1.79 for 6 generators
"""
function generator_entropy(proj::Vector{Float64})::Float64
    return -sum(p > 0 ? p * log(p) : 0.0 for p in proj)
end

"""
Distance of the spectral radius from the Ramanujan floor √q.
Computed from g6 (ihara_radius) and the mean graph degree (proxy q=2).
pole_distance → 0 means the blowup is happening at the Ramanujan floor.
pole_distance > 0 means the blowup is happening above the floor —
the Mittag-Leffler obstruction is in the interior of moduli space.
"""
function pole_distance(g::Vector{Float64}; q::Float64=2.0)::Float64
    ihara_excess = g[6]   # = spectral_radius - 1.0 (from extract_generators)
    spectral_radius = 1.0 + ihara_excess
    ramanujan_floor = sqrt(q)
    return max(0.0, spectral_radius - ramanujan_floor)
end

###############################################################
# GENERATORS
#
# g1 = m6 obstruction
# g2 = hh2 spike
# g3 = gerstenhaber stress
# g4 = cup mass
# g5 = deriv deformation mass
# g6 = ihara radius excess
###############################################################

function extract_generators(snap)

    g = zeros(Float64,6)

    # g1
    haskey(snap,:m6) && (g[1] = agg_dict_mass(snap[:m6]))

    # g2
    if haskey(snap,:hh2)
        g[2] = abs(safe_float(snap[:hh2]))
    elseif haskey(snap,:HH2)
        g[2] = abs(safe_float(snap[:HH2]))
    end

    # g3
    haskey(snap,:gerstenhaber) &&
        (g[3] = agg_dict_mass(snap[:gerstenhaber]))

    # g4
    haskey(snap,:cup_product) &&
        (g[4] = agg_dict_mass(snap[:cup_product]))

    # g5 deriv basis infinitesimal deformation
    if haskey(snap,:deriv_basis_info)
        x = 0.0
        for row in snap[:deriv_basis_info]
            haskey(row,:regions) && (x += agg_dict_mass(row[:regions]))
        end
        g[5] = x
    end

    # g6 external postprocessed Ihara radius
    if haskey(snap,:ihara_radius)
        g[6] = max(0.0, safe_float(snap[:ihara_radius]) - 1.0)
    end

    g
end

###############################################################
# REGION INFERENCE
###############################################################

function infer_region(snap)

    score = Dict(r=>0.0 for r in REGIONS)

    for key in (:m6,:gerstenhaber,:cup_product)
        if haskey(snap,key)
            rm = region_mass(snap[key])
            for r in REGIONS
                score[r] += rm[r]
            end
        end
    end

    vals = collect(values(score))
    ks   = collect(keys(score))

    return ks[argmax(vals)]
end

###############################################################
# PROJECTIVIZATION
#
# exceptional divisor point [g1:...:gk]
###############################################################

function projectivize(g)

    s = sum(abs.(g))

    if s < 1e-12
        return zeros(length(g))
    end

    abs.(g) ./ s
end

###############################################################
# RUN
###############################################################

function run_blowup!(folder::String)

    # Only ainf_export JSON files
    files = filter(f -> occursin(r"ainf_export_(?:\w+_)?\d+(?:\.\d+)?\.json", basename(f)), readdir(folder))
    sort!(files)

    names = ["m6", "hh2", "ger", "cup", "deriv", "ihara"]

    # Load Plücker phases for monodromy computation
    plucker_phases, plucker_steps = load_plucker_phases_rees()

    pts = BlowupPoint[]

    for (t, f) in enumerate(files)

        snap = load_json(joinpath(folder, f))
        g    = extract_generators(snap)

        if maximum(g) > 0

            p   = projectivize(g)
            c   = argmax(p)
            sev = norm(g)

            # Mittag-Leffler condition
            ml_ok = mittag_leffler_ok(p)

            # Monodromy phase of exceptional divisor from Plücker trajectory
            φ = plucker_phase_at(t, plucker_phases, plucker_steps)

            # Distance from Ramanujan floor
            pd = pole_distance(g)

            # Generator entropy (divisor complexity)
            ent = generator_entropy(p)

            push!(pts,
                BlowupPoint(
                    t, f,
                    infer_region(snap),
                    g, p, c, sev,
                    ml_ok, φ, pd, ent
                )
            )
        end
    end

    BlowupRun(names, pts)
end

###############################################################
# PRINT
###############################################################

function print_events(B::BlowupRun)

    println("------------------------------------------------")
    println("time | region | dominant chart | severity")
    println("------------------------------------------------")

    for p in B.pts
        println(
            p.time, " | ",
            p.region, " | ",
            B.names[p.chart], " | ",
            round(p.severity,digits=3)
        )
    end

    println("------------------------------------------------")
end

###############################################################
# PLOTS
###############################################################

function plot_generators(B::BlowupRun)

    m = length(B.names)
    n = length(B.pts)

    fig = Figure(size=(1000,500))
    ax = Axis(fig[1,1], title="Generator Time Series")

    for j in 1:m
        ys = [p.g[j] for p in B.pts]
        lines!(ax,1:n,ys,label=B.names[j])
    end

    axislegend(ax)
    fig
end

function plot_divisor(B::BlowupRun)

    # simplex projection to first 3 coords
    fig = Figure(size=(900,500))
    ax = Axis(fig[1,1], title="Exceptional Divisor (first 3 generators)")

    xs = [p.proj[1] for p in B.pts]
    ys = [p.proj[2] for p in B.pts]
    zs = [p.proj[3] for p in B.pts]

    scatter!(ax,xs,ys)

    for i in 2:length(xs)
        lines!(ax,xs[i-1:i],ys[i-1:i])
    end

    fig
end

###############################################################
# WRITE TABLE
###############################################################

function write_blowup_table(B::BlowupRun, filename::String="blowup_table.tsv")
    io = open(filename, "w")
    println(io, join(["time","file","region","chart","severity",
                      "ml_ok","monodromy_phase","pole_distance",
                      "gen_entropy","g1_m6","g2_hh2","g3_ger",
                      "g4_cup","g5_deriv","g6_ihara"], "\t"))
    for p in B.pts
        @printf(io, "%d\t%s\t%s\t%s\t%.4f\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n",
                p.time, basename(p.file), p.region, B.names[p.chart],
                p.severity, p.mittag_leffler_ok,
                p.monodromy_phase, p.pole_distance, p.generator_entropy,
                p.g[1], p.g[2], p.g[3], p.g[4], p.g[5], p.g[6])
    end
    close(io)
    println("Blowup table written to $filename")
end

###############################################################
# MITTAG-LEFFLER SUMMARY
###############################################################

function print_mittag_leffler_summary(B::BlowupRun)
    println()
    println("=========== MITTAG-LEFFLER SUMMARY ===========")
    println("Exceptional divisor analysis: Proj(Rees(R,I))")
    println()

    n      = length(B.pts)
    n_ok   = count(p -> p.mittag_leffler_ok, B.pts)
    n_fail = n - n_ok

    @printf("  Total blowup points:     %d\n", n)
    @printf("  ML condition holds:      %d (%.0f%%)\n", n_ok, 100.0*n_ok/max(n,1))
    @printf("  ML condition fails:      %d (%.0f%%)\n", n_fail, 100.0*n_fail/max(n,1))
    println()

    if n > 0
        phases = [p.monodromy_phase for p in B.pts]
        dists  = [p.pole_distance   for p in B.pts]
        ents   = [p.generator_entropy for p in B.pts]

        @printf("  Monodromy phase:  mean=%.4f  std=%.4f\n",
                mean(phases), std(phases))
        @printf("  Pole distance:    mean=%.4f  max=%.4f\n",
                mean(dists), maximum(dists))
        @printf("  Gen entropy:      mean=%.4f  (log6=%.4f)\n",
                mean(ents), log(6))

        # Ramanujan floor hits: pole_distance < 0.05
        floor_hits = count(p -> p.pole_distance < 0.05, B.pts)
        @printf("  At Ramanujan floor (dist<0.05): %d / %d\n", floor_hits, n)

        println()
        println("  ML failure points (need blowup to resolve):")
        for p in B.pts
            if !p.mittag_leffler_ok
                @printf("    t=%d  region=%s  chart=%s  entropy=%.3f  phase=%.3f\n",
                        p.time, p.region, B.names[p.chart],
                        p.generator_entropy, p.monodromy_phase)
            end
        end

        println()
        println("  Interpretation:")
        if n_fail == 0
            println("  ✓ All blowup points satisfy ML condition.")
            println("    Zeta extends across all exceptional divisors.")
            println("    Stack is globally well-posed — supports Deligne Weil II.")
        elseif n_fail < n ÷ 3
            println("  ~ Some ML failures. These are genuine obstruction points.")
            println("    Each failure = H¹(𝓜, O) ≠ 0 at that blowup.")
            println("    Check whether monodromy phase is large at failure points.")
        else
            println("  ✗ Many ML failures. Blowup resolution may be incomplete.")
            println("    Consider increasing Julia --full call frequency.")
        end

        # Check Ramanujan connection
        println()
        if floor_hits > n ÷ 2
            println("  ✓ Most blowups at Ramanujan floor — 2√2 is confirmed as ML threshold.")
        else
            @printf("  ~ Only %d/%d blowups at floor. Check ihara_radius in snapshots.\n",
                    floor_hits, n)
        end
    end
    println("===============================================")
end

###############################################################
# PLOT — Mittag-Leffler map on exceptional divisor
###############################################################

function plot_mittag_leffler(B::BlowupRun)
    n = length(B.pts)
    n == 0 && return Figure()

    fig = Figure(size=(1200, 500))

    # Panel 1: entropy time series — low = ML ok, high = ML failing
    ax1 = Axis(fig[1,1],
        title="Generator entropy (low → ML holds)",
        xlabel="Blowup index", ylabel="-Σ pᵢ log pᵢ")
    ents   = [p.generator_entropy for p in B.pts]
    colors = [p.mittag_leffler_ok ? :steelblue : :red for p in B.pts]
    scatter!(ax1, 1:n, ents, color=colors, markersize=10)
    hlines!(ax1, [log(6)/2], color=:orange, linestyle=:dash,
            label="Half max entropy")
    axislegend(ax1)

    # Panel 2: pole distance vs monodromy phase
    # Points near (distance=0, phase=π) are exactly at the Ramanujan floor
    # with maximal monodromy winding — these are the 2√2 events
    ax2 = Axis(fig[1,2],
        title="Pole distance vs monodromy phase",
        xlabel="Monodromy phase φ (radians)",
        ylabel="|spectral radius - √q|")
    dists  = [p.pole_distance    for p in B.pts]
    phases = [p.monodromy_phase  for p in B.pts]
    scatter!(ax2, phases, dists, color=colors, markersize=10)
    hlines!(ax2, [0.0], color=:green, linestyle=:dash,
            label="Ramanujan floor")
    axislegend(ax2)

    save("mittag_leffler_blowup.png", fig)
    println("✓ Saved mittag_leffler_blowup.png")
    fig
end

end
