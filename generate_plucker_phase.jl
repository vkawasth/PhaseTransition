###############################################################
# generate_plucker_phase.jl
#
# Generates plucker_phase.json from plucker_trajectory.json.
#
# Background:
#   In the stable zone the A∞-moduli space is nearly flat.
#   All prime paths degenerate to the same geodesic.
#   q12 dominates by factor ~100 over q34, so naive
#   atan(q34, q12) ≈ 0 everywhere → only 72-97 unique values.
#
# Solution: tangent angle of trajectory on Gr(2,4).
#   Direction of motion dq/dt — genuine monodromy phase
#   even in the flat regime because the trajectory moves
#   (score drifts upward along the unique geodesic).
#
# Three sources computed, highest-variation selected:
#   1. Tangent: direction of motion in Plücker space
#   2. Klein:   atan per pair, normalised by pair norm
#   3. Score:   cumulative phase from score drift
###############################################################

using JSON3, Statistics, Printf

println("="^60)
println("PLUCKER PHASE GENERATOR")
println("="^60)

if !isfile("plucker_trajectory.json")
    error("plucker_trajectory.json not found.")
end

d   = JSON3.read(read("plucker_trajectory.json", String))
q12 = Float64.(d["q12"])
q13 = Float64.(d["q13"])
q14 = haskey(d, "q14") ? Float64.(d["q14"]) : zeros(length(q12))
q23 = Float64.(d["q23"])
q24 = Float64.(d["q24"])
q34 = Float64.(d["q34"])
n   = length(q12)

norms = sqrt.(q12.^2 .+ q13.^2 .+ q14.^2 .+ q23.^2 .+ q24.^2 .+ q34.^2)

println("N history points: $n")
@printf("Norm range: %.4f – %.4f\n", minimum(norms), maximum(norms))
@printf("q12 range:  %.4f – %.4f  (dominates by factor %.0fx over q34)\n",
        minimum(q12), maximum(q12), maximum(q12)/max(maximum(q34),1e-10))

# ── SOURCE 1: Tangent angle ───────────────────────────────────────────────
dq12 = diff(q12); dq13 = diff(q13); dq14 = diff(q14)
dq23 = diff(q23); dq24 = diff(q24); dq34 = diff(q34)

tangent_norm = max.(sqrt.(dq12.^2 .+ dq13.^2 .+ dq14.^2 .+
                          dq23.^2 .+ dq24.^2 .+ dq34.^2), 1e-10)

# Two planes: (q12,q13) and (q23,q34) — combine with tangent weighting
phi_t1 = atan.(dq13 ./ tangent_norm, dq12 ./ tangent_norm)
phi_t2 = atan.(dq34 ./ tangent_norm, dq23 ./ tangent_norm)
tw = tangent_norm ./ max(maximum(tangent_norm), 1e-10)
phi_tang_raw = phi_t1 .* tw .+ phi_t2 .* (1 .- tw)
phi_tang = vcat(phi_tang_raw, phi_tang_raw[end])   # pad to n

# ── SOURCE 2: Klein pair phases ───────────────────────────────────────────
function safe_atan2(y, x)
    [xi == 0.0 && yi == 0.0 ? 0.0 : atan(yi, xi) for (xi,yi) in zip(x,y)]
end

p12n = max.(sqrt.(q12.^2 .+ q34.^2), 1e-10)
p13n = max.(sqrt.(q13.^2 .+ q24.^2), 1e-10)
p14n = max.(sqrt.(q14.^2 .+ q23.^2), 1e-10)

phi_k12 = safe_atan2(q34 ./ p12n, q12 ./ p12n)
phi_k13 = safe_atan2(q24 ./ p13n, q13 ./ p13n)
phi_k14 = safe_atan2(q23 ./ p14n, q14 ./ p14n)

tot = p12n .+ p13n .+ p14n
phi_klein = phi_k12 .* (p12n ./ tot) .+
            phi_k13 .* (p13n ./ tot) .+
            phi_k14 .* (p14n ./ tot)

# ── SOURCE 3: Score-based cumulative phase ────────────────────────────────
phi_score = zeros(n)
if isfile("chambers.tsv")
    try
        lines_ = readlines("chambers.tsv")
        header = split(lines_[1], "\t")
        score_col = findfirst(==("score"), header)
        if score_col !== nothing
            scores = Float64[]
            for line in lines_[2:end]
                parts = split(line, "\t")
                score_col <= length(parts) || continue
                val = tryparse(Float64, parts[score_col])
                val !== nothing && push!(scores, val)
            end
            if length(scores) >= 2
                if length(scores) < n
                    scores = vcat(scores, fill(scores[end], n - length(scores)))
                else
                    scores = scores[1:n]
                end
                sc = scores .- mean(scores)
                dscore = diff(sc)
                cp = cumsum(vcat(0.0, dscore ./ max.(abs.(sc[1:end-1]), 1e-10)))
                global phi_score; phi_score = mod.(cp .+ π, 2π) .- π
                println("Score phase loaded from chambers.tsv ($(length(scores)) points)")
            end
        end
    catch e
        println("Score phase fallback failed: $e")
    end
end

# ── Diagnostics ───────────────────────────────────────────────────────────
println("\nPhase source comparison:")
for (name, phi) in [("tangent", phi_tang), ("klein", phi_klein), ("score", phi_score)]
    u = length(unique(round.(phi, digits=3)))
    @printf("  %-8s  std=%.4f  unique=%-4d  range=[%+.3f, %+.3f]\n",
            name, std(phi), u, minimum(phi), maximum(phi))
end

# Select source with highest std
stds   = [std(phi_tang), std(phi_klein), std(phi_score)]
srcnam = ["tangent", "klein", "score"]
best   = argmax(stds)
println("\nSelected source: $(srcnam[best])  (std=$(round(stds[best],digits=4)))")
phase_primary = [phi_tang, phi_klein, phi_score][best]

# ── Smooth ────────────────────────────────────────────────────────────────
W = 5
phase_smooth = copy(phase_primary)
for i in (W÷2+1):(n-W÷2)
    phase_smooth[i] = mean(phase_primary[i-W÷2:i+W÷2])
end
for i in 1:n
    isnan(phase_smooth[i]) && (phase_smooth[i] = i>1 ? phase_smooth[i-1] : 0.0)
end

# ── Stable zone analysis ──────────────────────────────────────────────────
s0 = max(1, n÷10)
s1 = min(n, n - n÷10)
stable = phase_smooth[s0:s1]
sm = mean(stable); ss = std(stable)
uniq = length(unique(round.(stable, digits=3)))

println()
println("Stable zone (indices $(s0)–$(s1)):")
@printf("  Mean:             %.6f\n", sm)
@printf("  Std:              %.6f\n", ss)
@printf("  Unique:           %d\n", uniq)
@printf("  Prediction 1/2+1/120 = %.6f\n", 0.5+1/120)
@printf("  |mean - 1/2|    = %.6f\n", abs(sm - 0.5))
@printf("  |mean - pred|   = %.6f\n", abs(sm - (0.5+1/120)))

m = length(stable)
if m >= 3
    t1=mean(stable[1:m÷3]); t2=mean(stable[m÷3+1:2m÷3]); t3=mean(stable[2m÷3+1:end])
    @printf("  Three-thirds:     %.4f → %.4f → %.4f  %s\n",
            t1, t2, t3,
            abs(t3-0.5)<abs(t1-0.5) ? "(converging to 1/2)" : "(stable above 1/2)")
end

# ── Write ─────────────────────────────────────────────────────────────────
open("plucker_phase.json", "w") do f
    JSON3.write(f, Dict(
        "phase"         => phase_smooth,
        "phase_tangent" => phi_tang,
        "phase_klein"   => phi_klein,
        "phase_score"   => phi_score,
        "source_used"   => srcnam[best],
        "unique_count"  => length(unique(round.(phase_smooth, digits=3))),
        "mean_stable"   => sm,
        "std_stable"    => ss,
    ))
end

println()
println("✓ plucker_phase.json written")
@printf("  unique phases: %d (was 72)\n",
        length(unique(round.(phase_smooth, digits=3))))
println("="^60)
