"""
run_fukaya.jl

Run the Fukaya category analysis using Q7P Phase 2 chambers.tsv.

Usage:
  julia run_fukaya.jl chambers.tsv    # Q7P Phase 2 (or any chambers.tsv)
  julia run_fukaya.jl                 # standalone Q7P defaults

chambers.tsv columns (from load_chambers.jl):
  1:snap  2:time  3:dominant  4:gap  5:obs(m6)  6:score
  7:support  8:perv  9:bridgeland_phase  10:klein_constraint
  11:lefschetz_proxy  12:gap_re  13:gap_im  14:plucker_phase

The edge weights w_{rs}(t) are NOT in chambers.tsv directly.
We derive per-snapshot weights from the Plücker coordinates:
  w_{rs}(t) ∝ |q_{ij}(t)| × baseline_w_{rs}
This gives the correct Phase 2 deformation of the algebra.
"""

using DelimitedFiles, LinearAlgebra, Printf, Statistics

include("load_chambers.jl")

# ── Load chambers.tsv ──────────────────────────────────────────────────────
chambers_file = length(ARGS) > 0 ? ARGS[1] : "chambers.tsv"

if isfile(chambers_file)
    d = load_chambers(chambers_file)
    println("Loaded: $chambers_file  ($(d.n) snapshots)")
    println(@sprintf("  m6 range:  [%.2e, %.2e]", minimum(abs.(d.obs)), maximum(abs.(d.obs))))
    println(@sprintf("  K range:   [%.4f, %.4f]", minimum(abs.(d.K)), maximum(abs.(d.K))))
    println(@sprintf("  gap range: [%.4f, %.4f]", minimum(d.gap), maximum(d.gap)))
    println(@sprintf("  Wall crossings detected: %d", sum(d.walls)))

    # Pick three representative snapshots:
    #   snap_baseline: Phase 1 / early stable (low obs, K≈0)
    #   snap_crisis:   peak m6 obstruction (argmax |obs|)
    #   snap_recovery: late stable (low obs after crisis)
    snap_baseline = argmin(abs.(d.obs))
    snap_crisis   = argmax(abs.(d.obs))
    snap_recovery = findlast(abs.(d.obs) .< quantile(abs.(d.obs), 0.1))
    snap_recovery = isnothing(snap_recovery) ? d.n : snap_recovery

    println(@sprintf("\n  Baseline snapshot:  %d  (|m6|=%.2e, K=%.4f)",
            snap_baseline, abs(d.obs[snap_baseline]), abs(d.K[snap_baseline])))
    println(@sprintf("  Crisis  snapshot:   %d  (|m6|=%.2e, K=%.4f)",
            snap_crisis,   abs(d.obs[snap_crisis]),   abs(d.K[snap_crisis])))
    println(@sprintf("  Recovery snapshot:  %d  (|m6|=%.2e, K=%.4f)",
            snap_recovery, abs(d.obs[snap_recovery]), abs(d.K[snap_recovery])))

    USE_SNAP = snap_crisis   # analyse the crisis snapshot
    println(@sprintf("\nAnalysing snapshot %d (CRISIS — maximum m6 obstruction)", USE_SNAP))
else
    println("No chambers file — using Q7P Phase 1 defaults")
    d = nothing
    USE_SNAP = 1
end

# ── Q7P graph structure ────────────────────────────────────────────────────
const REGIONS    = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :PAL]
const N_VERTICES = 7

const EDGES_BASE = [
    (:CA1sp,:HPF, 1),  (:CA1sp,:BLA, 2),  (:CA1sp,:sAMY,3),
    (:HPF,:CA1sp, 4),  (:HPF,:BLA,   5),  (:HPF,:sAMY,  6),
    (:BLA,:sAMY,  7),  (:BLA,:LA,    8),  (:BLA,:HPF,   9),
    (:sAMY,:BLA, 10),  (:sAMY,:HY,  11),  (:sAMY,:HPF, 12),
    (:sAMY,:LA,  13),  (:sAMY,:PAL, 14),
    (:HY,:sAMY,  15),
    (:LA,:BLA,   16),  (:LA,:sAMY,  17),
    (:PAL,:sAMY, 18),
]
const N_EDGES = 18

# Baseline Phase 1 weights (geometric impedance)
const W_BASELINE = Dict(
    (:CA1sp,:HPF)=>2850.4, (:CA1sp,:BLA)=>27.2,   (:CA1sp,:sAMY)=>1170.8,
    (:HPF,:CA1sp)=>3421.6, (:HPF,:BLA)=>5840.5,   (:HPF,:sAMY)=>345.9,
    (:BLA,:sAMY)=>27.75,   (:BLA,:LA)=>2.06,      (:BLA,:HPF)=>158032.8,
    (:sAMY,:BLA)=>27.75,   (:sAMY,:HY)=>27.09,    (:sAMY,:HPF)=>37.54,
    (:sAMY,:LA)=>97.52,    (:sAMY,:PAL)=>144.0,
    (:HY,:sAMY)=>27.09,
    (:LA,:BLA)=>2.06,      (:LA,:sAMY)=>97.52,
    (:PAL,:sAMY)=>144.0,
)

# ── Derive Phase 2 weights from Plücker coordinates ───────────────────────
# The Phase 2 deformation scales each edge weight by the Plücker mode:
#   w_{rs}(t) = w_{rs}(0) × φ_{rs}(t)
# where φ_{rs}(t) is derived from q_{ij}(t) at snapshot t.
#
# Mapping: BLA-cycle edges (8,10,13,16,17) scale with |q12| (BLA phase)
#          PAL-cycle edges (14,15,18)       scale with |q34| (PAL phase)
#          sAMY-hub edges  (7,11,12)        scale with Klein constraint |K|
#          Interior edges  (1-6,9)          scale with gap/gap_max
function phase2_weights(snap_idx::Int, d)
    isnothing(d) && return W_BASELINE

    gap_norm = d.gap[snap_idx] / (maximum(d.gap) + 1e-8)
    q12_norm = abs(d.q12[snap_idx]) / (maximum(abs.(d.q12)) + 1e-8)
    q34_norm = abs(d.q34[snap_idx]) / (maximum(abs.(d.q34)) + 1e-8)
    K_norm   = abs(d.K[snap_idx])   / (maximum(abs.(d.K))   + 1e-8)
    m6_norm  = abs(d.obs[snap_idx]) / (maximum(abs.(d.obs)) + 1e-8)

    # Edge-specific scaling
    bla_cycle = Set([8, 10, 13, 16, 17])
    pal_cycle = Set([14, 15, 18])
    samy_hub  = Set([7, 11, 12])

    W_out = Dict{Tuple{Symbol,Symbol}, Float64}()
    for (s, t, idx) in EDGES_BASE
        base = get(W_BASELINE, (s,t), 1.0)
        scale = if idx ∈ bla_cycle
            1.0 + m6_norm * q12_norm * 10.0   # BLA cycle amplified by m6
        elseif idx ∈ pal_cycle
            1.0 + K_norm * q34_norm * 5.0      # PAL cycle scales with K
        elseif idx ∈ samy_hub
            1.0 + K_norm * 20.0                # sAMY hub amplified by K
        else
            max(0.1, gap_norm)                  # interior: scales with gap
        end
        W_out[(s,t)] = base * scale
    end
    return W_out
end

const EDGES        = EDGES_BASE
const W            = phase2_weights(USE_SNAP, isfile(chambers_file) ? d : nothing)
const LAMBDA_PLUS  = Set([(:BLA,:sAMY),(:sAMY,:BLA),(:LA,:sAMY),(:sAMY,:LA)])
const LAMBDA_MINUS = Set([(:sAMY,:HY),(:HY,:sAMY),(:PAL,:sAMY),(:sAMY,:PAL)])

println("\nPhase 2 weights at crisis snapshot:")
println(@sprintf("  Max weight amplification: %.2f×",
        maximum(W[k]/W_BASELINE[k] for k in keys(W_BASELINE))))
println(@sprintf("  sAMY→BLA weight: %.2f (baseline: %.2f)",
        get(W,(:sAMY,:BLA),0.0), get(W_BASELINE,(:sAMY,:BLA),0.0)))
println(@sprintf("  BLA→sAMY weight: %.2f (baseline: %.2f)",
        get(W,(:BLA,:sAMY),0.0), get(W_BASELINE,(:BLA,:sAMY),0.0)))

println("\n" * "="^70)
println("Running Fukaya category analysis on Q7P Phase 2 (crisis snapshot)")
println("="^70 * "\n")

# Run main analysis
include("fukaya_category.jl")

# ── Phase 2 specific: track sector across all snapshots ───────────────────
if !isnothing(d)
    println("\n\n── PHASE 2: QUANTUM SECTOR TRAJECTORY ────────────────────────────────")
    println("  k(t) = quantum sector at each snapshot")
    println("  Derived from Schubert stratum (K constraint)")
    println()

    # Sector k detection using log-midpoint threshold on m6
    # Works for both K=0 (sphere limit) and K≠0 runs
    m6_abs  = abs.(d.obs)
    m6_log  = log10.(m6_abs .+ 1.0)
    m6_max_log = maximum(m6_log)
    m6_med_log = median(m6_log)

    if m6_max_log - m6_med_log > 5
        # Bimodal distribution: log-midpoint separates crisis from baseline
        m6_thresh = 10 ^ ((m6_med_log + m6_max_log) / 2)
        println(@sprintf("  m6 threshold (log-midpoint): %.2e", m6_thresh))
    else
        # Unimodal: use 90th percentile
        m6_thresh = 10 ^ quantile(m6_log, 0.90)
        println(@sprintf("  m6 threshold (90th pct): %.2e", m6_thresh))
    end

    k_traj = [m6_abs[i] > m6_thresh ? 3 : 0 for i in 1:d.n]

    k0_frac = count(k_traj .== 0) / d.n
    k3_frac = count(k_traj .== 3) / d.n
    println(@sprintf("  k=0 (baseline):        %.1f%% of snapshots", 100*k0_frac))
    println(@sprintf("  k=3 (crisis, top 10%%): %.1f%% of snapshots", 100*k3_frac))

    # Find crisis onset and recovery
    crisis_onset = findfirst(k_traj .== 3)
    recovery     = isnothing(crisis_onset) ? nothing :
                   findnext(==(0), k_traj, crisis_onset + 1)

    if !isnothing(crisis_onset)
        println(@sprintf("\n  Crisis onset:    snapshot %d  (|m6|=%.2e)",
                crisis_onset, abs(d.obs[crisis_onset])))
        if !isnothing(recovery)
            println(@sprintf("  Recovery:        snapshot %d", recovery))
            println(@sprintf("  Crisis duration: %d snapshots", recovery - crisis_onset))
        end
    end

    # Net winding from wall crossings
    wall_snaps = findall(d.walls .== 1)
    println(@sprintf("\n  Wall crossings: %d total", length(wall_snaps)))
    # Use Bridgeland phase sign to distinguish forward vs backward crossings
    # Positive phase change = forward Λ⁺ crossing (opioid, n₊)
    # Negative phase change = backward Λ⁻ crossing (norcain, n₋)
    if length(wall_snaps) > 0
        phase_changes = [wall_snaps[i] > 1 ?
            d.bphase[wall_snaps[i]] - d.bphase[wall_snaps[i]-1] : 0.0
            for i in 1:length(wall_snaps)]
        n_plus  = count(phase_changes .>= 0)
        n_minus = count(phase_changes .< 0)
        w_net   = n_plus - n_minus
        println(@sprintf("  n₊ (forward,  Δφ≥0): %d", n_plus))
        println(@sprintf("  n₋ (backward, Δφ<0): %d", n_minus))
        println(@sprintf("  Net winding w = n₊-n₋ = %d", w_net))
        println(@sprintf("  (Paper values: n₊=245, n₋=239, w=+6 for full Q7P)"))
    end
end
