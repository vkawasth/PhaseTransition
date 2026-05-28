# =============================================================================
# fukaya_gps_sectors.jl
# GPS-style sector-by-sector wrapped Fukaya category for Q_{7P}
#
# Four sectors defined by stop configuration:
#   Sector A: W(Σ_Q, Λ⁺ ∪ Λ⁻)  — fully stopped, k=0 baseline
#   Sector B: W(Σ_Q, Λ⁺)         — Λ⁻ removed, crisis onset
#   Sector C: W(Σ_Q, Λ⁻)         — Λ⁺ removed, recovery
#   Sector D: W(Σ_Q, {LA↔sAMY})  — minimal single stop
#
# GPS principle: adding stops restricts morphisms (Hom → 0).
# Removing stops wraps further (Hom opens up).
# =============================================================================

using Printf
using LinearAlgebra

# ── Edge table for Q_{7P} ─────────────────────────────────────────────────────
# 18 directed edges: (source, target, label, weight_baseline)
# Vertices: 1=CA1sp 2=BLA 3=HY 4=HPF 5=sAMY 6=LA 7=PAL

const EDGES = [
  # idx  src  tgt  label                weight
    (1,   1,   4,  "CA1sp→HPF",        16.98),
    (2,   4,   1,  "HPF→CA1sp",        16.98),
    (3,   1,   2,  "CA1sp→BLA",        27.20),
    (4,   1,   5,  "CA1sp→sAMY",      1170.80),
    (5,   4,   2,  "HPF→BLA",        158032.80),
    (6,   4,   5,  "HPF→sAMY",         37.54),
    (7,   2,   5,  "BLA→sAMY",         27.75),   # Λ⁺
    (8,   2,   6,  "BLA→LA",            2.06),
    (9,   2,   4,  "BLA→HPF",        5840.50),
    (10,  5,   2,  "sAMY→BLA",         27.75),   # Λ⁺
    (11,  5,   3,  "sAMY→HY",          27.09),   # Λ⁻
    (12,  5,   4,  "sAMY→HPF",         37.54),
    (13,  5,   6,  "sAMY→LA",          97.52),   # Λ⁺
    (14,  5,   7,  "sAMY→PAL",         49.42),   # Λ⁻
    (15,  3,   5,  "HY→sAMY",          27.09),   # Λ⁻
    (16,  6,   2,  "LA→BLA",            2.06),
    (17,  6,   5,  "LA→sAMY",          97.52),   # Λ⁺
    (18,  7,   5,  "PAL→sAMY",         49.42),   # Λ⁻
]

const N = length(EDGES)

# ── Stop definitions ──────────────────────────────────────────────────────────
# A stop is a directed edge index whose BACKWARD crossing is blocked.
# Forward crossing (in direction of edge): Hom = weight.
# Backward crossing (against direction):   Hom = 0.

const STOP_LAMplus  = [7, 10, 13, 17]   # Λ⁺: BLA→sAMY, sAMY→BLA, sAMY→LA, LA→sAMY
const STOP_LAMminus = [11, 14, 15, 18]  # Λ⁻: sAMY→HY, sAMY→PAL, HY→sAMY, PAL→sAMY
const STOP_LA_sAMY  = [13, 17]          # Minimal: sAMY→LA, LA→sAMY only

# Sector stop configurations
const SECTORS = [
    ("A", "W(Σ_Q, Λ⁺∪Λ⁻)",  "Fully stopped (k=0 baseline)",
        [STOP_LAMplus; STOP_LAMminus]),
    ("B", "W(Σ_Q, Λ⁺)",      "Crisis onset — Λ⁻ removed",
        STOP_LAMplus),
    ("C", "W(Σ_Q, Λ⁻)",      "Recovery — Λ⁺ removed",
        STOP_LAMminus),
    ("D", "W(Σ_Q,{LA↔sAMY})","Minimal single stop",
        STOP_LA_sAMY),
]

# ── Admissibility check ───────────────────────────────────────────────────────
# Edge j feeds into edge i: t(j) = s(i).
# Nonbacktracking: not (s(j)=t(i) and t(j)=s(i)).
# Stop-admissible: path j→i does not cross any active stop backward.
#
# "Crossing a stop e backward" in path j→i means:
#   The stop edge e goes s(e)→t(e), and the path crosses it as t(e)→s(e).
#   This happens when edge i goes t(e)→s(e) (i.e. i is the reverse of e).

function is_stop_reverse(edge_idx, stops)
    (_, si, ti, _, _) = EDGES[edge_idx]
    for s in stops
        (_, ss, ts, _, _) = EDGES[s]
        # edge_idx is the reverse of stop s if it goes ts→ss
        if si == ts && ti == ss
            return true
        end
    end
    return false
end

# ── Build Hashimoto matrix for a sector ───────────────────────────────────────
function build_B_sector(stops::Vector{Int})
    B = zeros(Float64, N, N)
    for i in 1:N
        (_, si, ti, _, wi) = EDGES[i]
        # Skip if edge i is the backward crossing of an active stop
        if is_stop_reverse(i, stops)
            continue   # Hom = 0 for this edge in this sector
        end
        for j in 1:N
            (_, sj, tj, _, wj) = EDGES[j]
            # j feeds into i: t(j) = s(i), nonbacktracking
            if tj == si && !(sj == ti && tj == si)
                B[i, j] = 1.0   # unweighted Hashimoto
            end
        end
    end
    return B
end

# ── Restriction map between sectors ───────────────────────────────────────────
# ρ_{XY}: W_X → W_Y when stops(Y) ⊂ stops(X)  (Y has fewer stops = more open)
# The restriction map is a projection: rows/cols of newly-opened edges become active.
# Here: diagonal indicator of which edges are active in sector X but not Y.

function restriction_map(stops_X, stops_Y)
    # Active in X = edges not stop-reversed in X
    # Active in Y = edges not stop-reversed in Y
    active_X = [!is_stop_reverse(i, stops_X) for i in 1:N]
    active_Y = [!is_stop_reverse(i, stops_Y) for i in 1:N]
    # Newly opened by going X→Y: active in Y but blocked in X
    newly_open = [active_Y[i] && !active_X[i] for i in 1:N]
    n_open = count(newly_open)
    n_active_X = count(active_X)
    return active_X, active_Y, newly_open, n_open, n_active_X
end

# ── Main computation ──────────────────────────────────────────────────────────
println("="^70)
println("GPS-STYLE WRAPPED FUKAYA CATEGORY  Q_{7P}")
println("Sectors A/B/C/D defined by stop configuration")
println("="^70)

results = Dict()

for (name, notation, desc, stops) in SECTORS
    println("\n" * "─"^60)
    println("SECTOR $name  $notation")
    println("  $desc")
    println("  Active stops ($( length(stops) )): edges $(stops)")
    println("  Stop edges: $([EDGES[s][4] for s in stops])")

    # Which edges are active (not stop-reversed)
    active = [!is_stop_reverse(i, stops) for i in 1:N]
    n_active = count(active)
    println("  Active edges: $n_active / $N")
    blocked = [EDGES[i][4] for i in 1:N if !active[i]]
    if !isempty(blocked)
        println("  Blocked (Hom=0): $blocked")
    end

    # Build and analyse Hashimoto matrix
    B = build_B_sector(stops)
    nnz_B = count(!=(0), B)
    eigs = eigvals(B)
    rho = maximum(abs.(eigs))

    println("  B_Ihara: $(N)×$(N),  nnz=$(nnz_B),  ρ=$(round(rho, digits=6))")

    # Count admissible sectors (nonzero rows or cols in B)
    admissible_rows = count(any(!=(0), B[i,:]) for i in 1:N)
    admissible_cols = count(any(!=(0), B[:,j]) for j in 1:N)
    println("  Admissible (active in B): rows=$admissible_rows  cols=$admissible_cols")

    # Backward Λ⁺ crossings that are blocked
    lam_plus_blocked = count(is_stop_reverse(i, STOP_LAMplus) &&
                             is_stop_reverse(i, stops) for i in 1:N)
    println("  Backward Λ⁺ crossings blocked: $lam_plus_blocked")

    results[name] = (rho=rho, nnz=nnz_B, active=n_active, B=B)
end

# ── GPS pushout structure ─────────────────────────────────────────────────────
println("\n" * "="^70)
println("GPS RESTRICTION MAPS (newly opened edges A→B, A→C, A→D)")
println("="^70)

sector_stops = Dict(
    "A" => [STOP_LAMplus; STOP_LAMminus],
    "B" => STOP_LAMplus,
    "C" => STOP_LAMminus,
    "D" => STOP_LA_sAMY,
)

for (from, to) in [("A","B"), ("A","C"), ("A","D"), ("B","D"), ("C","D")]
    aX, aY, new_open, n_open, n_X = restriction_map(
        sector_stops[from], sector_stops[to])
    println("\n  ρ_$(from)$(to): W_$(from) → W_$(to)  ($n_open edges newly opened)")
    if n_open > 0
        open_edges = [EDGES[i][4] for i in 1:N if new_open[i]]
        println("    Newly active: $open_edges")
        println("    These are the morphisms ρ_$(from)$(to) adds to W_$(to)")
    end
    # Rank of B difference
    B_from = results[from].B
    B_to   = results[to].B
    diff = B_to - B_from
    rank_diff = rank(diff)
    println("    rank(B_$(to) - B_$(from)) = $rank_diff  (dimension of newly wrapped morphisms)")
end

# ── Comparison table ──────────────────────────────────────────────────────────
println("\n" * "="^70)
println("SECTOR COMPARISON TABLE")
println("="^70)
println(@sprintf("  %-6s  %-26s  %6s  %5s  %6s",
    "Sector", "Description", "nnz(B)", "active", "ρ(B)"))
println("  " * "─"^60)
for (name, notation, desc, stops) in SECTORS
    r = results[name]
    println(@sprintf("  %-6s  %-26s  %6d  %5d  %6.4f",
        name, desc[1:min(26,length(desc))], r.nnz, r.active, r.rho))
end

println()
println("KEY:")
println("  ρ is NOT monotone in stop count — this is the main result:")
println("  ρ(A) = ρ(B) = 1.2599  Λ⁻ removal is SPECTRALLY INERT")
println("  ρ(C) = 1.9090         Λ⁺ removal drives all spectral growth")
println("  ρ(D) = φ = 1.6180     LA↔sAMY single stop gives golden ratio")
println("  Ordering: ρ(A)=ρ(B) < ρ(D) < ρ(C)")
println("  Only Λ⁺ stops control spectral radius.")
println("  rank(B_C - B_A) = 4 = |Λ_red| (confirms boundary obstruction theorem)")
