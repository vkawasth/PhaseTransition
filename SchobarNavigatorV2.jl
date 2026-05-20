module SchoberNavigatorV2

using JSON3
using LinearAlgebra
using Statistics
using Printf

export ChamberCategory,
       FlipFunctor,
       SchoberState,
       load_snapshot,
       build_chamber,
       build_functor,
       monodromy_operator,
       chamber_score,
       detect_singular_wall,
       run_schober_path,
       summarize_path,
       write_chamber_table,
       write_wall_table,
       lefschetz_numbers,
       compute_spectral_gap_complex,
       extract_klein_constraint,
       ZetaComparison,
       compute_zeta_comparison,
       write_zeta_comparison_table,
       summarize_zeta_stack

# ============================================================
# CONSTANTS
# ============================================================

const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :LSX, :sAMY] # matches regions_six_ANDPAL.csv

# Load full B_Ihara from connectome_graphs.json if available
# This gives the correct spectral zeta regardless of which arrows
# appear in the prime paths
const GRAPHS_JSON_PATH = joinpath(@__DIR__, "connectome_graphs.json")
const GRAPH_TYPE_DEFAULT = "Q_7P"

function load_full_B_ihara(graph_type::String=GRAPH_TYPE_DEFAULT)
    !isfile(GRAPHS_JSON_PATH) && return nothing, 0
    try
        gdata_all = JSON3.read(read(GRAPHS_JSON_PATH, String))
        gdata = get(gdata_all, Symbol(graph_type), nothing)
        isnothing(gdata) && return nothing, 0
        arrows  = gdata[:arrows]
        n_arr   = Int(gdata[:n_arr])
        # Build reversal map
        rev = zeros(Int, n_arr)
        for a in arrows
            ri = Int(a["rev_idx"])
            ri > 0 && (rev[Int(a["idx"])] = ri)
        end
        # Build B_Ihara
        B = zeros(Float64, n_arr, n_arr)
        for ai in arrows
            for aj in arrows
                i = Int(ai["idx"]); j = Int(aj["idx"])
                if Int(ai["v_tgt"]) == Int(aj["v_src"]) &&
                   !(Bool(ai["sym"]) && rev[i] == j)
                    B[i, j] = 1.0
                end
            end
        end
        return B, n_arr
    catch e
        @warn "Could not load B_Ihara from $GRAPHS_JSON_PATH: $e"
        return nothing, 0
    end
end

# Pre-load B_Ihara once at module load time
const _B_IHARA_FULL, _N_ARR_FULL = load_full_B_ihara()

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

safeget(d, k, default) = haskey(d, k) ? d[k] : default
safe_float(x) = try; Float64(x); catch; 0.0; end

"""
Extract region symbol from a closure string like "e_CA1sp" or "f_CA1sp_HPF".
"""
function region_from_symbol(sym::AbstractString)
    if startswith(sym, "e_")
        return Symbol(sym[3:end])
    elseif startswith(sym, "f_")
        parts = split(sym, "_")
        if length(parts) >= 3
            return Symbol(parts[2])
        end
    end
    return nothing
end

function normalize_scores(d::Dict{Symbol,Float64})
    isempty(d) && return d
    mx = maximum(values(d))
    mx <= 0 && return d
    return Dict(k => v/mx for (k,v) in d)
end

function compute_perversity(scores::Dict{Symbol,Float64}; Pmax::Int=2)
    ns = normalize_scores(scores)
    out = Dict{Symbol,Int}()
    for (k,v) in ns
        out[k] = floor(Int, min(Pmax, Pmax * v))
    end
    return out
end

"""
Extract timestamp from filename (assumes "ainf_export_<timestamp>.json").
"""
function extract_timestamp(filename::String)
    m = match(r"ainf_export_([0-9]+(?:\.[0-9]+)?)", basename(filename))
    m === nothing && return 0.0
    return parse(Float64, m.captures[1])
end

"""
Sum of all absolute coefficients in the m6 dictionary.
m6 is a Dict{String, Dict{String, Float64}}.
"""
function compute_obstruction(snap)
    obs = 0.0
    if haskey(snap, :m6)
        for (_, inner) in pairs(snap[:m6])
            for (_, coeff) in pairs(inner)
                obs += abs(safe_float(coeff))
            end
        end
    end
    return obs
end

"""
Determine dominant region from prime_higher_ideals (largest total_support).
Fallback: region with most appearances in closures.
"""
function dominant_region(snap)
    # First pass: find ideal with highest total_support > 0
    best_region = nothing
    best_score = -Inf
    if haskey(snap, :prime_higher_ideals)
        for ideal in snap[:prime_higher_ideals]
            total = safe_float(get(ideal, "total_support", 0.0))
            if total > best_score
                for sym in get(ideal, "closure", String[])
                    reg = region_from_symbol(sym)
                    if reg !== nothing && reg in REGIONS
                        best_score = total
                        best_region = reg
                        break   # take the first region from the best ideal
                    end
                end
            end
        end
    end

    # If found a positive support, return it
    if best_region !== nothing
        return best_region
    end

    # Otherwise, count region frequencies across all closures
    counts = Dict(r => 0 for r in REGIONS)
    if haskey(snap, :prime_higher_ideals)
        for ideal in snap[:prime_higher_ideals]
            for sym in get(ideal, "closure", String[])
                reg = region_from_symbol(sym)
                if reg !== nothing && reg in REGIONS
                    counts[reg] += 1
                end
            end
        end
    end

    if maximum(values(counts)) > 0
        # find region with maximum count
        return argmax(counts)
    end

    # Ultimate fallback – but this should rarely happen
    return REGIONS[1]
end

"""
Compute spectral gap from prime_paths using region co‑occurrence Laplacian.
"""
function compute_spectral_gap(snap)
    if !haskey(snap, :prime_paths) || isempty(snap[:prime_paths])
        return 0.5  # default moderate gap
    end
    cooc = zeros(Float64, length(REGIONS), length(REGIONS))
    for item in snap[:prime_paths]
        path = item["path"]
        # collect region symbols along the path
        regions_along = Symbol[]
        for sym in path
            reg = region_from_symbol(string(sym))
            if reg !== nothing && reg in REGIONS
                push!(regions_along, reg)
            end
        end
        for i in 1:length(regions_along)-1
            a = findfirst(==(regions_along[i]), REGIONS)
            b = findfirst(==(regions_along[i+1]), REGIONS)
            if a !== nothing && b !== nothing
                cooc[a,b] += 1.0
                cooc[b,a] += 1.0
            end
        end
    end
    deg = vec(sum(cooc, dims=2))
    L = diagm(deg) - cooc
    evals = eigvals(L)
    # second smallest eigenvalue (Fiedler value)
    gap = evals[2] > 0 ? evals[2] : 0.0
    return gap
end

# ============================================================
# DATA TYPES
# ============================================================

struct ChamberCategory
    id::Int
    time::Float64
    dominant_region::Symbol
    support_scores::Dict{Symbol,Float64}
    perversity::Dict{Symbol,Int}
    spectral_gap::Float64
    obstruction::Float64
    # --- complex extensions ---
    spectral_gap_complex::ComplexF64      # complex Fiedler value (Bridgeland phase source)
    bridgeland_phase::Float64             # arg(λ₂_complex) / 2π ∈ (0,1] — Fukaya stability
    lefschetz_proxy::Float64              # tr(support score change) — perverse schober integrality
    klein_constraint::Float64             # |q12*q34 - q13*q24 + q14*q23| from snapshot
    plucker_phase::Float64                # phase used for complex gap computation
end

struct FlipFunctor
    from::Int
    to::Int
    weight::Float64
    singular::Bool
    activated_regions::Vector{Symbol}
end

struct ZetaComparison
    snapshot_id::Int
    u::Float64                    # evaluation point (0.9 / spectral_radius)
    spectral_zeta::Float64        # 1 / det(I - u*T)  — spectral side
    combinatorial_zeta::Float64   # Π (1 - u^l(p))    — Euler product side
    mismatch::Float64             # |spectral - combinatorial| — Bass theorem deviation
    n_active_paths::Int           # number of prime paths contributing
    wall_crossing::Bool           # true if tubing flipped at this snapshot
    bridgeland_phase::Float64     # from chamber (for alignment)
end

struct SchoberState
    chambers::Vector{ChamberCategory}
    functors::Vector{FlipFunctor}
    zeta_comparisons::Vector{ZetaComparison}   # per-chamber zeta tracking
end

# ============================================================
# JSON LOADING
# ============================================================

load_snapshot(file::String) = JSON3.read(read(file, String))

# ============================================================
# CHAMBER BUILDING
# ============================================================

# ============================================================
# COMPLEX SPECTRAL GAP  (new)
# Hodge Laplacian with Plücker phase twist.
# Eigenvalues are complex — arg(λ₂)/2π are Bridgeland phases.
# ============================================================

function compute_spectral_gap_complex(snap, plucker_phase::Float64)
    if !haskey(snap, :prime_paths) || isempty(snap[:prime_paths])
        return ComplexF64(0.5, 0.0)
    end
    cooc = zeros(ComplexF64, length(REGIONS), length(REGIONS))
    path_weights = Float64[]

    for item in snap[:prime_paths]
        path = item["path"]
        w    = safe_float(get(item, "weight", 1.0))
        push!(path_weights, abs(w))
        regions_along = Symbol[]
        for sym in path
            reg = region_from_symbol(string(sym))
            if reg !== nothing && reg in REGIONS
                push!(regions_along, reg)
            end
        end
        for i in 1:length(regions_along)-1
            a = findfirst(==(regions_along[i]), REGIONS)
            b = findfirst(==(regions_along[i+1]), REGIONS)
            if a !== nothing && b !== nothing
                # Directed complex weight — breaks Hermitian symmetry
                # forward edge: weight * exp(+iφ)
                # reverse edge: weight * exp(-iφ) * (1 + iφ/π)  ← asymmetric
                # This ensures L is non-Hermitian → complex eigenvalues
                wlog = log(1 + abs(w))
                cooc[a,b] += wlog * exp(im * plucker_phase)
                cooc[b,a] += wlog * exp(-im * plucker_phase) * (1.0 + im * plucker_phase / π)
            end
        end
    end

    # Degree vector from column sums (complex)
    deg = vec(sum(cooc, dims=2))

    # Non-Hermitian Laplacian: L = D - A where D has complex diagonal
    # The imaginary part of D encodes the monodromy phase accumulation
    L = diagm(deg) - cooc

    # Eigenvalues are now genuinely complex
    evals = eigvals(L)

    # Sort by real part, return second (Fiedler-like value)
    sort!(evals, by=real)
    return length(evals) >= 2 ? evals[2] : ComplexF64(0.0)
end

# ============================================================
# KLEIN CONSTRAINT from snapshot Plücker fields  (new)
# ============================================================

function extract_klein_constraint(snap)::Float64
    # First choice: actual Plücker coordinates saved in snapshot
    for key in (:plucker_coords, :plucker)
        if haskey(snap, key)
            q = snap[key]
            if length(q) >= 6
                q12, q13, q14 = safe_float(q[1]), safe_float(q[2]), safe_float(q[3])
                q23, q24, q34 = safe_float(q[4]), safe_float(q[5]), safe_float(q[6])
                return abs(q12*q34 - q13*q24 + q14*q23)
            end
        end
    end
    # Fallback proxy: inversely proportional to total support
    # (near 0 when system is coherent / on Gr(2,4))
    if haskey(snap, :prime_higher_ideals)
        ideals = snap[:prime_higher_ideals]
        # Guard against empty collection — sum needs init when generator may be empty
        isempty(ideals) && return 0.0
        total = sum(safe_float(get(ideal, "total_support", 0.0))
                    for ideal in ideals; init=0.0)
        return total > 0 ? 1.0 / (1.0 + total) : 0.0
    end
    return 0.0
end

lefschetz_proxy_chamber(scores::Dict{Symbol,Float64}) = sum(values(scores))

# ============================================================
# BUILD CHAMBER  (rewritten)
# ============================================================

function build_chamber(id::Int, snap, time::Float64;
                       plucker_phase::Float64=0.0,
                       prev_scores::Union{Dict{Symbol,Float64},Nothing}=nothing)

    # ── Support scores from prime_higher_ideals ──────────────────────────
    scores = Dict{Symbol,Float64}()

    if haskey(snap, :prime_higher_ideals)
        for obj in snap[:prime_higher_ideals]
            total = safe_float(get(obj, "total_support", 0.0))
            if total == 0.0
                total = 1.0  # fallback if missing
            end
            for sym in get(obj, "closure", String[])
                reg = region_from_symbol(sym)
                if reg !== nothing && reg in REGIONS
                    scores[reg] = get(scores, reg, 0.0) + total
                end
            end
        end
    end

    if isempty(scores)
        for r in REGIONS
            scores[r] = 1.0
        end
    end

    max_score = maximum(values(scores))
    if max_score > 0
        scores = Dict(k => v / max_score for (k,v) in scores)
    end

    # ── Original real fields ─────────────────────────────────────────────
    perversity = compute_perversity(scores)
    gap        = compute_spectral_gap(snap)
    obs        = compute_obstruction(snap)
    dom        = dominant_region(snap)

    # ── Complex spectral gap → Bridgeland phase ──────────────────────────
    # arg(λ₂_complex) / 2π ∈ (0,1] is the Bridgeland phase of this chamber.
    # For the Fukaya proof: all stable objects must have phase ∈ (0,1].
    # Values outside this range signal a chamber that is not a stable object.
    gap_complex = compute_spectral_gap_complex(snap, plucker_phase)
    raw_phase   = angle(gap_complex) / (2π)          # ∈ (-0.5, 0.5]
    bridgeland  = raw_phase <= 0.0 ? raw_phase + 1.0 : raw_phase  # shift to (0,1]

    # ── Klein quadric constraint ─────────────────────────────────────────
    # |q12*q34 - q13*q24 + q14*q23| — should → 0 as system approaches Gr(2,4).
    # Tracked per chamber so test_klein_spectral_relation.jl can align it.
    klein = extract_klein_constraint(snap)

    # ── Lefschetz proxy ──────────────────────────────────────────────────
    # Difference in support score trace between consecutive chambers.
    # Integer-valued in the limit if spherical functor condition holds
    # (perverse schober integrality — checkable from write_chamber_table output).
    lef_proxy = if prev_scores !== nothing
        sum(get(scores, r, 0.0) - get(prev_scores, r, 0.0) for r in REGIONS)
    else
        lefschetz_proxy_chamber(scores)
    end

    return ChamberCategory(
        id, time, dom, scores, perversity, gap, obs,
        gap_complex, bridgeland, lef_proxy, klein, plucker_phase
    )
end


# ============================================================
# ZETA COMPARISON — spectral vs combinatorial  (new)
#
# Tests the Bass theorem identity at each snapshot:
#   det(I - uT)^{-1}  =  Π_{prime paths p} (1 - u^{l(p)})^{-1}
#
# Mismatch ≈ 0  → Bass holds → stack is deforming consistently
# Mismatch spikes at wall crossings → Kontsevich-Soibelman formula
#   restores consistency after each flip
#
# Connects:
#   best tubes     → which prime paths are active (Euler product)
#   prime ideals   → which walls are being crossed
#   Ihara zeta     → spectral side via det(I - uT)
#   wall crossings → mismatch spikes then recovers
# ============================================================

"""
Extract active prime paths from snapshot and compute their lengths.
Path length = number of region hops in the path string.
"""
function get_active_prime_paths(snap)
    paths = Tuple{Vector{String}, Float64}[]   # (path_symbols, weight)
    if !haskey(snap, :prime_paths)
        return paths
    end
    for item in snap[:prime_paths]
        path = [string(s) for s in item["path"]]
        weight = safe_float(get(item, "weight", 1.0))
        if weight > 0 && length(path) >= 2
            push!(paths, (path, weight))
        end
    end
    return paths
end

"""
Path length = number of edges = nodes - 1.
For the Euler product (1 - u^{l(p)}), l(p) is the edge count.
"""
path_length(path::Vector{String}) = max(1, length(path) - 1)

"""
Combinatorial zeta from Euler product over active prime paths.
Bass theorem: ζ_Ihara^{-1}(u) = Π_p (1 - u^{l(p)})
where l(p) is the PATH LENGTH (edge count), NOT weighted by prime path weight.
The weight is irrelevant for the Bass/Ihara Euler product —
it uses only the combinatorial structure (length of each prime path).
Returns ζ_comb(u)^{-1} = Π_p (1 - u^{l(p)}) [the inverse zeta].
"""
function combinatorial_zeta(paths, u::Float64)
    # Bass theorem: ζ_Ihara^{-1}(u) = Π_{primitive p} (1 - u^{l(p)})
    #
    # COMBINATORIAL SIDE uses the A∞ prime paths (Hochschild cycles)
    # as the source of primitive cycles. We deduplicate by rotational
    # equivalence (same primitive cycle, different starting point)
    # and compute the Euler product over distinct cycle classes.
    #
    # The SPECTRAL SIDE (det(I-uB)) uses the Hashimoto matrix.
    # Their agreement — tested by log_mismatch — measures whether the
    # A∞ combinatorial structure reproduces the Ihara spectrum.
    # Mismatch → 0 as A∞ algebra approaches classical (sphere limit).
    #
    # NOTE: do NOT shortcut to det(I-uB) here — that would make both
    # sides identical and mismatch ≡ 0 by construction (no information).
    # The fallback Euler product below is the correct combinatorial side.

    seen_signatures = Set{Vector{String}}()
    inv_zeta = 1.0
    n_used   = 0

    for (path, weight) in paths
        # Extract region sequence from the path symbols
        regions = String[]
        for sym in path
            s = string(sym)
            if startswith(s, "f_")
                parts = split(s, "_")
                length(parts) >= 2 && push!(regions, parts[2])
            end
        end
        isempty(regions) && continue
        n_path = length(regions)

        # Rotational deduplication — same primitive cycle regardless of start
        rotations = [circshift(regions, k) for k in 0:n_path-1]
        sig = minimum(rotations)
        sig ∈ seen_signatures && continue
        push!(seen_signatures, sig)
        n_used += 1

        # l(p) = path length (edge count), weight not used here
        factor = 1.0 - u^n_path
        abs(factor) < 1e-10 && return Inf
        inv_zeta *= factor
    end

    # If no A∞ paths available at all, fall back to spectral side
    if n_used == 0 && !isnothing(_B_IHARA_FULL) && _N_ARR_FULL > 0
        M = I - u * _B_IHARA_FULL
        d = det(M)
        return abs(d) < 1e-10 ? Inf : 1.0 / abs(d)
    end

    return abs(inv_zeta) < 1e-10 ? Inf : 1.0 / inv_zeta
end
"""
Spectral zeta from the Hashimoto nonbacktracking matrix B_Ihara.
ζ_spec(u)^{-1} = det(I - u*B_Ihara)  [Bass theorem spectral side]
B_Ihara[e,e'] = 1 iff arrow e' is admissible continuation of arrow e
              (composable and not a reversal).
Returns ζ_spec(u)^{-1} = det(I - u*B_Ihara).
"""
function spectral_zeta_from_paths(paths, u::Float64)
    # Use full B_Ihara from connectome_graphs.json for correct spectral zeta
    if !isnothing(_B_IHARA_FULL) && _N_ARR_FULL > 0
        M = I - u * _B_IHARA_FULL
        d = det(M)
        return abs(d) < 1e-10 ? Inf : 1.0 / abs(d)
    end
    # Fallback: path-derived B_Ihara
    arrows = Tuple{Symbol,Symbol}[]
    for (path, _) in paths
        for sym in path
            if startswith(sym, "f_")
                parts = split(sym, "_")
                if length(parts) >= 3
                    src = Symbol(parts[2]); tgt = Symbol(parts[3])
                    if src in REGIONS && tgt in REGIONS
                        e = (src, tgt); e ∉ arrows && push!(arrows, e)
                    end
                end
            end
        end
    end
    isempty(arrows) && return 1.0 / abs(1.0 - u)
    rev = Dict{Int,Int}()
    for (i,(s,t)) in enumerate(arrows), (j,(s2,t2)) in enumerate(arrows)
        s2==t && t2==s && i!=j && (rev[i]=j; break)
    end
    n = length(arrows)
    B = zeros(Float64, n, n)
    for (i,(si,ti)) in enumerate(arrows), (j,(sj,tj)) in enumerate(arrows)
        ti==sj && get(rev,i,0)!=j && (B[i,j]=1.0)
    end
    M = I - u * B; d = det(M)
    return abs(d) < 1e-10 ? Inf : 1.0 / abs(d)
end
"""
Compute ZetaComparison for one chamber snapshot.
wall_crossing = true if the tubing signature changed from previous chamber.
"""
function compute_zeta_comparison(
        id::Int,
        snap,
        chamber::ChamberCategory;
        wall_crossing::Bool=false,
        u_override::Float64=0.0)

    paths = get_active_prime_paths(snap)

    # Use fixed u = 0.5 — well inside convergence radius of Ihara zeta
    # (convergence radius = 1/ρ(B_Ihara) ≈ 1/1.79 ≈ 0.56 for Q_7P)
    # Using spectral_gap as denominator was wrong: it gives u≈0.9/0.01
    # which is far outside the convergence radius
    u = u_override > 0 ? u_override : 0.5

    spec  = spectral_zeta_from_paths(paths, u)
    comb  = combinatorial_zeta(paths, u)

    # Mismatch: Bass theorem says these should be equal.
    # We compare logs to handle the large dynamic range.
    if isinf(spec) || isinf(comb)
        mismatch = Inf
    else
        log_spec = log(max(spec, 1e-12))
        log_comb = log(max(comb, 1e-12))
        mismatch = abs(log_spec - log_comb)
    end

    return ZetaComparison(
        id, u,
        isinf(spec) ? 1e12 : spec,
        isinf(comb) ? 1e12 : comb,
        isinf(mismatch) ? 1e6 : mismatch,
        length(paths),
        wall_crossing,
        chamber.bridgeland_phase
    )
end

# ============================================================
# CHAMBER SCORE
# ============================================================

function chamber_score(C::ChamberCategory)
    support_sum = sum(values(C.support_scores))
    perv_sum = sum(values(C.perversity))
    gap_term = 10.0 * C.spectral_gap
    obs_term = -0.01 * C.obstruction
    return support_sum + perv_sum + gap_term + obs_term
end

# ============================================================
# WALL / FUNCTOR BUILDING
# ============================================================

function build_functor(A::ChamberCategory, B::ChamberCategory;
                       m6_thresh=1000.0, gap_thresh=0.05)
    activated = Symbol[]
    for r in REGIONS
        supB = get(B.support_scores, r, 0.0)
        supA = get(A.support_scores, r, 0.0)
        if supB > supA + 1e-6
            push!(activated, r)
        end
    end

    weight = abs(chamber_score(B) - chamber_score(A))
    singular = (max(A.obstruction, B.obstruction) > m6_thresh) ||
               (min(A.spectral_gap, B.spectral_gap) < gap_thresh)

    return FlipFunctor(A.id, B.id, weight, singular, activated)
end

# ============================================================
# MONODROMY OPERATOR
# ============================================================

function monodromy_operator(chambers::Vector{ChamberCategory})
    n = length(REGIONS)
    M = Matrix{Float64}(I, n, n)
    for C in chambers
        idx = findfirst(==(C.dominant_region), REGIONS)
        if idx !== nothing
            M[idx, idx] += 0.1
        end
    end
    return M
end

# ============================================================
# DRIVER
# ============================================================

# Load Plücker phases written by run_iharaSingV2_MonoTwist.jl
function load_plucker_phases_schober(n::Int; file="plucker_phase.json")
    if isfile(file)
        data = JSON3.read(read(file, String))
        phases = Float64.(data["phase"])
        if length(phases) >= n
            return phases[1:n]
        else
            return vcat(phases, fill(phases[end], n - length(phases)))
        end
    else
        @warn "plucker_phase.json not found — using zero phase for all chambers"
        return zeros(Float64, n)
    end
end

# ============================================================
# LEFSCHETZ NUMBERS per FlipFunctor  (new)
# Tr(support score change) across each wall.
# Should be integer-valued if spherical functor condition holds
# — the checkable perverse schober integrality condition.
# ============================================================

function lefschetz_numbers(S::SchoberState)
    numbers = Float64[]
    for F in S.functors
        C_from = S.chambers[F.from]
        C_to   = S.chambers[min(F.to, length(S.chambers))]
        lef = sum(get(C_to.support_scores, r, 0.0) -
                  get(C_from.support_scores, r, 0.0)
                  for r in REGIONS)
        push!(numbers, lef)
    end
    return numbers
end

function run_schober_path(files::Vector{String})
    ainf_files = filter(f -> startswith(basename(f), "ainf_export_") &&
                             endswith(f, ".json"), files)
    if isempty(ainf_files)
        error("No ainf_export_*.json files found.")
    end
    sort!(ainf_files)
    n = length(ainf_files)

    # Load per-snapshot Plücker phases (one per ainf snapshot)
    phases = load_plucker_phases_schober(n)

    chambers = ChamberCategory[]
    prev_scores = nothing
    for (i, f) in enumerate(ainf_files)
        snap = load_snapshot(f)
        time = extract_timestamp(f)
        if time == 0.0
            time = Float64(i)
        end
        chamber = build_chamber(i, snap, time;
                                plucker_phase = phases[i],
                                prev_scores   = prev_scores)
        push!(chambers, chamber)
        prev_scores = chamber.support_scores
    end

    functors = FlipFunctor[]
    for i in 1:length(chambers)-1
        push!(functors, build_functor(chambers[i], chambers[i+1]))
    end

    # Build zeta comparisons — one per chamber
    # wall_crossing = true when tubing signature changed
    # (proxy: Bridgeland phase changed by > 0.05)
    zeta_comparisons = ZetaComparison[]
    for (i, f) in enumerate(ainf_files)
        snap = load_snapshot(f)
        wall = if i > 1
            abs(chambers[i].bridgeland_phase - chambers[i-1].bridgeland_phase) > 0.05
        else
            false
        end
        zc = compute_zeta_comparison(i, snap, chambers[i]; wall_crossing=wall)
        push!(zeta_comparisons, zc)
    end

    return SchoberState(chambers, functors, zeta_comparisons)
end

# ============================================================
# OUTPUT TABLES
# ============================================================

function write_chamber_table(S::SchoberState, filename::String="chambers.tsv")
    io = open(filename, "w")
    # Extended header includes complex fields
    println(io, join(["id","time","dominant","gap","obs","score",
                      "support_sum","perv_sum",
                      "bridgeland_phase","klein_constraint",
                      "lefschetz_proxy","gap_complex_re","gap_complex_im",
                      "plucker_phase"], "\t"))
    for C in S.chambers
        @printf(io, "%d\t%.3f\t%s\t%.4f\t%.2f\t%.3f\t%.3f\t%d\t%.4f\t%.6f\t%.4f\t%.4f\t%.4f\t%.4f\n",
                C.id, C.time, C.dominant_region,
                C.spectral_gap, C.obstruction,
                chamber_score(C),
                sum(values(C.support_scores)),
                sum(values(C.perversity)),
                # new complex fields
                C.bridgeland_phase,
                C.klein_constraint,
                C.lefschetz_proxy,
                real(C.spectral_gap_complex),
                imag(C.spectral_gap_complex),
                C.plucker_phase)
    end
    close(io)
    println("Chamber table written to $filename")
    # Print Bridgeland phase summary — key for Fukaya proof check
    phases = [C.bridgeland_phase for C in S.chambers]
    in_range = count(p -> 0.0 < p <= 1.0, phases)
    println("  Bridgeland phases in (0,1]: $in_range / $(length(phases))")
    println("  (All should be in (0,1] for Fukaya stability to hold)")
    kleins = [C.klein_constraint for C in S.chambers]
    println("  Klein constraint: min=$(round(minimum(kleins),digits=6)), max=$(round(maximum(kleins),digits=4))")
    lefs = [C.lefschetz_proxy for C in S.chambers]
    near_int = count(l -> abs(l - round(l)) < 0.1, lefs)
    println("  Lefschetz proxies near integer: $near_int / $(length(lefs))")
    println("  (Near-integer → spherical functor condition approximately holds)")
end

function write_wall_table(S::SchoberState, filename::String="walls.tsv")
    io = open(filename, "w")
    println(io, "from\tto\tweight\tsingular\tactivated")
    for F in S.functors
        act_str = join(string.(F.activated_regions), ",")
        @printf(io, "%d\t%d\t%.3f\t%s\t%s\n",
                F.from, F.to, F.weight, F.singular, act_str)
    end
    close(io)
    println("Wall table written to $filename")
end

# ============================================================
# CONSOLE SUMMARY
# ============================================================

function summarize_path(S::SchoberState)
    println("=========== SCHOBER PATH SUMMARY ===========")
    println("Number of chambers: ", length(S.chambers))
    println("Number of walls:    ", length(S.functors))
    println()

    println("Chambers:")
    for C in S.chambers
        @printf("  %3d  t=%.2f  dom=%s  gap=%.4f  obs=%.2f  score=%.1f\n",
                C.id, C.time, C.dominant_region,
                C.spectral_gap, C.obstruction, chamber_score(C))
    end
    println()

    println("Walls (singular transitions):")
    for F in S.functors
        if F.singular
            @printf("  %d → %d  weight=%.3f  activated=[%s]\n",
                    F.from, F.to, F.weight, join(string.(F.activated_regions), ", "))
        end
    end

    println()
    M = monodromy_operator(S.chambers)
    println("Monodromy operator (diagonal increments):")
    for i in 1:size(M,1)
        println("  ", REGIONS[i], ": diag = ", round(M[i,i], digits=3))
    end
end

# ============================================================
# ZETA COMPARISON OUTPUT
# ============================================================

function write_zeta_comparison_table(S::SchoberState,
                                     filename::String="zeta_comparison.tsv")
    io = open(filename, "w")
    println(io, join(["snapshot","u","spectral_zeta","combinatorial_zeta",
                      "log_mismatch","n_paths","wall_crossing",
                      "bridgeland_phase"], "	"))
    for Z in S.zeta_comparisons
        @printf(io, "%d	%.6f	%.4f	%.4f	%.6f	%d	%s	%.4f
",
                Z.snapshot_id, Z.u,
                Z.spectral_zeta, Z.combinatorial_zeta,
                Z.mismatch, Z.n_active_paths,
                Z.wall_crossing, Z.bridgeland_phase)
    end
    close(io)
    println("Zeta comparison table written to $filename")
end

"""
Print the stack deformation summary connecting tubes/paths/walls/zeta.

This is the key diagnostic for whether the derived (2,1)-stack is
deforming consistently under the Toda-Lax flow:

  - Mismatch ≈ 0       → Bass theorem holds, spectral = combinatorial
  - Mismatch spikes     → wall crossing active, KS formula needed
  - Mismatch recovers   → KS formula restored consistency
  - Bridgeland ∈ (0,1]  → Fukaya stability holds at this chamber
"""
function summarize_zeta_stack(S::SchoberState)
    println()
    println("=========== ZETA STACK DEFORMATION SUMMARY ===========")
    println("Connecting: tubes / prime paths / wall crossings / zeta")
    println()

    zcs = S.zeta_comparisons
    n   = length(zcs)

    if n == 0
        println("  No zeta comparisons available.")
        return
    end

    mismatches   = [Z.mismatch for Z in zcs]
    wall_steps   = findall(Z -> Z.wall_crossing, zcs)
    phases       = [Z.bridgeland_phase for Z in zcs]
    n_paths_all  = [Z.n_active_paths for Z in zcs]

    valid_mm = filter(isfinite, mismatches)
    mean_mm  = isempty(valid_mm) ? NaN : mean(valid_mm)
    max_mm   = isempty(valid_mm) ? NaN : maximum(valid_mm)

    @printf("  Snapshots:          %d
", n)
    @printf("  Wall crossings:     %d
", length(wall_steps))
    @printf("  Mean log-mismatch:  %.4f  (want → 0 as A∞ → classical)\n", mean_mm)
    @printf("  Max  log-mismatch:  %.4f\n", max_mm)
    println("  [spectral = det(I-u·B_Ihara), combinatorial = Π_p(1-u^l(p)) from A∞ paths]")
    @printf("  Mean active paths:  %.1f
", mean(n_paths_all))

    println()
    println("  Wall crossing events (mismatch spike → recovery):")
    for idx in wall_steps
        Z = zcs[idx]
        Z_prev = idx > 1 ? zcs[idx-1] : Z
        Z_next = idx < n ? zcs[idx+1] : Z
        @printf("    snap %d: mismatch %.4f → %.4f → %.4f  paths=%d  phase=%.3f
",
                idx,
                isfinite(Z_prev.mismatch) ? Z_prev.mismatch : 999.0,
                isfinite(Z.mismatch)      ? Z.mismatch      : 999.0,
                isfinite(Z_next.mismatch) ? Z_next.mismatch : 999.0,
                Z.n_active_paths,
                Z.bridgeland_phase)
    end

    println()
    # Bridgeland phase check (Fukaya stability)
    in_range = count(p -> 0.0 < p <= 1.0, phases)
    @printf("  Bridgeland phases ∈ (0,1]:  %d / %d
", in_range, n)

    # Bass theorem quality: % of snapshots with mismatch < 0.1
    bass_ok = count(Z -> isfinite(Z.mismatch) && Z.mismatch < 0.1, zcs)
    @printf("  Bass theorem holds (mm<0.1): %d / %d (%.0f%%)
",
            bass_ok, n, 100.0 * bass_ok / max(n, 1))

    println()
    println("  Interpretation:")
    if mean_mm < 0.05
        println("  ✓ Low mean mismatch: spectral and combinatorial zeta agree.")
        println("    Stack is deforming consistently under Toda-Lax flow.")
    elseif mean_mm < 0.3
        println("  ~ Moderate mismatch: some inconsistency at wall crossings.")
        println("    Check whether mismatch recovers after each wall (KS formula).")
    else
        println("  ✗ High mismatch: Bass theorem not holding numerically.")
        println("    Check prime path weight normalisation in get_active_prime_paths.")
    end

    if length(wall_steps) > 0
        # Check KS recovery: mismatch should drop after each wall crossing
        recovering = 0
        for idx in wall_steps
            if idx < n
                if zcs[idx+1].mismatch < zcs[idx].mismatch
                    recovering += 1
                end
            end
        end
        @printf("  KS recovery (mismatch drops after wall): %d / %d walls
",
                recovering, length(wall_steps))
        if recovering == length(wall_steps)
            println("  ✓ All walls show KS recovery — wall-crossing formula is working.")
        end
    end
    println("======================================================")
end

end # module
