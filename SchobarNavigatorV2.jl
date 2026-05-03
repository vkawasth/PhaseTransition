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
       write_wall_table

# ============================================================
# CONSTANTS
# ============================================================

const REGIONS = [:CA1sp, :BLA, :HY, :HPF, :sAMY, :LA]

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
end

struct FlipFunctor
    from::Int
    to::Int
    weight::Float64
    singular::Bool
    activated_regions::Vector{Symbol}
end

struct SchoberState
    chambers::Vector{ChamberCategory}
    functors::Vector{FlipFunctor}
end

# ============================================================
# JSON LOADING
# ============================================================

load_snapshot(file::String) = JSON3.read(read(file, String))

# ============================================================
# CHAMBER BUILDING
# ============================================================

function build_chamber(id::Int, snap, time::Float64)

    # --- Support scores from prime_higher_ideals ---
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

    # If no scores, assign default 1.0 to all regions
    if isempty(scores)
        for r in REGIONS
            scores[r] = 1.0
        end
    end

    # Normalize to avoid overflow
    max_score = maximum(values(scores))
    if max_score > 0
        scores = Dict(k => v / max_score for (k,v) in scores)
    end

    perversity = compute_perversity(scores)
    gap = compute_spectral_gap(snap)
    obs = compute_obstruction(snap)
    dom = dominant_region(snap)

    return ChamberCategory(id, time, dom, scores, perversity, gap, obs)
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

function run_schober_path(files::Vector{String})
    # Filter only ainf_export files
    #ainf_files = filter(f -> occursin(r"ainf_export_\d+", basename(f)), files)
    ainf_files = filter(f -> startswith(basename(f), "ainf_export_") && endswith(f, ".json"), files)
    if isempty(ainf_files)
        error("No ainf_export_*.json files found.")
    end
    sort!(ainf_files)

    chambers = ChamberCategory[]
    for (i, f) in enumerate(ainf_files)
        snap = load_snapshot(f)
        time = extract_timestamp(f)
        if time == 0.0
            time = Float64(i)  # fallback
        end
        push!(chambers, build_chamber(i, snap, time))
    end

    functors = FlipFunctor[]
    for i in 1:length(chambers)-1
        push!(functors, build_functor(chambers[i], chambers[i+1]))
    end

    return SchoberState(chambers, functors)
end

# ============================================================
# OUTPUT TABLES
# ============================================================

function write_chamber_table(S::SchoberState, filename::String="chambers.tsv")
    io = open(filename, "w")
    println(io, "id\ttime\tdominant\tgap\tobs\tscore\tsupport_sum\tperv_sum")
    for C in S.chambers
        @printf(io, "%d\t%.3f\t%s\t%.4f\t%.2f\t%.3f\t%.3f\t%d\n",
                C.id, C.time, C.dominant_region,
                C.spectral_gap, C.obstruction,
                chamber_score(C),
                sum(values(C.support_scores)),
                sum(values(C.perversity)))
    end
    close(io)
    println("Chamber table written to $filename")
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

end # module
