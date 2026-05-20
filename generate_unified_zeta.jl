"""
generate_unified_zeta.jl
========================
Generates unified_zeta.json from ainf_export_*.json files.

This replaces the unified_zeta computation from run_iharaSingV2.jl
without requiring the full bridge pipeline.

Usage:
    julia generate_unified_zeta.jl [folder] [graph_type]
    
    folder:     path to ainf_export JSON files (default: .)
    graph_type: Q_6 | Q_7P | Q_7L | Q_8 (default: Q_7P)

Output:
    unified_zeta.json  with keys: times, magnitude, log_magnitude
"""

using JSON3, LinearAlgebra, Statistics, Printf

folder     = length(ARGS) >= 1 ? ARGS[1] : "."
graph_type = length(ARGS) >= 2 ? ARGS[2] : "Q_7P"

# ── Region setup (matches IharaAssociahedronBridgeV2) ─────────────────────────

REGIONS_MAP = Dict(
    "Q_6"  => [:CA1sp, :BLA, :HY, :HPF, :sAMY, :LA],
    "Q_7P" => [:CA1sp, :BLA, :HY, :HPF, :sAMY, :LA, :PAL],
    "Q_7L" => [:CA1sp, :BLA, :HY, :HPF, :sAMY, :LA, :LSX],
    "Q_8"  => [:CA1sp, :BLA, :HY, :HPF, :sAMY, :LA, :LSX, :PAL],
)

REGIONS = get(REGIONS_MAP, graph_type, REGIONS_MAP["Q_7P"])
N = length(REGIONS)
REGION_IDX = Dict(r => i for (i,r) in enumerate(REGIONS))

function idx(sym::Symbol)
    return get(REGION_IDX, sym, nothing)
end

function region_from_string(s::String)
    for r in REGIONS
        if occursin(string(r), s)
            return r
        end
    end
    return nothing
end

function region_hits(path_str::String)
    hits = Symbol[]
    for r in REGIONS
        if occursin(string(r), path_str)
            push!(hits, r)
        end
    end
    return hits
end

function region_hits_from_path(path)
    regions = Symbol[]
    for sym in path
        s = string(sym)
        if startswith(s, "f_") || startswith(s, "e_")
            parts = split(s, "_")
            if length(parts) >= 2
                r = Symbol(parts[2])
                if r in REGIONS
                    push!(regions, r)
                end
            end
        end
    end
    return regions
end

# ── Base graph from connectome_graphs.json ────────────────────────────────────

function load_base_graph()
    gpath = joinpath(folder, "connectome_graphs.json")
    if !isfile(gpath)
        gpath = joinpath(@__DIR__, "connectome_graphs.json")
    end
    !isfile(gpath) && return zeros(Float64, N, N)
    
    try
        gdata = JSON3.read(read(gpath, String))
        g = get(gdata, Symbol(graph_type), nothing)
        isnothing(g) && return zeros(Float64, N, N)
        
        A = zeros(Float64, N, N)
        for arrow in g[:arrows]
            vs = Int(arrow["v_src"]) 
            vt = Int(arrow["v_tgt"])
            w  = Float64(arrow["weight"])
            vs <= N && vt <= N && (A[vs, vt] += log(1 + w))
        end
        return A
    catch e
        @warn "Could not load base graph: $e"
        return zeros(Float64, N, N)
    end
end

BASE_GRAPH = load_base_graph()
println("Base graph loaded: $(N)×$(N), non-zero=$(sum(BASE_GRAPH .!= 0))")

# ── Effective adjacency (matches IharaAssociahedronBridgeV2 logic) ────────────

function safe_float(x)
    try
        v = Float64(x)
        return isfinite(v) ? v : 0.0
    catch
        return 0.0
    end
end

function effective_adjacency(snap)
    A = copy(BASE_GRAPH)
    
    # prime_paths
    if haskey(snap, :prime_paths)
        for item in snap[:prime_paths]
            path = haskey(item, "path") ? item["path"] : (haskey(item, :path) ? item[:path] : [])
            rs = region_hits_from_path(path)
            length(rs) < 2 && continue
            w = log(1 + abs(safe_float(get(item, "weight", 0.0))))
            for i in 1:length(rs)-1
                a = idx(rs[i]); b = idx(rs[i+1])
                if a !== nothing && b !== nothing
                    A[a,b] += w; A[b,a] += w
                end
            end
        end
    end
    
    # cup_product
    if haskey(snap, :cup_product)
        total = sum((abs(safe_float(get(e, "coeff", 0.0))) for e in snap[:cup_product]); init=0.0)
        total > 0 && (A .+= 0.1 * log(1 + total))
    end
    
    # gerstenhaber
    if haskey(snap, :gerstenhaber)
        total = sum((abs(safe_float(get(e, "coeff", 0.0))) for e in snap[:gerstenhaber]); init=0.0)
        total > 0 && (A .-= 0.05 * log(1 + total))
    end
    
    # m6
    if haskey(snap, :m6)
        for (k, v) in pairs(snap[:m6])
            rs = region_hits(string(k))
            w = 0.15 * log(1 + abs(safe_float(v)))
            for r in rs
                i = idx(r)
                i !== nothing && (A[i,i] += w)
            end
        end
    end
    
    # clamp
    A .= max.(A, 0.0)
    return A
end

# ── Process all JSON files ────────────────────────────────────────────────────

files = sort(filter(f -> endswith(f, ".json") && startswith(f, "ainf_export"),
                    readdir(folder)))

println("Found $(length(files)) ainf_export JSON files")
isempty(files) && (println("ERROR: no files found in $folder"); exit(1))

unified_mag    = Float64[]
unified_logmag = Float64[]
timestamps     = Float64[]

for (i, f) in enumerate(files)
    snap = JSON3.read(read(joinpath(folder, f), String))
    
    A = effective_adjacency(snap)
    evals = eigvals(Symmetric(A))   # symmetric → real eigenvalues, faster
    ρ = maximum(abs.(evals))
    t = ρ > 1e-10 ? min(0.9/ρ, 0.5) : 0.5
    
    M = I - t * A
    d = det(M)
    
    if abs(d) < 1e-12
        push!(unified_mag, 1e12)
        push!(unified_logmag, 28.0)
    else
        mag = abs(1.0 / d)
        push!(unified_mag, mag)
        push!(unified_logmag, -log(abs(d)))
    end
    
    # Extract timestamp from filename: ainf_export_1234567890.12.json
    ts = try
        parts = split(replace(f, ".json"=>""), "_")
        parse(Float64, parts[end])
    catch
        Float64(i)
    end
    push!(timestamps, ts)
    
    i % 100 == 0 && @printf("  %d/%d  mag=%.4f\n", i, length(files), unified_mag[end])
end

# ── Save unified_zeta.json ────────────────────────────────────────────────────

out = Dict(
    "times"         => timestamps,
    "magnitude"     => unified_mag,
    "log_magnitude" => unified_logmag,
    "n_points"      => length(unified_mag),
    "graph_type"    => graph_type,
    "n_regions"     => N,
)

outpath = joinpath(folder, "unified_zeta.json")
open(outpath, "w") do io
    JSON3.write(io, out)
end

println()
println("Saved: $outpath")
println("  n_points = $(length(unified_mag))")
@printf("  mag: min=%.4f  max=%.4f  mean=%.4f\n",
        minimum(unified_mag), maximum(unified_mag), mean(unified_mag))
println()
println("Now run: julia run_ReesGrassmannBridge.jl")
