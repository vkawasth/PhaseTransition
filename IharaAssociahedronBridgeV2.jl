###############################################################
# IharaAssociahedronBridgeV2.jl
#
# Drop-in bridge:
#   JSON snapshots -> effective graph -> Ihara proxy spectrum
#   + associahedron score / flip coupling
#
# Uses:
#   OnlineAssociahedronNavigatorV3.jl   (or V4)
#
# Corrected for actual JSON structure:
#   prime_paths: array of {path, weight}
#   cup_product: array of {i,j,k,coeff}
#   gerstenhaber: array of {i,j,k,coeff}
#   blowup detection: prime_higher_ideals key
###############################################################

module IharaAssociahedronBridge

using LinearAlgebra
using Statistics
using JSON3
using CairoMakie
import Base: basename

# Choose your navigator version (V3 or V4). We assume V3 for compatibility.
include("OnlineAssociahedronNavigatorV3.jl")
using .OnlineAssociahedronNavigatorV3

export BridgeState,
       run_bridge!,
       plot_pole_radius,
       plot_flip_vs_poles,
       plot_stress,
       load_transition_times,
       nearest_file_index

###############################################################
# REGIONS
###############################################################

const REGIONS = [:CA1sp,:BLA,:HY,:HPF,:sAMY,:LA]
const N = length(REGIONS)

idx(r::Symbol) = findfirst(==(r), REGIONS)

###############################################################
# HELPERS
###############################################################

safe_float(x) = try
    Float64(x)
catch
    1.0
end

function region_hits(s::String)
    unique([r for r in REGIONS if occursin(String(r), s)])
end

load_json(file) = JSON3.read(read(file, String))

###############################################################
# BASE STATIC GRAPH
###############################################################

function base_graph()
    A = zeros(Float64, N, N)
    edges = [
        (:CA1sp, :HPF),
        (:CA1sp, :sAMY),
        (:BLA,   :HPF),
        (:BLA,   :LA),
        (:BLA,   :sAMY),
        (:HY,    :sAMY),
        (:HPF,   :sAMY),
        (:LA,    :sAMY)
    ]
    for (a,b) in edges
        i, j = idx(a), idx(b)
        A[i,j] = 1.0
        A[j,i] = 1.0
    end
    A
end

###############################################################
# EFFECTIVE ADJACENCY FROM JSON DATA
###############################################################

function effective_adjacency(snapshot)
    A = base_graph()

    # ---- prime_paths (array) ----
    if haskey(snapshot, :prime_paths)
        for item in snapshot[:prime_paths]
            path_str = join(item["path"], " → ")
            rs = region_hits(path_str)
            if length(rs) < 2
                continue
            end
            w = log(1 + abs(safe_float(item["weight"])))
            for i in 1:length(rs)-1
                a = idx(rs[i]); b = idx(rs[i+1])
                A[a,b] += w
                A[b,a] += w
            end
        end
    end

    # ---- cup_product (array of {i,j,k,coeff}) ----
    if haskey(snapshot, :cup_product)
        # Cup product constants are given in derivation indices, not region strings.
        # For simplicity, we aggregate the absolute coefficients as a global stress
        # and add it to all edges (or to diagonal). A more refined mapping would
        # require derivation→region projection, which is in V4 but not here.
        # We'll simply add a uniform boost to all edges proportional to total cup strength.
        total_cup = 0.0
        for entry in snapshot[:cup_product]
            total_cup += abs(safe_float(entry["coeff"]))
        end
        # Weak influence on all edges (optional)
        if total_cup > 0
            global_boost = 0.1 * log(1 + total_cup)
            A .+= global_boost
        end
    end

    # ---- gerstenhaber (array) ----
    if haskey(snapshot, :gerstenhaber)
        # Similarly, use total bracket magnitude as a penalty on all edges
        total_bracket = 0.0
        for entry in snapshot[:gerstenhaber]
            total_bracket += abs(safe_float(entry["coeff"]))
        end
        if total_bracket > 0
            global_penalty = 0.05 * log(1 + total_bracket)
            A .-= global_penalty
        end
    end

    # ---- m6 obstruction (dictionary) ----
    if haskey(snapshot, :m6)
        for (k, v) in pairs(snapshot[:m6])
            rs = region_hits(String(k))
            w = 0.15 * log(1 + abs(safe_float(v)))
            for r in rs
                i = idx(r)
                if i !== nothing
                    A[i,i] += w
                end
            end
        end
    end

    # Ensure non‑negative entries
    return max.(A, 0.0)
end

###############################################################
# IHARA PROXY (spectral radius + eigenvalue entropy)
###############################################################

function ihara_proxy(A)
    # Use symmetric part (it is already symmetric from construction)
    vals = eigvals(Symmetric(A))
    mags = abs.(vals)
    r = maximum(mags)
    p = mags ./ max(sum(mags), 1e-12)
    H = -sum(x > 0 ? x * log(x) : 0.0 for x in p)
    return (radius = r, entropy = H, eigvals = vals)
end

###############################################################
# BRIDGE STATE
###############################################################

mutable struct BridgeState
    nav                    # Navigator object (from run_folder!)
    pole_radius::Vector{Float64}
    pole_entropy::Vector{Float64}
    flip::Vector{Int}
    stress::Vector{Float64}
end

###############################################################
# RUN THE BRIDGE
###############################################################

function run_bridge!(folder::String)
    # First, run the navigator to get tubing history
    nav = run_folder!(folder)   # this will process all JSON files

    files = filter(f -> endswith(lowercase(f), ".json"), readdir(folder))
    sort!(files)

    B = BridgeState(nav, Float64[], Float64[], Int[], Float64[])

    prev_sig = ""

    for (t, f) in enumerate(files)
        snap = load_json(joinpath(folder, f))
        A = effective_adjacency(snap)
        z = ihara_proxy(A)

        push!(B.pole_radius, z.radius)
        push!(B.pole_entropy, z.entropy)

        # tubing signature (uses function from navigator)
        sig = tubing_signature(nav.hist_tubes[t])
        flip = (sig == prev_sig) ? 0 : 1
        push!(B.flip, flip)
        prev_sig = sig

        # stress: number of non‑zero Gerstenhaber entries + number of m6 entries
        gs = haskey(snap, :gerstenhaber) ? length(snap[:gerstenhaber]) : 0
        m6 = haskey(snap, :m6) ? length(keys(snap[:m6])) : 0
        push!(B.stress, gs + m6)
    end

    return B
end

###############################################################
# PLOTTING
###############################################################

function plot_pole_radius(B)
    fig = Figure(size=(900,400))
    ax = Axis(fig[1,1], title="Ihara Proxy Pole Radius")
    lines!(ax, 1:length(B.pole_radius), B.pole_radius)
    fig
end

function plot_flip_vs_poles(B)
    fig = Figure(size=(900,400))
    ax = Axis(fig[1,1], title="Flips vs Pole Radius")
    lines!(ax, 1:length(B.pole_radius), B.pole_radius)
    flip_indices = findall(==(1), B.flip)
    if !isempty(flip_indices)
        scatter!(ax, flip_indices, B.pole_radius[flip_indices], color=:red)
    end
    fig
end

function plot_stress(B)
    fig = Figure(size=(900,400))
    ax = Axis(fig[1,1], title="Stress (Gerstenhaber + m6 counts)")
    lines!(ax, 1:length(B.stress), B.stress)
    fig
end

function load_transition_times(file="transition_times.json")
    if !isfile(file)
        return Float64[]
    end
    data = JSON3.read(read(file, String))
    return Float64.(data)
end

function nearest_file_index(times, filenames)
    # filenames are the full paths; we assume each filename contains a timestamp (e.g., "ainf_export_1777435790.69.json")
    # Extract timestamp from filename
    ts = Float64[]
    for f in filenames
        m = match(r"([0-9]+\.[0-9]+)\.json", basename(f))
        if m !== nothing
            push!(ts, parse(Float64, m.captures[1]))
        else
            push!(ts, NaN)
        end
    end
    # For each transition time, find closest index
    indices = Int[]
    for t in times
        # find closest timestamp
        idx = argmin(abs.(ts .- t))
        push!(indices, idx)
    end
    return indices
end

end
