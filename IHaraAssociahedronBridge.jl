###############################################################
# IharaAssociahedronBridge.jl
#
# Drop-in bridge:
#   JSON snapshots -> effective graph -> Ihara proxy spectrum
#   + associahedron score / flip coupling
#
# Uses:
#   OnlineAssociahedronNavigatorV2.jl
#
###############################################################

module IharaAssociahedronBridge

using LinearAlgebra
using Statistics
using JSON3
using CairoMakie

include("OnlineAssociahedronNavigatorV2.jl")
using .OnlineAssociahedronNavigatorV2

export BridgeState,
       run_bridge!,
       plot_pole_radius,
       plot_flip_vs_poles,
       plot_stress

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
catch e
    1.0
end

region_hits(s::String) =
    unique([r for r in REGIONS if occursin(String(r), s)])

load_json(file) = JSON3.read(read(file,String))

###############################################################
# BUILD EFFECTIVE ADJACENCY
###############################################################

function base_graph()

    A = zeros(Float64,N,N)

    edges = [
        (:CA1sp,:HPF),
        (:CA1sp,:sAMY),
        (:BLA,:HPF),
        (:BLA,:LA),
        (:BLA,:sAMY),
        (:HY,:sAMY),
        (:HPF,:sAMY),
        (:LA,:sAMY)
    ]

    for (a,b) in edges
        i,j = idx(a), idx(b)
        A[i,j] = 1
        A[j,i] = 1
    end

    A
end

function effective_adjacency(snapshot)

    A = base_graph()

    # prime paths reinforce edges
    if haskey(snapshot,:prime_paths)
        for (k,v) in pairs(snapshot[:prime_paths])
            rs = region_hits(String(k))
            w = log(1 + abs(safe_float(v)))

            for i in 1:length(rs)-1
                a,b = idx(rs[i]), idx(rs[i+1])
                A[a,b] += w
                A[b,a] += w
            end
        end
    end

    # cup products reinforce
    if haskey(snapshot,:cup_product)
        for (k,v) in pairs(snapshot[:cup_product])
            rs = region_hits(String(k))
            w = 0.4 * log(1 + abs(safe_float(v)))
            for r1 in rs, r2 in rs
                if r1 != r2
                    A[idx(r1),idx(r2)] += w
                end
            end
        end
    end

    # gerstenhaber penalizes instability
    if haskey(snapshot,:gerstenhaber)
        for (k,v) in pairs(snapshot[:gerstenhaber])
            rs = region_hits(String(k))
            w = 0.3 * log(1 + abs(safe_float(v)))
            for r1 in rs, r2 in rs
                if r1 != r2
                    A[idx(r1),idx(r2)] -= w
                end
            end
        end
    end

    # m6 stress
    if haskey(snapshot,:m6)
        for (k,v) in pairs(snapshot[:m6])
            rs = region_hits(String(k))
            w = 0.15 * log(1 + abs(safe_float(v)))
            for r in rs
                A[idx(r),idx(r)] += w
            end
        end
    end

    max.(A,0.0)
end

###############################################################
# IHARA PROXY
#
# We use adjacency spectrum proxy:
# radius(A), entropy(|eig|)
###############################################################

function ihara_proxy(A)

    vals = eigvals(Symmetric((A+A')/2))
    mags = abs.(vals)

    r = maximum(mags)

    p = mags ./ max(sum(mags),1e-9)
    H = -sum(x>0 ? x*log(x) : 0.0 for x in p)

    (radius=r, entropy=H, eig=vals)
end

###############################################################
# STATE
###############################################################

mutable struct BridgeState
    nav
    pole_radius::Vector{Float64}
    pole_entropy::Vector{Float64}
    flip::Vector{Int}
    stress::Vector{Float64}
end

###############################################################
# RUN
###############################################################

function run_bridge!(folder::String)

    files = filter(f->endswith(lowercase(f),".json"), readdir(folder))
    sort!(files)

    snap0 = load_json(joinpath(folder,files[1]))
    nav = run_folder!(folder)   # reuse navigator batch result

    B = BridgeState(nav, Float64[], Float64[], Int[], Float64[])

    prevsig = ""

    for (t,f) in enumerate(files)

        snap = load_json(joinpath(folder,f))
        A = effective_adjacency(snap)
        z = ihara_proxy(A)

        push!(B.pole_radius, z.radius)
        push!(B.pole_entropy, z.entropy)

        sig = tubing_signature(nav.hist_tubes[t])
        push!(B.flip, sig == prevsig ? 0 : 1)
        prevsig = sig

        gs = haskey(snap,:gerstenhaber) ? length(keys(snap[:gerstenhaber])) : 0
        m6 = haskey(snap,:m6) ? length(keys(snap[:m6])) : 0
        push!(B.stress, gs + m6)
    end

    B
end

###############################################################
# PLOTS
###############################################################

function plot_pole_radius(B)

    fig = Figure(size=(900,400))
    ax = Axis(fig[1,1], title="Ihara Proxy Pole Radius")
    lines!(ax,1:length(B.pole_radius),B.pole_radius)
    fig
end

function plot_flip_vs_poles(B)

    fig = Figure(size=(900,400))
    ax = Axis(fig[1,1], title="Flips vs Pole Radius")
    lines!(ax,1:length(B.pole_radius),B.pole_radius)
    scatter!(ax,findall(==(1),B.flip),
                B.pole_radius[findall(==(1),B.flip)])
    fig
end

function plot_stress(B)

    fig = Figure(size=(900,400))
    ax = Axis(fig[1,1], title="Stress")
    lines!(ax,1:length(B.stress),B.stress)
    fig
end

end
