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

export BlowupPoint,
       BlowupRun,
       run_blowup!,
       plot_divisor,
       plot_generators,
       print_events

###############################################################
# TYPES
###############################################################

mutable struct BlowupPoint
    time::Int
    file::String
    region::Symbol
    g::Vector{Float64}      # generators
    proj::Vector{Float64}   # normalized projective coords
    chart::Int             # dominant generator chart
    severity::Float64
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

    files = filter(f->endswith(lowercase(f),".json"), readdir(folder))
    sort!(files)

    names = [
        "m6",
        "hh2",
        "ger",
        "cup",
        "deriv",
        "ihara"
    ]

    pts = BlowupPoint[]

    for (t,f) in enumerate(files)

        snap = load_json(joinpath(folder,f))

        g = extract_generators(snap)

        if maximum(g) > 0

            p = projectivize(g)
            c = argmax(p)
            sev = norm(g)

            push!(pts,
                BlowupPoint(
                    t,
                    f,
                    infer_region(snap),
                    g,
                    p,
                    c,
                    sev
                )
            )
        end
    end

    BlowupRun(names,pts)
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

end
