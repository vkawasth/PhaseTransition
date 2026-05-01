###############################################################
# SingularityTracker.jl
#
# Mixed snapshot tracker:
#   regular JSON + blow-up JSON
#
# Detects:
#   - singularity events
#   - migrating crisis region
#   - pre/post stabilization
#   - associahedron face changes
#
# Depends on:
#   OnlineAssociahedronNavigatorV3.jl
#   IharaAssociahedronBridge.jl
###############################################################

module SingularityTracker

using JSON3
using Statistics
using CairoMakie

include("OnlineAssociahedronNavigatorV3.jl")
include("IharaAssociahedronBridge.jl")

using .OnlineAssociahedronNavigatorV3
using .IharaAssociahedronBridge

export TrackerResult,
       run_tracker!,
       plot_tracker,
       event_table

###############################################################
# REGIONS
###############################################################

const REGIONS = [:CA1sp,:BLA,:HY,:HPF,:sAMY,:LA]

###############################################################
# HELPERS
###############################################################

load_json(file) = JSON3.read(read(file, String))

safe_float(x) = try; Float64(x); catch; 0.0; end

function region_hits(s::String)
    unique([r for r in REGIONS if occursin(String(r), s)])
end

###############################################################
# EVENT DETECTION (using actual keys from your JSON)
###############################################################

function is_blowup(snap)
    # Blow‑up JSON files contain prime_higher_ideals (array) and may have vtk_file.
    return haskey(snap, :prime_higher_ideals) || haskey(snap, :vtk_file)
end

###############################################################
# REGION EXTRACTION FROM BLOW‑UP SNAPSHOT
###############################################################

function infer_region(snap)
    # Try explicit region field (unlikely)
    if haskey(snap, :region)
        r = Symbol(String(snap[:region]))
        r in REGIONS && return r
    end

    # Scan prime_higher_ideals (they contain closure symbols)
    if haskey(snap, :prime_higher_ideals)
        counts = Dict(r => 0.0 for r in REGIONS)
        for ideal in snap[:prime_higher_ideals]
            for sym in ideal["closure"]
                for r in REGIONS
                    if occursin(String(r), String(sym))
                        counts[r] += 1.0
                    end
                end
            end
        end
        if maximum(values(counts)) > 0
            return argmax(values(counts)) |> x -> collect(keys(counts))[x]
        end
    end

    # Fallback: highest total_support in ideals
    if haskey(snap, :prime_higher_ideals)
        best_region = :Unknown
        best_score = -Inf
        for ideal in snap[:prime_higher_ideals]
            sup = safe_float(ideal.get("total_support", 0.0))
            for sym in ideal["closure"]
                for r in REGIONS
                    if occursin(String(r), String(sym))
                        if sup > best_score
                            best_score = sup
                            best_region = r
                        end
                    end
                end
            end
        end
        if best_region !== :Unknown
            return best_region
        end
    end

    # Last resort: use m6 keys
    if haskey(snap, :m6)
        counts = Dict(r => 0.0 for r in REGIONS)
        for (k, v) in pairs(snap[:m6])
            for r in REGIONS
                if occursin(String(r), String(k))
                    counts[r] += safe_float(v)
                end
            end
        end
        if maximum(values(counts)) > 0
            return argmax(values(counts)) |> x -> collect(keys(counts))[x]
        end
    end

    return :Unknown
end

###############################################################
# RESULT TYPE
###############################################################

mutable struct TrackerResult
    bridge::BridgeState
    event_times::Vector{Int}
    event_regions::Vector{Symbol}
    pre_radius::Vector{Float64}
    post_radius::Vector{Float64}
    gain::Vector{Float64}
    face_change::Vector{Int}
end

###############################################################
# RUN TRACKER
###############################################################

function run_tracker!(folder::String)
    B = run_bridge!(folder)

    files = filter(f -> endswith(lowercase(f), ".json"), readdir(folder))
    sort!(files)

    ev_times = Int[]
    ev_regions = Symbol[]
    pre_rad = Float64[]
    post_rad = Float64[]
    gains = Float64[]
    face_changes = Int[]

    for i in eachindex(files)
        snap = load_json(joinpath(folder, files[i]))
        if is_blowup(snap)
            push!(ev_times, i)
            push!(ev_regions, infer_region(snap))

            pre = i > 1 ? B.pole_radius[i-1] : B.pole_radius[i]
            post = i < length(files) ? B.pole_radius[i+1] : B.pole_radius[i]
            push!(pre_rad, pre)
            push!(post_rad, post)
            push!(gains, pre - post)

            # face change: 1 if tubing flips at this time
            fc = i > 1 ? B.flip[i] : 0
            push!(face_changes, fc)
        end
    end

    return TrackerResult(B, ev_times, ev_regions, pre_rad, post_rad, gains, face_changes)
end

###############################################################
# TABLE & PLOTS
###############################################################

function event_table(T::TrackerResult)
    println("----------------------------------------------------")
    println("time | region | pre_radius | post_radius | gain")
    println("----------------------------------------------------")
    for i in eachindex(T.event_times)
        println(
            T.event_times[i], " | ",
            T.event_regions[i], " | ",
            round(T.pre_radius[i], digits=4), " | ",
            round(T.post_radius[i], digits=4), " | ",
            round(T.gain[i], digits=4)
        )
    end
    println("----------------------------------------------------")
end

function plot_tracker(T::TrackerResult)
    B = T.bridge
    fig = Figure(size=(1000, 800))

    # top: pole radius
    ax1 = Axis(fig[1,1], title="Ihara Pole Radius")
    lines!(ax1, 1:length(B.pole_radius), B.pole_radius)
    for t in T.event_times
        scatter!(ax1, [t], [B.pole_radius[t]], color=:red)
    end

    # middle: tube score
    ax2 = Axis(fig[2,1], title="Tube Score")
    lines!(ax2, 1:length(B.nav.hist_score), B.nav.hist_score)
    for t in T.event_times
        scatter!(ax2, [t], [B.nav.hist_score[t]], color=:red)
    end

    # bottom: singular region index
    region_index = Dict(
        :CA1sp => 1, :BLA => 2, :HY => 3, :HPF => 4,
        :sAMY => 5, :LA => 6, :Unknown => 0
    )
    ys = [region_index[r] for r in T.event_regions]
    ax3 = Axis(fig[3,1],
        title="Singularity Region",
        yticks = (0:6, ["Unk","CA1sp","BLA","HY","HPF","sAMY","LA"])
    )
    scatter!(ax3, T.event_times, ys, color=:red)

    fig
end

end
