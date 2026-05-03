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
#   IharaAssociahedronBridgeV2.jl
###############################################################

module SingularityTracker

using JSON3
using Statistics
using CairoMakie

include("OnlineAssociahedronNavigatorV3.jl")
include("IharaAssociahedronBridgeV2.jl")

using .OnlineAssociahedronNavigatorV3
using .IharaAssociahedronBridgeV2

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
    # Helper to find key with maximum value in a Dict
    function argmax_dict(d)
        isempty(d) && return nothing
        max_val = -Inf
        max_key = nothing
        for (k, v) in d
            if v > max_val
                max_val = v
                max_key = k
            end
        end
        return max_key
    end

    # Try explicit region field
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
            best = argmax_dict(counts)
            best !== nothing && return best
        end
    end

    # Fallback: highest total_support in ideals
    if haskey(snap, :prime_higher_ideals)
        best_region = :Unknown
        best_score = -Inf
        for ideal in snap[:prime_higher_ideals]
            sup = safe_float(get(ideal, "total_support", 0.0))
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
                if occursin(String(r), string(k))
                    counts[r] += safe_float(v)
                end
            end
        end
        if maximum(values(counts)) > 0
            best = argmax_dict(counts)
            best !== nothing && return best
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
    event_perversities::Vector{Int}   # new
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

    files = filter(f -> occursin(r"ainf_export_(?:\w+_)?\d+(?:\.\d+)?\.json", basename(f)), readdir(folder))
    sort!(files)

    ev_times = Int[]
    ev_regions = Symbol[]
    event_perversities = Int[]          # new: store perversity of each blow‑up
    pre_rad = Float64[]
    post_rad = Float64[]
    gains = Float64[]
    face_changes = Int[]

    for i in eachindex(files)
        snap = load_json(joinpath(folder, files[i]))
        if is_blowup(snap)
            push!(ev_times, i)
            push!(ev_regions, infer_region(snap))

            # Compute max perversity among prime higher ideals
            max_perv = 0
            if haskey(snap, :prime_higher_ideals)
                for ideal in snap[:prime_higher_ideals]
                    perv = get(ideal, "perversity", 0)
                    if perv > max_perv
                        max_perv = perv
                    end
                end
            end
            push!(event_perversities, max_perv)

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

    # Return with the new field (8 arguments)
    return TrackerResult(B, ev_times, ev_regions, event_perversities,
                         pre_rad, post_rad, gains, face_changes)
end

###############################################################
# TABLE & PLOTS
###############################################################

function event_table(T::TrackerResult)
    println("----------------------------------------------------")
    println("time | perv | region | pre_radius | post_radius | gain")
    println("----------------------------------------------------")
    for i in eachindex(T.event_times)
        println(
            T.event_times[i], " | ",
            T.event_perversities[i], " | ",
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
    scatter!(ax3, T.event_times, ys, color = T.event_perversities, colormap = :plasma)

    fig
end

end
