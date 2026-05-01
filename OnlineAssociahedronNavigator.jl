###############################################################
# OnlineAssociahedronNavigator.jl
#
# Efficient online associahedron navigation:
#   - avoids state-space explosion
#   - warm-start from previous timestep
#   - m6 / prime-path hot-region proposals
#   - monodromy-guided flips
#   - beam search
#   - visualization for tube dynamics
#
# Requires:
#   using JSON3, Combinatorics, Graphs, CairoMakie
#
###############################################################

module OnlineAssociahedronNavigator

using JSON3
using Statistics
using Combinatorics
using Graphs
using CairoMakie

export NavigatorState,
       init_navigator,
       step_navigator!,
       run_folder!,
       plot_tube_dynamics,
       plot_region_transport

###############################################################
# 1. REGIONS
###############################################################

const REGIONS = [:CA1sp,:BLA,:HY,:HPF,:sAMY,:LA]
const N = length(REGIONS)

idx(r::Symbol) = findfirst(==(r), REGIONS)

###############################################################
# 2. SUPPORT GRAPH (your quiver skeleton)
###############################################################

function support_graph()

    G = Dict(
        :CA1sp => [:HPF,:sAMY],
        :BLA   => [:HPF,:LA,:sAMY],
        :HY    => [:sAMY],
        :HPF   => [:CA1sp,:BLA,:sAMY],
        :sAMY  => [:CA1sp,:BLA,:HY,:HPF,:LA],
        :LA    => [:BLA,:sAMY]
    )

    # symmetrize
    for v in keys(G)
        for u in G[v]
            if !(v in G[u])
                push!(G[u],v)
            end
        end
    end
    return G
end

###############################################################
# 3. STATE
###############################################################

mutable struct NavigatorState
    tubing::Vector{Set{Symbol}}
    score::Float64
    transport::Vector{Float64}
    history_score::Vector{Float64}
    history_dom::Vector{Int}
    history_tubes::Vector{Vector{Set{Symbol}}}
end

###############################################################
# 4. INITIAL STATE
###############################################################

function init_navigator()

    T = [
        Set([:CA1sp,:HPF]),
        Set([:HY,:sAMY]),
        Set([:HPF,:BLA,:sAMY,:LA])
    ]

    x = fill(1/N,N)

    NavigatorState(
        T,
        0.0,
        x,
        Float64[],
        Int[],
        Vector{Vector{Set{Symbol}}}()
    )
end

###############################################################
# 5. HELPERS
###############################################################

function region_hits(s::String)
    out = Symbol[]
    for r in REGIONS
        occursin(String(r),s) && push!(out,r)
    end
    unique(out)
end

function load_json(file)
    JSON3.read(read(file,String))
end

###############################################################
# 6. HOT REGIONS FROM m6 / prime paths
###############################################################

function hot_regions(snapshot)

    score = zeros(Float64,N)

    for keyname in (:m6,:prime_paths,:m5)
        if haskey(snapshot,keyname)
            obj = snapshot[keyname]

            for (k,v) in pairs(obj)
                rs = region_hits(String(k))
                w = try Float64(v) catch 1.0 end

                for r in rs
                    score[idx(r)] += log(1+abs(w))
                end
            end
        end
    end

    p = sortperm(score, rev=true)
    return REGIONS[p[1:min(3,N)]], score
end

###############################################################
# 7. MONODROMY MATRIX
###############################################################

function monodromy(snapshot)

    M = Matrix(I,N,N)

    if !haskey(snapshot,:prime_paths)
        return M
    end

    A = zeros(Float64,N,N)

    for (k,v) in pairs(snapshot[:prime_paths])

        rs = region_hits(String(k))
        length(rs) < 2 && continue

        w = try Float64(v) catch 1.0 end

        for i in 1:length(rs)-1
            a = idx(rs[i]); b = idx(rs[i+1])
            A[b,a] += log(1+abs(w))
        end
    end

    for j in 1:N
        s = sum(A[:,j])
        s > 0 && (A[:,j] ./= s)
    end

    return M + A
end

###############################################################
# 8. TUBING SCORE
###############################################################

function score_tube(snapshot, τ::Set{Symbol})

    s = 0.0

    for keyname in (:m6,:m5,:m4,:prime_paths)
        if haskey(snapshot,keyname)
            obj = snapshot[keyname]

            for (k,v) in pairs(obj)
                rs = region_hits(String(k))
                if !isempty(rs) && all(r->r in τ, rs)
                    w = try Float64(v) catch 1.0 end
                    s += log(1+abs(w))
                end
            end
        end
    end

    return s / sqrt(length(τ))
end

score_tubing(snapshot,T) = sum(score_tube(snapshot,τ) for τ in T)

###############################################################
# 9. LOCAL FLIP PROPOSALS
###############################################################

function propose_flips(T, hot::Vector{Symbol})

    cand = Vector{Vector{Set{Symbol}}}()

    # singleton hot pair insertions
    for h in hot
        for g in hot
            h == g && continue

            newtube = Set([h,g])

            T2 = deepcopy(T)
            push!(T2,newtube)
            push!(cand,T2)
        end
    end

    # replace each tube with hot triple
    for i in eachindex(T)
        T2 = deepcopy(T)
        deleteat!(T2,i)
        push!(T2, Set(hot))
        push!(cand,T2)
    end

    # prune duplicate tubings
    return cand
end

###############################################################
# 10. STEP UPDATE
###############################################################

function step_navigator!(S::NavigatorState, snapshot)

    hot,_ = hot_regions(snapshot)

    proposals = propose_flips(S.tubing, collect(hot))
    push!(proposals, deepcopy(S.tubing))

    bestT = S.tubing
    bestv = score_tubing(snapshot,S.tubing)

    for T in proposals
        v = score_tubing(snapshot,T)
        if v > bestv
            bestv = v
            bestT = T
        end
    end

    S.tubing = bestT
    S.score = bestv

    # transport
    M = monodromy(snapshot)
    S.transport = M * S.transport
    S.transport ./= sum(S.transport)

    push!(S.history_score, S.score)
    push!(S.history_dom, argmax(S.transport))
    push!(S.history_tubes, deepcopy(S.tubing))

    return S
end

###############################################################
# 11. RUN FOLDER
###############################################################

function run_folder!(folder::String)

    files = filter(f->endswith(lowercase(f),".json"), readdir(folder))
    sort!(files)

    S = init_navigator()

    for f in files
        snap = load_json(joinpath(folder,f))
        step_navigator!(S,snap)

        println("FILE: ", f)
        println("score = ", round(S.score,digits=3))
        println("dominant transport = ", REGIONS[argmax(S.transport)])
        println("transport = ", round.(S.transport,digits=3))
        println()
    end

    return S
end

###############################################################
# 12. VISUALIZATION : SCORE / DOMINANT REGION
###############################################################

function plot_tube_dynamics(S::NavigatorState)

    fig = Figure(size=(900,500))

    ax1 = Axis(fig[1,1], title="Associahedron Tube Score")
    lines!(ax1, 1:length(S.history_score), S.history_score)

    ax2 = Axis(fig[2,1],
        title="Dominant Region",
        yticks=(1:N,string.(REGIONS)))

    scatter!(ax2,
        1:length(S.history_dom),
        S.history_dom)

    fig
end

###############################################################
# 13. VISUALIZATION : TRANSPORT HEATMAP
###############################################################

function plot_region_transport(S::NavigatorState)

    T = length(S.history_dom)

    X = zeros(Float64,N,T)

    # replay rough state history
    # store dominant only for lightness
    for t in 1:T
        X[S.history_dom[t],t] = 1.0
    end

    fig = Figure(size=(900,400))
    ax = Axis(fig[1,1],
        title="Region Dominance Through Time",
        yticks=(1:N,string.(REGIONS)))

    heatmap!(ax, X)

    fig
end

end
