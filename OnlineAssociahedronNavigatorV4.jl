# Uses driv-basis for infintesimal deformations for monodromy
###############################################################
# OnlineAssociahedronNavigatorV4.jl
#
# Gerstenhaber‑based monodromy transport:
#   - Builds a Lie algebra matrix from gerstenhaber constants
#   - Uses derivation basis region weights to project to regions
#   - Beam search for best tubing (same as V3)
#
# Requires: JSON3, Statistics, Combinatorics, CairoMakie, LinearAlgebra
###############################################################

module OnlineAssociahedronNavigatorV4

using JSON3
using Statistics
using Combinatorics
using CairoMakie
using LinearAlgebra

export Navigator,
       run_folder!,
       plot_scores,
       plot_transport,
       plot_tubes

###############################################################
# REGIONS & SUPPORT GRAPH
###############################################################

const REGIONS = [:CA1sp,:BLA,:HY,:HPF,:sAMY,:LA]
const N = length(REGIONS)

idx(r::Symbol) = findfirst(==(r), REGIONS)

function support_graph()
    G = Dict(
        :CA1sp => [:HPF,:sAMY],
        :BLA   => [:HPF,:LA,:sAMY],
        :HY    => [:sAMY],
        :HPF   => [:CA1sp,:BLA,:sAMY],
        :sAMY  => [:CA1sp,:BLA,:HY,:HPF,:LA],
        :LA    => [:BLA,:sAMY]
    )
    for v in keys(G)
        for u in G[v]
            if !(v in G[u])
                push!(G[u], v)
            end
        end
    end
    return G
end

const G = support_graph()

###############################################################
# HELPERS
###############################################################

region_hits(s::String) =
    unique([r for r in REGIONS if occursin(String(r), s)])

load_json(file) = JSON3.read(read(file, String))

###############################################################
# COMPATIBILITY & TUBES (same as V3)
###############################################################

function adjacent_sets(A, B)
    for a in A, b in B
        if b in G[a]
            return true
        end
    end
    false
end

function compatible(A::Set{Symbol}, B::Set{Symbol})
    if A ⊆ B || B ⊆ A
        return true
    end
    if isempty(intersect(A, B)) && !adjacent_sets(A, B)
        return true
    end
    false
end

function tubing_ok(T)
    for i in 1:length(T), j in i+1:length(T)
        compatible(T[i], T[j]) || return false
    end
    true
end

function induced_neighbors(S, v)
    [u for u in G[v] if u in S]
end

function connected_subset(sub)
    isempty(sub) && return false
    S = Set(sub)
    seen = Set([sub[1]])
    Q = [sub[1]]
    while !isempty(Q)
        x = popfirst!(Q)
        for y in induced_neighbors(S, x)
            if !(y in seen)
                push!(seen, y)
                push!(Q, y)
            end
        end
    end
    length(seen) == length(S)
end

function all_tubes()
    tubes = Set{Symbol}[]
    for k in 1:length(REGIONS)-1
        for c in combinations(REGIONS, k)
            v = collect(c)
            connected_subset(v) && push!(tubes, Set(v))
        end
    end
    return tubes
end

const BASE_TUBES = all_tubes()

###############################################################
# SCORES (same as V3)
###############################################################

function score_tube(snapshot, τ)
    s = 0.0
    for keyname in (:m6, :m5, :m4)
        if haskey(snapshot, keyname)
            for (k, v) in pairs(snapshot[keyname])
                rs = region_hits(String(k))
                if !isempty(rs) && all(r -> r in τ, rs)
                    w = tryparse(Float64, string(v))
                    w === nothing && (w = 1.0)
                    s += log(1 + abs(w))
                end
            end
        end
    end
    if haskey(snapshot, :prime_paths)
        for item in snapshot[:prime_paths]
            path_str = join(item["path"], " → ")
            rs = region_hits(path_str)
            if !isempty(rs) && all(r -> r in τ, rs)
                w = tryparse(Float64, string(item["weight"]))
                w === nothing && (w = 1.0)
                s += log(1 + abs(w))
            end
        end
    end
    if haskey(snapshot, :prime_higher_ideals)
        for ideal in snapshot[:prime_higher_ideals]
            closure = ideal["closure"]
            if all(sym -> begin
                reg = if startswith(sym, "e_")
                       Symbol(sym[3:end])
                     elseif startswith(sym, "f_")
                       Symbol(split(sym, "_")[2])
                     else
                       nothing
                     end
                reg !== nothing && reg in τ
            end, closure)
                s += ideal["total_support"]
            end
        end
    end
    if haskey(snapshot, :cup_product) && haskey(snapshot, :prime_higher_ideals)
        cup_const = snapshot[:cup_product]
        ideals = snapshot[:prime_higher_ideals]
        n_ideal = length(ideals)
        cup_mat = zeros(Float64, n_ideal, n_ideal)
        for entry in cup_const
            i = entry["i"] + 1
            j = entry["j"] + 1
            coef = abs(entry["coeff"])
            cup_mat[i, j] += coef
            cup_mat[j, i] += coef
        end
        for i in 1:n_ideal, j in i+1:n_ideal
            if all(sym in τ for sym in ideals[i]["closure"]) &&
               all(sym in τ for sym in ideals[j]["closure"])
                s += cup_mat[i, j] * (ideals[i]["total_support"] + ideals[j]["total_support"])
            end
        end
    end
    return s / sqrt(length(τ))
end

score_tubing(snapshot, T) = sum(score_tube(snapshot, τ) for τ in T)

function candidate_tubes(snapshot; topk=15)
    vals = [(score_tube(snapshot, τ), τ) for τ in BASE_TUBES]
    sort!(vals, by=x->x[1], rev=true)
    [vals[i][2] for i in 1:min(topk, length(vals))]
end

function complete_maximal(T, cands, snapshot)
    cur = deepcopy(T)
    vals = [(score_tube(snapshot, τ), τ) for τ in cands]
    sort!(vals, by=x->x[1], rev=true)
    for (_, τ) in vals
        if !(τ in cur)
            T2 = [cur; [τ]]
            if tubing_ok(T2)
                push!(cur, τ)
            end
        end
    end
    return cur
end

function greedy_initial(snapshot)
    cands = candidate_tubes(snapshot)
    complete_maximal(Set{Symbol}[], cands, snapshot)
end

function tubing_signature(T)
    join(sort([join(sort(string.(collect(x))), "_") for x in T]), "|")
end

function neighbors(T, snapshot)
    cands = candidate_tubes(snapshot)
    out = Vector{Vector{Set{Symbol}}}()
    for i in eachindex(T)
        T2 = deepcopy(T)
        deleteat!(T2, i)
        push!(out, complete_maximal(T2, cands, snapshot))
    end
    for τ in cands
        T2 = deepcopy(T)
        push!(T2, τ)
        if tubing_ok(T2)
            push!(out, complete_maximal(T2, cands, snapshot))
        end
    end
    for i in eachindex(T), τ in cands
        T2 = deepcopy(T)
        T2[i] = τ
        if tubing_ok(T2)
            push!(out, complete_maximal(T2, cands, snapshot))
        end
    end
    seen = Set{String}()
    uniq = Vector{Vector{Set{Symbol}}}()
    for X in out
        sig = tubing_signature(X)
        if !(sig in seen)
            push!(seen, sig)
            push!(uniq, X)
        end
    end
    return uniq
end

###############################################################
# GERSTENHABER MONODROMY (NEW)
###############################################################

function build_gerstenhaber_matrix(gerstenhaber_const, dim::Int)
    # Build skew‑symmetric matrix L where L[i,j] = sum_k coeff_ijk (as an infinitesimal generator)
    L = zeros(Float64, dim, dim)
    for entry in gerstenhaber_const
        i = entry["i"] + 1
        j = entry["j"] + 1
        coeff = entry["coeff"]
        # bracket [X_i, X_j] = Σ coeff * X_k; for transport we use the adjoint action
        # Use absolute value to get a non‑negative influence matrix (simplified)
        L[i, j] += abs(coeff)
        L[j, i] += abs(coeff)
    end
    # Normalise rows to get a stochastic transition matrix
    for i in 1:dim
        s = sum(L[i, :])
        if s > 0
            L[i, :] ./= s
        else
            L[i, i] = 1.0
        end
    end
    return L
end

function project_derivations_to_regions(deriv_info, state)
    # state is a probability vector over derivations (length dim)
    # deriv_info: list of dicts with keys "vector" (unused) and "regions" (Dict{String,Float64})
    # Returns region probability vector of length N
    region_vec = zeros(Float64, N)
    for (d_idx, info) in enumerate(deriv_info)
        prob_d = state[d_idx]
        if prob_d == 0.0
            continue
        end
        reg_weights = info["regions"]
        for (reg_str, w) in reg_weights
            r = Symbol(reg_str)
            r_idx = idx(r)
            if r_idx !== nothing
                region_vec[r_idx] += prob_d * w
            end
        end
    end
    # Normalise
    s = sum(region_vec)
    s > 0 && (region_vec ./= s)
    return region_vec
end

function monodromy_gerstenhaber(snapshot)
    if !haskey(snapshot, :gerstenhaber) || !haskey(snapshot, :derivation_basis)
        return nothing
    end
    ger = snapshot[:gerstenhaber]
    basis_info = snapshot[:derivation_basis]
    dim = length(basis_info)
    L = build_gerstenhaber_matrix(ger, dim)
    # Exponentiate to get a transition matrix for a unit time step (simplified: use L itself as a stochastic matrix)
    # For a more accurate flow, we could use exp(L), but L is already normalised to be row‑stochastic.
    return L, dim, basis_info
end

###############################################################
# NAVIGATOR STATE (beam search, plus transport)
###############################################################

mutable struct Navigator
    beam::Vector{Vector{Set{Symbol}}}
    transport::Vector{Float64}          # region probability vector
    deriv_state::Vector{Float64}        # derivation probability vector (if Gerstenhaber available)
    hist_score::Vector{Float64}
    hist_dom::Vector{Int}
    hist_tubes::Vector{Vector{Set{Symbol}}}
end

function Navigator(snapshot)
    T0 = greedy_initial(snapshot)
    # Initialise transport uniformly over regions
    transport = fill(1/N, N)
    # Try to initialise derivation state if derivation_basis exists
    deriv_state = nothing
    if haskey(snapshot, :derivation_basis)
        dim = length(snapshot[:derivation_basis])
        deriv_state = fill(1/dim, dim)
    end
    Navigator([T0], transport, deriv_state, Float64[], Int[], Vector{Vector{Set{Symbol}}}())
end

###############################################################
# BEAM STEP (with Gerstenhaber transport)
###############################################################

function beam_step!(S::Navigator, snapshot; width=7)
    pool = Vector{Vector{Set{Symbol}}}()
    for T in S.beam
        append!(pool, neighbors(T, snapshot))
    end
    sig_map = Dict{String, Vector{Set{Symbol}}}()
    for T in pool
        sig = tubing_signature(T)
        sig_map[sig] = T
    end
    unique_pool = collect(values(sig_map))

    scored = [(score_tubing(snapshot, T), T) for T in unique_pool]
    sort!(scored, by=x->x[1], rev=true)
    keep = min(width, length(scored))
    S.beam = [scored[i][2] for i in 1:keep]

    best = S.beam[1]
    bestscore = scored[1][1]

    # ---- Gerstenhaber transport ----
    mg = monodromy_gerstenhaber(snapshot)
    if mg !== nothing
        L, dim, basis_info = mg
        if S.deriv_state === nothing || length(S.deriv_state) != dim
            # Initialise derivation state if not yet
            S.deriv_state = fill(1/dim, dim)
        end
        # One step of Lie flow: new_deriv_state = L * old_deriv_state
        S.deriv_state = L * S.deriv_state
        S.deriv_state ./= sum(S.deriv_state)
        # Project to region space
        S.transport = project_derivations_to_regions(basis_info, S.deriv_state)
    else
        # Fallback to prime‑path monodromy (as in V3)
        if haskey(snapshot, :prime_paths)
            M = Matrix(I, N, N)
            A = zeros(Float64, N, N)
            for item in snapshot[:prime_paths]
                path_str = join(item["path"], " → ")
                rs = region_hits(path_str)
                length(rs) < 2 && continue
                w = tryparse(Float64, string(item["weight"]))
                w === nothing && (w = 1.0)
                for i in 1:length(rs)-1
                    a = idx(rs[i]); b = idx(rs[i+1])
                    A[b, a] += log(1 + abs(w))
                end
            end
            for j in 1:N
                s = sum(A[:, j])
                s > 0 && (A[:, j] ./= s)
            end
            M += A
            S.transport = M * S.transport
            S.transport ./= sum(S.transport)
        else
            # No transport data – keep previous
        end
    end

    push!(S.hist_score, bestscore)
    push!(S.hist_dom, argmax(S.transport))
    push!(S.hist_tubes, deepcopy(best))
end

###############################################################
# RUN FOLDER
###############################################################

function run_folder!(folder::String)
    files = filter(f -> endswith(lowercase(f), ".json"), readdir(folder))
    sort!(files)
    isempty(files) && error("No JSON files found in $folder")
    snap0 = load_json(joinpath(folder, files[1]))
    S = Navigator(snap0)
    for f in files
        snap = load_json(joinpath(folder, f))
        beam_step!(S, snap)
        println("FILE: $f")
        println("score = ", round(S.hist_score[end], digits=3))
        println("dominant = ", REGIONS[S.hist_dom[end]])
    end
    return S
end

###############################################################
# VISUALISATIONS (same as V3)
###############################################################

function plot_scores(S)
    fig = Figure(size=(900,400))
    ax = Axis(fig[1,1], title="Tube Score")
    lines!(ax, 1:length(S.hist_score), S.hist_score)
    fig
end

function plot_transport(S)
    fig = Figure(size=(900,400))
    ax = Axis(fig[1,1],
        title="Dominant Region",
        yticks=(1:N, string.(REGIONS)))
    scatter!(ax, 1:length(S.hist_dom), S.hist_dom)
    fig
end

function plot_tubes(S)
    vals = zeros(Float64, N, length(S.hist_tubes))
    for t in 1:length(S.hist_tubes)
        for τ in S.hist_tubes[t], r in τ
            vals[idx(r), t] += 1
        end
    end
    fig = Figure(size=(900,450))
    ax = Axis(fig[1,1],
        title="Tube Participation",
        yticks=(1:N, string.(REGIONS)))
    heatmap!(ax, vals)
    fig
end

end
