##############################
# AssociahedronFromAinf.jl
#
# Build a weighted graph associahedron using:
#   (1) Quiver/path algebra support graph
#   (2) Obstruction data from your JSON exports:
#         m3,m4,m5,m6,HH2_dim,
#         prime_paths, support_infty, etc.
#
# Output:
#   - tubes
#   - maximal tubings
#   - scores
#   - flip graph
#   - best state trajectory over JSON files
##############################

module AssociahedronFromAinf

using JSON3
using Combinatorics
using Statistics

export load_snapshot,
       build_support_graph,
       connected_tubes,
       maximal_tubings,
       score_tube,
       score_tubing,
       tubing_flip_graph,
       analyze_folder

########################################################
# 0. YOUR FIXED REGION SET
########################################################

const REGIONS = [
    :CA1sp,
    :BLA,
    :HY,
    :HPF,
    :sAMY,
    :LA
]

########################################################
# 1. BASE SUPPORT GRAPH FROM YOUR QUIVER
########################################################

function build_support_graph()
    G = Dict{Symbol,Vector{Symbol}}(
        :CA1sp => [:HPF,:sAMY],
        :BLA   => [:HPF,:LA,:sAMY],
        :HY    => [:sAMY],
        :HPF   => [:CA1sp,:BLA,:sAMY],
        :sAMY  => [:CA1sp,:BLA,:HY,:HPF,:LA],
        :LA    => [:BLA,:sAMY]
    )

    # make symmetric
    for v in keys(G)
        for u in G[v]
            if !(v in G[u])
                push!(G[u], v)
            end
        end
    end
    return G
end

########################################################
# 2. JSON LOADER
########################################################

function load_snapshot(file::String)
    raw = JSON3.read(read(file,String))
    return raw
end

########################################################
# 3. HELPERS
########################################################

function all_subsets(vs)
    out = Vector{Vector{Symbol}}()
    n = length(vs)
    for k in 1:n-1
        for c in combinations(vs,k)
            push!(out, collect(c))
        end
    end
    return out
end

function induced_neighbors(G,S,v)
    [u for u in G[v] if u in S]
end

function is_connected_subset(G, subset::Vector{Symbol})
    isempty(subset) && return false
    S = Set(subset)

    seen = Set([subset[1]])
    queue = [subset[1]]

    while !isempty(queue)
        x = popfirst!(queue)
        for y in induced_neighbors(G,S,x)
            if !(y in seen)
                push!(seen,y)
                push!(queue,y)
            end
        end
    end

    return length(seen)==length(S)
end

########################################################
# 4. TUBES
########################################################

function connected_tubes(G)
    tubes = Vector{Set{Symbol}}()
    for sub in all_subsets(REGIONS)
        if is_connected_subset(G,sub)
            push!(tubes, Set(sub))
        end
    end
    return tubes
end

########################################################
# 5. COMPATIBILITY
########################################################

function adjacent_sets(G,A,B)
    for a in A, b in B
        if b in G[a]
            return true
        end
    end
    return false
end

function compatible(G,A,B)
    if A ⊆ B || B ⊆ A
        return true
    end
    if isempty(intersect(A,B)) && !adjacent_sets(G,A,B)
        return true
    end
    return false
end

function tubing_ok(G,T)
    for i in 1:length(T), j in i+1:length(T)
        compatible(G,T[i],T[j]) || return false
    end
    return true
end

########################################################
# 6. MAXIMAL TUBINGS
########################################################

function maximal_tubings(G)
    tubes = connected_tubes(G)
    good = Vector{Vector{Set{Symbol}}}()

    for r in 1:length(tubes)
        for cand in combinations(tubes,r)
            T = collect(cand)
            tubing_ok(G,T) || continue

            maximal = true
            for τ in tubes
                if !(τ in T)
                    if tubing_ok(G,[T; [τ]])
                        maximal = false
                        break
                    end
                end
            end

            maximal && push!(good,T)
        end
    end

    m = maximum(length.(good))
    return [T for T in good if length(T)==m]
end

########################################################
# 7. SCORE TUBES USING JSON OBSTRUCTION DATA
########################################################

# parse region mentions in arbitrary key strings
function region_hits(str)
    s = String(str)
    hits = Symbol[]
    for r in REGIONS
        occursin(String(r), s) && push!(hits,r)
    end
    return unique(hits)
end

function count_matches(obj, subset::Set{Symbol})
    c = 0.0
    if obj === nothing
        return c
    end

    for (k,v) in pairs(obj)
        rs = region_hits(k)
        if !isempty(rs) && all(x->x in subset, rs)
            c += 1.0
        end
    end
    return c
end

function score_tube(snapshot, τ::Set{Symbol})

    m3 = haskey(snapshot,:m3) ? snapshot[:m3] : nothing
    m4 = haskey(snapshot,:m4) ? snapshot[:m4] : nothing
    m5 = haskey(snapshot,:m5) ? snapshot[:m5] : nothing
    m6 = haskey(snapshot,:m6) ? snapshot[:m6] : nothing
    pp = haskey(snapshot,:prime_paths) ? snapshot[:prime_paths] : nothing

    s =
        1.0*count_matches(m3,τ) +
        2.0*count_matches(m4,τ) +
        3.0*count_matches(m5,τ) +
        5.0*count_matches(m6,τ) +
        4.0*count_matches(pp,τ)

    # size regularization
    s /= sqrt(length(τ))

    return s
end

function score_tubing(snapshot,T)
    sum(score_tube(snapshot,τ) for τ in T)
end

function score_tube_using_ideals(snapshot, τ::Set{Symbol})
    total = 0.0
    for ideal in snapshot.prime_higher_ideals
        # ideal is a dict with keys "path", "weight", "closure", "total_support"
        # Check if all symbols in the closure belong to the tube τ
        if all(sym -> symbol_region(sym) in τ, ideal["closure"])
            total += ideal["total_support"]
        end
    end
    return total
end
########################################################
# 8. FLIP GRAPH
########################################################

function one_flip(T1,T2)
    A = [x for x in T1 if !(x in T2)]
    B = [x for x in T2 if !(x in T1)]

    if length(A)==1 && length(B)==1
        return true, A[1], B[1]
    else
        return false, nothing, nothing
    end
end

function tubing_flip_graph(vertices)
    E = []
    for i in 1:length(vertices), j in i+1:length(vertices)
        ok,a,b = one_flip(vertices[i],vertices[j])
        ok && push!(E,(i,j,a,b))
    end
    return E
end

########################################################
# 9. FOLDER ANALYSIS
########################################################

function analyze_folder(folder::String)

    files = filter(f->endswith(lowercase(f),".json"), readdir(folder))
    sort!(files)

    G = build_support_graph()
    verts = maximal_tubings(G)
    flips = tubing_flip_graph(verts)

    println("Regions: ", REGIONS)
    println("Maximal tubings: ", length(verts))
    println("Flip edges: ", length(flips))
    println()

    prev_best = nothing

    for f in files
        snap = load_snapshot(joinpath(folder,f))

        vals = [score_tubing(snap,T) for T in verts]
        idx = argmax(vals)

        println("FILE: ", f)
        println("Best tubing vertex = ", idx)
        println("Score = ", vals[idx])

        if prev_best !== nothing && prev_best != idx
            println("Transition: ", prev_best, " -> ", idx)
        end

        prev_best = idx
        println()
    end

    return verts, flips
end

end
