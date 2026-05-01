############################################################
# DriverMonodromyTransport.jl
#
# Uses:
#   AssociahedronFromAinf.jl
#
# Purpose:
#   Combine
#     (A) Monodromy transport from recurring prime paths
#     (B) Associahedron state flips
#
# Produces:
#   Dynamic transport state over JSON simulation files
#
# Concept:
#   JSON_t
#      -> dominant paths
#      -> monodromy matrix M_t
#      -> best associahedron vertex V_t
#      -> transport operator T_t = M_t * state(V_t)
#
############################################################

using LinearAlgebra
using Statistics
using JSON3

include("AssociahedronFromAinf.jl")
using .AssociahedronFromAinf

############################################################
# 1. REGION INDEX
############################################################

const REGIONS = [:CA1sp,:BLA,:HY,:HPF,:sAMY,:LA]
const N = length(REGIONS)

region_index(r::Symbol) = findfirst(==(r), REGIONS)

############################################################
# 2. PARSE REGION TOKENS FROM STRING
############################################################

function regions_in_string(s::String)
    out = Symbol[]
    for r in REGIONS
        occursin(String(r), s) && push!(out,r)
    end
    return out
end

############################################################
# 3. PRIME PATH MONODROMY MATRIX
#
# Build transport matrix from prime_paths
############################################################

function monodromy_matrix(snapshot)

    M = zeros(Float64,N,N)

    if !haskey(snapshot,:prime_paths)
        return I + M
    end

    pp = snapshot[:prime_paths]

    for (k,v) in pairs(pp)

        rs = regions_in_string(String(k))
        length(rs) < 2 && continue

        # weight
        w = try
            Float64(v)
        catch
            1.0
        end

        # consecutive transitions
        for i in 1:length(rs)-1
            a = region_index(rs[i])
            b = region_index(rs[i+1])
            M[b,a] += log(1 + abs(w))
        end
    end

    # normalize columns
    for j in 1:N
        s = sum(M[:,j])
        if s > 0
            M[:,j] ./= s
        end
    end

    # add identity persistence
    return Matrix(I,N,N) + M
end

############################################################
# 4. ASSOCIAHEDRON STATE VECTOR
#
# Convert best tubing into region emphasis vector
############################################################

function tubing_vector(T)

    x = zeros(Float64,N)

    for tube in T
        wt = length(tube)

        for r in tube
            x[region_index(r)] += wt
        end
    end

    s = sum(x)
    s > 0 && (x ./= s)

    return x
end

############################################################
# 5. TRANSPORT STEP
############################################################

function transport_step(M, x)
    y = M * x
    s = sum(y)
    s > 0 && (y ./= s)
    return y
end

############################################################
# 6. DIAGNOSTICS
############################################################

function entropy(x)
    e = 0.0
    for v in x
        if v > 1e-12
            e -= v * log(v)
        end
    end
    return e
end

function dominant_region(x)
    REGIONS[argmax(x)]
end

############################################################
# 7. MAIN DRIVER
############################################################

function run_transport(folder::String)

    files = filter(f->endswith(lowercase(f),".json"), readdir(folder))
    sort!(files)

    G = build_support_graph()
    verts = maximal_tubings(G)

    println("Maximal associahedron vertices = ", length(verts))
    println()

    prev_vertex = nothing
    prev_dom = nothing

    for (t,f) in enumerate(files)

        snap = load_snapshot(joinpath(folder,f))

        ####################################################
        # best associahedron state
        ####################################################
        vals = [score_tubing(snap,T) for T in verts]
        idx = argmax(vals)
        Tbest = verts[idx]

        ####################################################
        # monodromy matrix
        ####################################################
        M = monodromy_matrix(snap)

        ####################################################
        # transport state
        ####################################################
        x0 = tubing_vector(Tbest)
        xt = transport_step(M,x0)

        dom = dominant_region(xt)
        H = entropy(xt)

        ####################################################
        # print report
        ####################################################
        println("------------------------------------------------")
        println("STEP ", t, " : ", f)
        println("Best associahedron vertex: ", idx)
        println("Dominant transport region: ", dom)
        println("Entropy: ", round(H,digits=4))
        println("Distribution: ", round.(xt,digits=3))

        if prev_vertex !== nothing && prev_vertex != idx
            println("Tube flip: ", prev_vertex, " -> ", idx)
        end

        if prev_dom !== nothing && prev_dom != dom
            println("Transport shift: ", prev_dom, " -> ", dom)
        end

        prev_vertex = idx
        prev_dom = dom
    end
end

############################################################
# 8. OPTIONAL CONTINUOUS ITERATED MONODROMY
############################################################

function run_iterated(folder::String; steps_per_file=10)

    files = filter(f->endswith(lowercase(f),".json"), readdir(folder))
    sort!(files)

    G = build_support_graph()
    verts = maximal_tubings(G)

    x = fill(1/N,N)

    for f in files

        snap = load_snapshot(joinpath(folder,f))

        vals = [score_tubing(snap,T) for T in verts]
        idx = argmax(vals)
        Tbest = verts[idx]

        M = monodromy_matrix(snap)
        anchor = tubing_vector(Tbest)

        # blend current state with associahedron anchor
        x = 0.5x + 0.5anchor

        for _ in 1:steps_per_file
            x = transport_step(M,x)
        end

        println(f,
            "  dom=", dominant_region(x),
            "  entropy=", round(entropy(x),digits=4),
            "  state=", round.(x,digits=3))
    end
end

############################################################
# RUN EXAMPLE
############################################################
 run_transport("./")
# run_iterated("/path/to/jsonfolder")

