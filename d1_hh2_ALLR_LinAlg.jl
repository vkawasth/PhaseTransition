using LinearAlgebra

const T = Float64

#--------------------------------------------------
# 1. Nodes
#--------------------------------------------------
nodes = [:CA1sp, :HPF, :BLA]

# identities
e = Dict(n => Symbol("e_" * String(n)) for n in nodes)

# initial edges (you define)
edges = Set([
    (:CA1sp, :HPF),
    (:HPF, :BLA),
    (:CA1sp, :BLA)
])

#--------------------------------------------------
# 2. Relations (ADD YOUR FULL LIST HERE)
#--------------------------------------------------
raw_relations = [
    ((:CA1sp,:HPF), (:HPF,:BLA), 37.0, (:CA1sp,:BLA)),
    # ADD MORE HERE
]

#--------------------------------------------------
# 3. Build path-closed edge set
#--------------------------------------------------
for (a,b,_,c) in raw_relations
    push!(edges, a)
    push!(edges, b)
    push!(edges, c)
end

# convert to symbols
f = Dict((i,j) => Symbol("f_" * String(i) * "_" * String(j)) for (i,j) in edges)

#--------------------------------------------------
# 4. Basis
#--------------------------------------------------
basis = vcat(collect(values(e)), collect(values(f)))
dimA = length(basis)

function idx(x)
    findfirst(==(x), basis)
end

#--------------------------------------------------
# 5. Build relation dictionary
#--------------------------------------------------
relations = Dict{Tuple{Symbol,Symbol}, Tuple{Float64,Symbol}}()

for (a,b,w,c) in raw_relations
    relations[(f[a], f[b])] = (w, f[c])
end

#--------------------------------------------------
# 6. Multiplication (CRITICAL FIX)
#--------------------------------------------------
function mult(x, y)

    # identity
    if startswith(String(x), "e_")
        return (1.0, y)
    end
    if startswith(String(y), "e_")
        return (1.0, x)
    end

    # relation-defined
    if haskey(relations, (x,y))
        return relations[(x,y)]
    end

    # composable path?
    if startswith(String(x), "f_") && startswith(String(y), "f_")
        xi, xj = split(String(x)[3:end], "_")
        yi, yj = split(String(y)[3:end], "_")

        if xj == yi
            candidate = Symbol("f_" * xi * "_" * yj)

            if candidate in basis
                # path exists but no weight specified → treat as ZERO (important)
                return (0.0, nothing)
            end
        end
    end

    return (0.0, nothing)
end

function mult_vec(i,j)
    coeff, res = mult(basis[i], basis[j])
    v = zeros(T, dimA)
    if coeff != 0.0 && res !== nothing
        v[idx(res)] = coeff
    end
    return v
end

#--------------------------------------------------
# 7. Indexing
#--------------------------------------------------
idx1(i,k) = (i-1)*dimA + k
idx2(i,j,k) = (i-1)*dimA^2 + (j-1)*dimA + k

nvars1 = dimA^2
nvars2 = dimA^3

#--------------------------------------------------
# 8. Build d2
#--------------------------------------------------
function build_d2()
    rows = Vector{Vector{T}}()

    for a in 1:dimA, b in 1:dimA, c in 1:dimA
        block = zeros(T, dimA, nvars2)

        for k in 1:dimA
            coeff = mult_vec(a,k)
            for m in 1:dimA
                block[m, idx2(b,c,k)] += coeff[m]
            end
        end

        ab = mult_vec(a,b)
        for k in 1:dimA
            if ab[k] != 0
                for m in 1:dimA
                    block[m, idx2(k,c,m)] -= ab[k]
                end
            end
        end

        bc = mult_vec(b,c)
        for k in 1:dimA
            if bc[k] != 0
                for m in 1:dimA
                    block[m, idx2(a,k,m)] += bc[k]
                end
            end
        end

        for k in 1:dimA
            coeff = mult_vec(k,c)
            for m in 1:dimA
                block[m, idx2(a,b,k)] -= coeff[m]
            end
        end

        for r in 1:dimA
            push!(rows, vec(block[r,:]))
        end
    end

    return hcat(rows...)'
end

#--------------------------------------------------
# 9. Build d1
#--------------------------------------------------
function build_d1()
    rows = Vector{Vector{T}}()

    for a in 1:dimA, b in 1:dimA
        block = zeros(T, dimA, nvars1)

        for k in 1:dimA
            coeff = mult_vec(a,k)
            for m in 1:dimA
                block[m, idx1(b,k)] += coeff[m]
            end
        end

        ab = mult_vec(a,b)
        for k in 1:dimA
            if ab[k] != 0
                for m in 1:dimA
                    block[m, idx1(k,m)] -= ab[k]
                end
            end
        end

        for k in 1:dimA
            coeff = mult_vec(k,b)
            for m in 1:dimA
                block[m, idx1(a,k)] += coeff[m]
            end
        end

        for r in 1:dimA
            push!(rows, vec(block[r,:]))
        end
    end

    return hcat(rows...)'
end

#--------------------------------------------------
# 10. Main
#--------------------------------------------------
function main()

    println("Basis size: ", dimA)

    M2 = build_d2()
    M1 = build_d1()

    println("d2 size: ", size(M2))
    println("d1 size: ", size(M1))
    println("||d2 * d1|| = ", norm(M2 * M1))

    K = nullspace(M2)
    dimK = size(K,2)

    println("dim ker(d2) = ", dimK)

    rank_d1 = rank(M1)

    combined = hcat(K, M1)
    rank_combined = rank(combined)

    dim_intersection = dimK + rank_d1 - rank_combined

    println("dim(im(d1) ∩ ker(d2)) = ", dim_intersection)

    HH2_dim = dimK - dim_intersection

    println("===================================")
    println("HH^2 dimension = ", HH2_dim)
    println("===================================")
end

main()