using LinearAlgebra

#--------------------------------------------------
# 1. Base field
#--------------------------------------------------
const T = Float64

#--------------------------------------------------
# 2. Basis (5D algebra)
#--------------------------------------------------
basis = [:e, :a, :f, :o, :z]   # z = zero-like placeholder
dimA = length(basis)

function idx(x)
    findfirst(==(x), basis)
end

#--------------------------------------------------
# 3. Associative multiplication
#--------------------------------------------------
function mult(x, y)

    # identity
    if x == :e return (1.0, y) end
    if y == :e return (1.0, x) end

    # core noncommutative relation
    if x == :a && y == :f
        return (1.0, :o)
    end

    # truncate paths (length ≥ 2 → zero)
    if x == :a && y == :a return (0.0, nothing) end
    if x == :f && y == :f return (0.0, nothing) end
    if x == :f && y == :a return (0.0, nothing) end

    # anything involving o → zero (prevents associativity issues)
    if x == :o || y == :o
        return (0.0, nothing)
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
# 4. Indexing
#--------------------------------------------------
idx1(i,k) = (i-1)*dimA + k
idx2(i,j,k) = (i-1)*dimA^2 + (j-1)*dimA + k

nvars1 = dimA^2
nvars2 = dimA^3

#--------------------------------------------------
# 5. Build d2
#--------------------------------------------------
rows2 = Vector{Vector{T}}()

for a in 1:dimA, b in 1:dimA, c in 1:dimA
    block = zeros(T, dimA, nvars2)

    # a * φ(b,c)
    for k in 1:dimA
        coeff = mult_vec(a, k)
        for m in 1:dimA
            block[m, idx2(b,c,k)] += coeff[m]
        end
    end

    # -φ(ab, c)
    ab = mult_vec(a,b)
    for k in 1:dimA
        if ab[k] != 0
            for m in 1:dimA
                block[m, idx2(k,c,m)] -= ab[k]
            end
        end
    end

    # +φ(a, bc)
    bc = mult_vec(b,c)
    for k in 1:dimA
        if bc[k] != 0
            for m in 1:dimA
                block[m, idx2(a,k,m)] += bc[k]
            end
        end
    end

    # -φ(a,b) * c
    for k in 1:dimA
        coeff = mult_vec(k,c)
        for m in 1:dimA
            block[m, idx2(a,b,k)] -= coeff[m]
        end
    end

    for r in 1:dimA
        push!(rows2, vec(block[r,:]))
    end
end

M2 = hcat(rows2...)'
println("d2 size: ", size(M2))

#--------------------------------------------------
# 6. Build d1
#--------------------------------------------------
rows1 = Vector{Vector{T}}()

for a in 1:dimA, b in 1:dimA
    block = zeros(T, dimA, nvars1)

    # term 1: a * f(b)
    for k in 1:dimA
        coeff = mult_vec(a, k)
        for m in 1:dimA
            block[m, idx1(b, k)] += coeff[m]
        end
    end

    # term 2: -f(ab)
    ab = mult_vec(a,b)
    for k in 1:dimA
        if ab[k] != 0
            for m in 1:dimA
                block[m, idx1(k, m)] -= ab[k]
            end
        end
    end

    # term 3: +f(a) * b
    for k in 1:dimA
        coeff = mult_vec(k, b)
        for m in 1:dimA
            block[m, idx1(a, k)] += coeff[m]
        end
    end

    for r in 1:dimA
        push!(rows1, vec(block[r,:]))
    end
end

M1 = hcat(rows1...)'
println("d1 size: ", size(M1))

#--------------------------------------------------
# 7. CHECK associativity
#--------------------------------------------------
function check_associativity()
    for x in basis, y in basis, z in basis
        lhs = zeros(T, dimA)
        xy = mult_vec(idx(x), idx(y))
        for k in 1:dimA
            if xy[k] != 0
                lhs += xy[k] * mult_vec(k, idx(z))
            end
        end

        rhs = zeros(T, dimA)
        yz = mult_vec(idx(y), idx(z))
        for k in 1:dimA
            if yz[k] != 0
                rhs += yz[k] * mult_vec(idx(x), k)
            end
        end

        if norm(lhs - rhs) > 1e-8
            println("FAIL at ", x, y, z)
            return false
        end
    end
    return true
end

println("Associative? ", check_associativity())

#--------------------------------------------------
# 8. CHECK chain condition
#--------------------------------------------------
println("||d2 * d1|| = ", norm(M2 * M1))

#--------------------------------------------------
# 9. HH² computation
#--------------------------------------------------
function compute_HH2()

    K = nullspace(M2)
    println("dim ker(d2) = ", size(K,2))

    A = hcat(K, -M1)
    N = nullspace(A)

    dim_intersection = 0

    for i in 1:size(N,2)
        if norm(N[1:size(K,2), i]) > 1e-8
            dim_intersection += 1
        end
    end

    println("dim(im(d1) ∩ ker(d2)) = ", dim_intersection)

    HH2_dim = size(K,2) - dim_intersection

    println("===================================")
    println("HH^2 dimension = ", HH2_dim)
    println("===================================")

end

compute_HH2()

#--------------------------------------------------
# 10. Extract HH² generators
#--------------------------------------------------
println("\nExtracting HH^2 generators (clean)...")

K = nullspace(M2)

# orthonormal basis of im(d1)
Q1, _ = qr(M1)
Q1 = Matrix(Q1)

function project_out_image(v)
    return v - Q1 * (Q1' * v)
end

HH2_generators = []

for i in 1:size(K,2)
    v = K[:, i]
    v_clean = project_out_image(v)

    if norm(v_clean) > 1e-8
        push!(HH2_generators, v_clean / norm(v_clean))
    end
end

println("Raw candidates: ", length(HH2_generators))

# stack and orthogonalize
if length(HH2_generators) > 0
    M = hcat(HH2_generators...)
    Q, _ = qr(M)
    Q = Matrix(Q)

    # keep only independent ones
    HH2_basis = []

    for i in 1:size(Q,2)
        if norm(Q[:,i]) > 1e-8
            push!(HH2_basis, Q[:,i])
        end
    end

    println("Final HH^2 basis size: ", length(HH2_basis))
else
    HH2_basis = []
end

#=
println("\nExtracting HH^2 generators...")

# basis of ker(d2)
K = nullspace(M2)

# orthonormal basis of im(d1)
Q1, _ = qr(M1)
Q1 = Matrix(Q1)

function in_image(v)
    proj = Q1 * (Q1' * v)
    return norm(v - proj) < 1e-8
end

# collect independent HH² generators
HH2_generators = []

for i in 1:size(K,2)
    v = K[:, i]

    if !in_image(v)
        push!(HH2_generators, v)
    end
end

println("Number of HH^2 generators found: ", length(HH2_generators))
=#

function print_generator(v, idx2, basis)
    dimA = length(basis)

    println("---- Generator ----")

    for i in 1:dimA, j in 1:dimA
        terms = []

        for k in 1:dimA
            val = v[idx2(i,j,k)]
            if abs(val) > 1e-6
                push!(terms, "$(round(val, digits=3))*$(basis[k])")
            end
        end

        if !isempty(terms)
            println("φ(", basis[i], ", ", basis[j], ") = ", join(terms, " + "))
        end
    end
end

for i in 1:min(5, length(HH2_generators))
    print_generator(HH2_generators[i], idx2, basis)
end
