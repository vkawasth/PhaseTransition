using Oscar

#--------------------------------------------------
# 1. Base field
#--------------------------------------------------
QQ = RationalField()

#--------------------------------------------------
# 2. Free algebra (noncommutative)
#--------------------------------------------------
A_free, (a,f,o) = FreeAlgebra(QQ, ["a","f","o"])

#--------------------------------------------------
# 3. Define weighted relation
#--------------------------------------------------
I = ideal(A_free, [a*f - 37*o])

A = A_free / I

println("Algebra defined.")

#--------------------------------------------------
# 4. Truncate algebra manually (degree ≤ 2)
#--------------------------------------------------
# Basis elements we allow:
# 1, a, f, o, a*f

oneA = A(1)

basis = [
    oneA,
    A(a),
    A(f),
    A(o),
    A(a*f)
]

dimA = length(basis)
println("dim(A) = ", dimA)

#--------------------------------------------------
# 5. Multiplication table (in coordinates)
#--------------------------------------------------
# Represent elements as vectors in basis

function to_vector(x)
    v = zeros(QQ, dimA)
    for (i,b) in enumerate(basis)
        if x == b
            v[i] = 1
        end
    end
    return v
end

# multiplication lookup
mult = Dict()

for i in 1:dimA, j in 1:dimA
    prod = basis[i] * basis[j]

    # reduce modulo ideal automatically
    prod_vec = zeros(QQ, dimA)

    for k in 1:dimA
        if prod == basis[k]
            prod_vec[k] = 1
        end
    end

    mult[(i,j)] = prod_vec
end

println("Multiplication table built.")

#--------------------------------------------------
# 6. Build C^2 variables
# φ(i,j) = vector in A → dimA coefficients
# total vars = dimA^3
#--------------------------------------------------
nvars = dimA^3

# index map
function idx(i,j,k)
    return (i-1)*dimA^2 + (j-1)*dimA + k
end

#--------------------------------------------------
# 7. Build d2 equations
# dφ(a,b,c)= aφ(b,c) - φ(ab,c) + φ(a,bc) - φ(a,b)c = 0
#--------------------------------------------------
rows = []

for a_i in 1:dimA, b_i in 1:dimA, c_i in 1:dimA

    row = zeros(QQ, nvars)

    # term 1: a * φ(b,c)
    for k in 1:dimA
        coeff = mult[(a_i, k)]
        for m in 1:dimA
            row[idx(b_i, c_i, m)] += coeff[m]
        end
    end

    # term 2: - φ(ab, c)
    for k in 1:dimA
        if mult[(a_i,b_i)][k] != 0
            for m in 1:dimA
                row[idx(k, c_i, m)] -= mult[(a_i,b_i)][k]
            end
        end
    end

    # term 3: + φ(a, bc)
    for k in 1:dimA
        if mult[(b_i,c_i)][k] != 0
            for m in 1:dimA
                row[idx(a_i, k, m)] += mult[(b_i,c_i)][k]
            end
        end
    end

    # term 4: - φ(a,b) * c
    for k in 1:dimA
        coeff = mult[(k, c_i)]
        for m in 1:dimA
            row[idx(a_i, b_i, k)] -= coeff[m]
        end
    end

    push!(rows, row)
end

M = matrix(QQ, rows)

println("d2 matrix size: ", size(M))

#--------------------------------------------------
# 8. Compute kernel of d2
#--------------------------------------------------
K = nullspace(M)

println("dim ker(d2) = ", ncols(K))

#--------------------------------------------------
# 9. (Optional) Build d1 and quotient
#--------------------------------------------------
println("NOTE: d1 not yet implemented → this is Z^2, not HH^2")

