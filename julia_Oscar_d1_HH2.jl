using Oscar
S = Oscar.Singular

# Define noncommutative algebra
R, (a,f,o) = S.FreeAlgebra(S.QQ, ["a","f","o"], 3)

basis = [:1, :a, :f, :o, :af]

function mult(x, y)
    if x == :1 return y end
    if y == :1 return x end

    if x == :a && y == :f
        return 37, :o
    end

    # truncate higher paths
    return 0, :1
end

basis_list = [:1, :a, :f, :o, :af]
dimA = length(basis_list)

function basis_index(x)
    findfirst(==(x), basis_list)
end

function mult_vec(i,j)
    coeff, res = mult(basis_list[i], basis_list[j])
    v = zeros(QQ, dimA)
    if coeff != 0
        v[basis_index(res)] = coeff
    end
    return v
end



#--------------------------------------------------
# 4. Finite basis (truncate manually)
#--------------------------------------------------
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
# 5. Multiplication table in coordinates
#--------------------------------------------------
function to_vector(x)
    v = zeros(QQ, dimA)
    for (i,b) in enumerate(basis)
        if x == b
            v[i] = 1
        end
    end
    return v
end

mult = Dict{Tuple{Int,Int}, Vector{QQElem}}()

for i in 1:dimA, j in 1:dimA
    prod = basis[i] * basis[j]
    v = zeros(QQ, dimA)

    for k in 1:dimA
        if prod == basis[k]
            v[k] = 1
        end
    end

    mult_vec[(i,j)] = v
end

println("Multiplication table built.")

#--------------------------------------------------
# 6. Index helpers
#--------------------------------------------------
# C¹ index: f(i) → vector
idx1(i,k) = (i-1)*dimA + k

# C² index: φ(i,j) → vector
idx2(i,j,k) = (i-1)*dimA^2 + (j-1)*dimA + k

#--------------------------------------------------
# 7. Build d2 matrix
#--------------------------------------------------
nvars2 = dimA^3
rows2 = []

for a_i in 1:dimA, b_i in 1:dimA, c_i in 1:dimA

    row = zeros(QQ, nvars2)

    # term 1: a * φ(b,c)
    for k in 1:dimA
        coeff = mult[(a_i, k)]
        for m in 1:dimA
            row[idx2(b_i, c_i, m)] += coeff[m]
        end
    end

    # term 2: - φ(ab, c)
    for k in 1:dimA
        if mult[(a_i,b_i)][k] != 0
            for m in 1:dimA
                row[idx2(k, c_i, m)] -= mult[(a_i,b_i)][k]
            end
        end
    end

    # term 3: + φ(a, bc)
    for k in 1:dimA
        if mult[(b_i,c_i)][k] != 0
            for m in 1:dimA
                row[idx2(a_i, k, m)] += mult[(b_i,c_i)][k]
            end
        end
    end

    # term 4: - φ(a,b) * c
    for k in 1:dimA
        coeff = mult[(k, c_i)]
        for m in 1:dimA
            row[idx2(a_i, b_i, k)] -= coeff[m]
        end
    end

    push!(rows2, row)
end

M2 = matrix(QQ, rows2)
println("d2 matrix size: ", size(M2))

#--------------------------------------------------
# 8. Build d1 matrix
#--------------------------------------------------
# d1: C¹ → C²
nvars1 = dimA^2
rows1 = []

for a_i in 1:dimA, b_i in 1:dimA

    row = zeros(QQ, nvars1)

    # (d1 f)(a,b) = a f(b) - f(ab) + f(a) b

    # term 1: a * f(b)
    for k in 1:dimA
        coeff = mult[(a_i, k)]
        for m in 1:dimA
            row[idx1(b_i, m)] += coeff[m]
        end
    end

    # term 2: - f(ab)
    for k in 1:dimA
        if mult[(a_i,b_i)][k] != 0
            for m in 1:dimA
                row[idx1(k, m)] -= mult[(a_i,b_i)][k]
            end
        end
    end

    # term 3: + f(a) * b
    for k in 1:dimA
        coeff = mult[(k, b_i)]
        for m in 1:dimA
            row[idx1(a_i, k)] += coeff[m]
        end
    end

    push!(rows1, row)
end

M1 = matrix(QQ, rows1)
println("d1 matrix size: ", size(M1))

#--------------------------------------------------
# 9. Compute HH²
#--------------------------------------------------
K = nullspace(M2)
println("dim ker(d2) = ", ncols(K))

Im = image(M1)
println("dim im(d1) = ", ncols(Im))

HH2_dim = ncols(K) - ncols(Im)

println("===================================")
println("HH^2 dimension = ", HH2_dim)
println("===================================")
