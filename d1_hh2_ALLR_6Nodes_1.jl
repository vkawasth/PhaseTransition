using LinearAlgebra
using SparseArrays
using SuiteSparse

# =========================
# 1. Algebra & Geometry Setup
# =========================
const T = Float64

nodes = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA]

# These are your manual observations. 
# The script will now treat these as 'targets' and ensure associativity.
raw_coeffs = Dict(
    (:f_CA1sp_HPF, :f_HPF_BLA) => 37.0,
    (:f_CA1sp_HPF, :f_HPF_sAMY) => 12.0,
    (:f_CA1sp_sAMY, :f_sAMY_BLA) => 69.0,
    (:f_CA1sp_sAMY, :f_sAMY_HY) => 16.0,
    (:f_CA1sp_sAMY, :f_sAMY_HPF) => 78.0,
    (:f_BLA_sAMY, :f_sAMY_HY) => 68.0,
    (:f_BLA_sAMY, :f_sAMY_HPF) => 9.0,
    (:f_BLA_sAMY, :f_sAMY_LA) => 73.0,
    (:f_BLA_LA, :f_LA_sAMY) => 14.0,
    (:f_HY_sAMY, :f_sAMY_BLA) => 72.0,
    (:f_HY_sAMY, :f_sAMY_HPF) => 19.0,
    (:f_HY_sAMY, :f_sAMY_LA) => 60.0,
    (:f_HPF_CA1sp, :f_CA1sp_sAMY) => 97.0,
    (:f_HPF_BLA, :f_BLA_sAMY) => 66.0,
    (:f_HPF_BLA, :f_BLA_LA) => 83.0,
    (:f_HPF_sAMY, :f_sAMY_BLA) => 56.0,
    (:f_HPF_sAMY, :f_sAMY_HY) => 22.0,
    (:f_HPF_sAMY, :f_sAMY_LA) => 98.0,
    (:f_sAMY_BLA, :f_BLA_LA) => 64.0,
    (:f_sAMY_HPF, :f_HPF_CA1sp) => 39.0,
    (:f_sAMY_HPF, :f_HPF_BLA) => 46.0,
    (:f_sAMY_LA, :f_LA_BLA) => 27.0,
    (:f_LA_BLA, :f_BLA_sAMY) => 9.0,
    (:f_LA_sAMY, :f_sAMY_BLA) => 54.0,
    (:f_LA_sAMY, :f_sAMY_HY) => 96.0,
    (:f_LA_sAMY, :f_sAMY_HPF) => 84.0,
    # Identities (loops) - using the ones you provided
    (:f_CA1sp_HPF, :f_HPF_CA1sp) => 67.0,
    (:f_HPF_CA1sp, :f_CA1sp_HPF) => 67.0,
    (:f_BLA_LA, :f_LA_BLA) => 85.0,
    (:f_LA_BLA, :f_BLA_LA) => 85.0,
    (:f_BLA_sAMY, :f_sAMY_BLA) => 66.0,
    (:f_sAMY_BLA, :f_BLA_sAMY) => 66.0,
    (:f_HY_sAMY, :f_sAMY_HY) => 65.0,
    (:f_sAMY_HY, :f_HY_sAMY) => 65.0
)

# Extract all unique arrows into basis
all_arrows = Set{Symbol}()
for (k, _) in raw_coeffs; push!(all_arrows, k[1], k[2]); end
basis = unique([[Symbol("e_$n") for n in nodes]; collect(all_arrows)])
dimA = length(basis)
b_index = Dict(b => i for (i, b) in enumerate(basis))

# =========================
# 2. Projective Weight Resolver
# =========================
# We calculate the multiplication values based on the arrow names to ensure associativity.
# Any arrow f_X_Y * f_Y_Z must equal C * f_X_Z or C * e_X
function mult(x::Symbol, y::Symbol)
    sx, sy = String(x), String(y)
    
    # 1. Identity Logic (e_A * f_AB = f_AB)
    if startswith(sx, "e_")
        n = sx[3:end]
        if startswith(sy, "f_") && split(sy, "_")[2] == n; return (1.0, y); end
        if sy == sx; return (1.0, x); end
    elseif startswith(sy, "e_")
        n = sy[3:end]
        if startswith(sx, "f_") && split(sx, "_")[3] == n; return (1.0, x); end
    end

    # 2. Path Composition Logic
    if haskey(raw_coeffs, (x, y))
        val = raw_coeffs[(x, y)] # Extract the Float64
        
        # Determine the target symbol:
        # If f_A_B * f_B_C, target is f_A_C. If f_A_B * f_B_A, target is e_A.
        px, py = split(sx, "_"), split(sy, "_")
        target = px[2] == py[3] ? Symbol("e_$(px[2])") : Symbol("f_$(px[2])_$(py[3])")
        
        if haskey(b_index, target)
            return (val, target)
        end
    end

    # 3. Default Path Algebra (weight 1.0) if not in raw_coeffs
    if startswith(sx, "f_") && startswith(sy, "f_")
        px, py = split(sx, "_"), split(sy, "_")
        if px[3] == py[2] # Composability check
            target = px[2] == py[3] ? Symbol("e_$(px[2])") : Symbol("f_$(px[2])_$(py[3])")
            if haskey(b_index, target)
                return (1.0, target)
            end
        end
    end

    return (0.0, nothing)
end

# =========================
# 3. Matrices & HH2 (Sparse Rank)
# =========================
function build_d1()
    nC1, nC2 = dimA^2, dimA^3
    mat = spzeros(T, nC2, nC1)
    for a in 1:dimA, b in 1:dimA, k in 1:dimA
        row = (a-1)*dimA^2 + (b-1)*dimA + k
        for i in 1:dimA
            # Term: a * phi(b)
            v1, s1 = mult(basis[a], basis[i])
            if s1 == basis[k]; mat[row, (b-1)*dimA + i] += v1; end
            # Term: -phi(ab)
            v_ab, s_ab = mult(basis[a], basis[b])
            if s_ab == basis[i] && i == k; mat[row, (b_index[s_ab]-1)*dimA + k] -= v_ab; end
            # Term: phi(a) * b
            v3, s3 = mult(basis[i], basis[b])
            if s3 == basis[k]; mat[row, (a-1)*dimA + i] += v3; end
        end
    end
    return mat
end

function build_d2()
    nC2, nC3 = dimA^3, dimA^4
    mat = spzeros(T, nC3, nC2)
    for a in 1:dimA, b in 1:dimA, c in 1:dimA, k in 1:dimA
        row = (a-1)*dimA^3 + (b-1)*dimA^2 + (c-1)*dimA + k
        for i in 1:dimA
            v1, s1 = mult(basis[a], basis[i])
            if s1 == basis[k]; mat[row, (b-1)*dimA^2 + (c-1)*dimA + i] += v1; end
            v_ab, s_ab = mult(basis[a], basis[b])
            if s_ab !== nothing; mat[row, (b_index[s_ab]-1)*dimA^2 + (c-1)*dimA + k] -= v_ab; end
            v_bc, s_bc = mult(basis[b], basis[c])
            if s_bc !== nothing; mat[row, (a-1)*dimA^2 + (b_index[s_bc]-1)*dimA + k] += v_bc; end
            v4, s4 = mult(basis[i], basis[c])
            if s4 == basis[k]; mat[row, (a-1)*dimA^2 + (b-1)*dimA + i] -= v4; end
        end
    end
    return mat
end

# Where does associativity breaks?

function find_breaks(basis_list::Vector{Symbol}, multiplication_function::Function)
    breaks = []
    println("Scanning $(length(basis_list)^3) triplets for associativity breaks...")
    
    for i in basis_list, j in basis_list, k in basis_list
        # Path A: (i * j) * k
        v1, s1 = multiplication_function(i, j)
        v_left, s_left = (s1 === nothing) ? (0.0, nothing) : multiplication_function(s1, k)
        val_left = v1 * v_left

        # Path B: i * (j * k)
        v2, s2 = multiplication_function(j, k)
        v_right, s_right = (s2 === nothing) ? (0.0, nothing) : multiplication_function(i, s2)
        val_right = v2 * v_right

        # If the target symbols don't match or the scalars are different
        if s_left != s_right || !isapprox(val_left, val_right, atol=1e-5)
            # Only record if at least one path is non-zero (to ignore trivial 0=0)
            if val_left != 0 || val_right != 0
                push!(breaks, (i, j, k, s_left, val_left, s_right, val_right))
            end
        end
    end
    return breaks
end

# A dictionary to hold local "healing" factors for specific manual edges
local_corrections = Dict{Tuple{Symbol, Symbol}, Float64}()

# Example: If (f_HY_sAMY, f_sAMY_BLA, f_BLA_sAMY) is the only break
# You calculate the ratio needed to make Left == Right and store it:
local_corrections[(:f_sAMY, :BLA)] = 66.0 / 72.0 

function mult_selective(x, y)
    val, sym = mult_original(x, y) # Your raw voxel-derived multiplication
    
    # Apply the surgical correction if this edge is a known disruptor
    if haskey(local_corrections, (x, y))
        val *= local_corrections[(x, y)]
    end
    
    return val, sym
end

results = find_breaks(basis, mult)
for b in results
    println("Break at ($(b[1]), $(b[2]), $(b[3])): Left -> $(b[4]) ($(b[5])), Right -> $(b[6]) ($(b[7]))")
end

# Execution
println("Algebra Dimension: $dimA")
d1 = build_d1()
d2 = build_d2()

# Check condition
diff = d2 * d1
n_val = norm(diff)
println("||d2 * d1|| = $n_val")

if n_val < 1e-5
    r2 = rank(qr(d2).R) # Sparse-safe rank
    r1 = rank(qr(d1).R)
    println("HH2 Dimension: $((size(d2,2) - r2) - r1)")
else
    # Find specific non-associative triplet
    for i=1:dimA, j=1:dimA, k=1:dimA
        v1, s1 = mult(basis[i], basis[j])
        v2, s2 = s1 === nothing ? (0.0, nothing) : mult(s1, basis[k])
        
        v3, s3 = mult(basis[j], basis[k])
        v4, s4 = s3 === nothing ? (0.0, nothing) : mult(basis[i], s3)
        
        if s2 != s4 || !isapprox(v1*v2, v3*v4)
            println("Associativity break: ($(basis[i]), $(basis[j]), $(basis[k]))")
            break
        end
    end
end
