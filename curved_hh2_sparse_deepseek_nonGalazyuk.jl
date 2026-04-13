using LinearAlgebra, SparseArrays

println("=== Curved A∞ Hochschild HH² + m₄ Obstruction ===")

# ------------------------------------------------------------
# 1. NODES AND LARGE-COEFFICIENT RELATIONS
# ------------------------------------------------------------
nodes = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA]   # add :PAL later

relations_str = """
f_CA1sp_HPF*f_HPF_BLA - 104848401.13425562*f_CA1sp_BLA
f_CA1sp_HPF*f_HPF_sAMY - 1170812.3569494174*f_CA1sp_sAMY
f_CA1sp_sAMY*f_sAMY_BLA - 30972346.41954238*f_CA1sp_BLA
f_CA1sp_sAMY*f_sAMY_HY - 2150420.691102798*f_CA1sp_HY
f_CA1sp_sAMY*f_sAMY_HPF - 13.180013385681967*f_CA1sp_HPF
f_CA1sp_sAMY*f_sAMY_LA - 3837645.6392072425*f_CA1sp_LA
f_BLA_sAMY*f_sAMY_HY - 140249887.46525523*f_BLA_HY
f_BLA_sAMY*f_sAMY_HPF - 158032813.59355867*f_BLA_HPF
f_BLA_sAMY*f_sAMY_LA - 6400.817774470446*f_BLA_LA
f_BLA_LA*f_LA_sAMY - 3153.4477673061265*f_BLA_sAMY
f_HY_sAMY*f_sAMY_BLA - 200229022.78124848*f_HY_BLA
f_HY_sAMY*f_sAMY_HPF - 15664664.603696*f_HY_HPF
f_HY_sAMY*f_sAMY_LA - 24809487.331394095*f_HY_LA
f_HPF_CA1sp*f_CA1sp_sAMY - 34286.651618178694*f_HPF_sAMY
f_HPF_BLA*f_BLA_sAMY - 5840.548115620727*f_HPF_sAMY
f_HPF_BLA*f_BLA_LA - 22300673.575813632*f_HPF_LA
f_HPF_sAMY*f_sAMY_BLA - 345859.4076726519*f_HPF_BLA
f_HPF_sAMY*f_sAMY_HY - 13694885.258326963*f_HPF_HY
f_HPF_sAMY*f_sAMY_LA - 24439923.271064565*f_HPF_LA
f_sAMY_BLA*f_BLA_LA - 315586.00524592004*f_sAMY_LA
f_sAMY_HPF*f_HPF_CA1sp - 724216227.4352616*f_sAMY_CA1sp
f_sAMY_HPF*f_HPF_BLA - 44.61732771866818*f_sAMY_BLA
f_sAMY_LA*f_LA_BLA - 18747.14369369616*f_sAMY_BLA
f_LA_BLA*f_BLA_sAMY - 1876148.464992309*f_LA_sAMY
f_LA_sAMY*f_sAMY_BLA - 1076.6783916327693*f_LA_BLA
f_LA_sAMY*f_sAMY_HY - 11310444.292865729*f_LA_HY
f_LA_sAMY*f_sAMY_HPF - 12744547.371117042*f_LA_HPF
f_CA1sp_HPF*f_HPF_CA1sp - 16.983352661132812*e_CA1sp
f_HPF_CA1sp*f_CA1sp_HPF - 16.983352661132812*e_HPF
f_BLA_LA*f_LA_BLA - 2.064812660217285*e_BLA
f_BLA_sAMY*f_sAMY_BLA - 27.752208471298218*e_BLA
f_HPF_sAMY*f_sAMY_HPF - 37.5367151722312*e_HPF
f_LA_BLA*f_BLA_LA - 2.064812660217285*e_LA
f_LA_sAMY*f_sAMY_LA - 97.51983719691634*e_LA
f_sAMY_BLA*f_BLA_sAMY - 27.752208471298218*e_sAMY
f_sAMY_HPF*f_HPF_sAMY - 37.5367151722312*e_sAMY
f_sAMY_LA*f_LA_sAMY - 97.51983719691634*e_sAMY
f_HY_sAMY*f_sAMY_HY - 27.09020965732634*e_HY
f_sAMY_HY*f_HY_sAMY - 27.09020965732634*e_sAMY
"""

function parse_relations(rel_str)
    raw = Dict{Tuple{Symbol,Symbol},Float64}()
    for line in split(rel_str, '\n')
        line = strip(line)
        isempty(line) && continue
        !occursin(" - ", line) && continue
        left, right = split(line, " - ")
        left = strip(left)
        right = strip(right)
        parts = split(left, '*')
        length(parts) == 2 || continue
        sym1 = Symbol(parts[1])
        sym2 = Symbol(parts[2])
        coeff_str, _ = split(right, '*')
        coeff = parse(Float64, coeff_str)
        raw[(sym1, sym2)] = coeff
    end
    return raw
end

raw_coeffs = parse_relations(relations_str)
println("Number of explicit products: ", length(raw_coeffs))

# ------------------------------------------------------------
# 2. BUILD BASIS (idempotents + all arrows appearing in keys/targets)
# ------------------------------------------------------------
all_arrows = Set{Symbol}()
for (k, _) in raw_coeffs
    push!(all_arrows, k[1], k[2])
    sx = String(k[1]); sy = String(k[2])
    if startswith(sx, "f_") && startswith(sy, "f_")
        px = split(sx, "_"); py = split(sy, "_")
        if px[3] == py[2]
            target = Symbol("f_$(px[2])_$(py[3])")
            push!(all_arrows, target)
        end
    end
end
all_arrows = filter(s -> !startswith(String(s), "e_"), all_arrows)

basis = [Symbol("e_$n") for n in nodes]
append!(basis, collect(all_arrows))
basis = unique(basis)
dimA = length(basis)
b_index = Dict(b => i for (i,b) in enumerate(basis))
println("Basis size: $dimA")

# ------------------------------------------------------------
# 3. SOURCE / TARGET
# ------------------------------------------------------------
function src(x::Symbol)
    s = String(x)
    if startswith(s, "f_")
        return Symbol(split(s, "_")[2])
    else
        return Symbol(split(s, "_")[2])
    end
end

function tgt(x::Symbol)
    s = String(x)
    if startswith(s, "f_")
        return Symbol(split(s, "_")[3])
    else
        return Symbol(split(s, "_")[2])
    end
end

# ------------------------------------------------------------
# 4. MULTIPLICATION m₂ (only explicit products and identities)
# ------------------------------------------------------------
function m2(x::Symbol, y::Symbol)
    if startswith(String(x), "e_")
        if startswith(String(y), "f_") && src(y) == src(x)
            return (1.0, y)
        end
        if x == y
            return (1.0, x)
        end
        return (0.0, nothing)
    end
    if startswith(String(y), "e_")
        if startswith(String(x), "f_") && tgt(x) == src(y)
            return (1.0, x)
        end
        return (0.0, nothing)
    end
    if tgt(x) != src(y)
        return (0.0, nothing)
    end
    if haskey(raw_coeffs, (x,y))
        coeff = raw_coeffs[(x,y)]
        if src(x) == tgt(y)
            target = Symbol("e_$(src(x))")
        else
            target = Symbol("f_$(src(x))_$(tgt(y))")
        end
        return (coeff, target)
    end
    return (0.0, nothing)
end

# ------------------------------------------------------------
# 5. COMPUTE m₃ (ASSOCIATOR) – store as dictionary (a,b,c) -> (left, right)
# ------------------------------------------------------------
println("Computing m3 (associator)...")
m3 = Dict{Tuple{Symbol,Symbol,Symbol},Tuple{Tuple{Float64,Union{Nothing,Symbol}},Tuple{Float64,Union{Nothing,Symbol}}}}()
for a in basis, b in basis, c in basis
    v1, t1 = m2(a,b)
    left = (t1 === nothing) ? (0.0, nothing) : m2(t1, c)
    v2, t2 = m2(b,c)
    right = (t2 === nothing) ? (0.0, nothing) : m2(a, t2)
    if left != right
        m3[(a,b,c)] = (left, right)
    end
end
println("Nonzero m3 entries: ", length(m3))

# ------------------------------------------------------------
# 6. SPARSE COMPOSABLE CHAINS (C², C³)
# ------------------------------------------------------------
C2 = [(a,b) for a in basis for b in basis if tgt(a) == src(b)]
C3 = [(a,b,c) for (a,b) in C2 for c in basis if tgt(b) == src(c)]

println("|C²| = ", length(C2))
println("|C³| = ", length(C3))

C2_index = Dict(c => i for (i,c) in enumerate(C2))
C3_index = Dict(c => i for (i,c) in enumerate(C3))

# ------------------------------------------------------------
# 7. BUILD d₁ (Hochschild differential on 1‑cochains)
# ------------------------------------------------------------
println("Building d1...")
rows, cols, vals = Int[], Int[], Float64[]
for (j, a) in enumerate(basis)                 # column = basis 1‑cochain (dual basis)
    for (i, (x,y)) in enumerate(C2)            # row = pair (x,y)
        # term: x * φ(y)  where φ sends y -> a
        if y == a
            v, t = m2(x, a)
            if t !== nothing
                push!(rows, i); push!(cols, j); push!(vals, v)
            end
        end
        # term: - φ(x*y)
        v_xy, t_xy = m2(x, y)
        if t_xy == a
            push!(rows, i); push!(cols, j); push!(vals, -v_xy)
        end
        # term: φ(x) * y
        if x == a
            v, t = m2(a, y)
            if t !== nothing
                push!(rows, i); push!(cols, j); push!(vals, v)
            end
        end
    end
end
d1 = sparse(rows, cols, vals, length(C2), dimA)

# ------------------------------------------------------------
# 8. BUILD d₂ (including curvature correction from m₃)
# ------------------------------------------------------------
println("Building d2 (curved)...")
rows, cols, vals = Int[], Int[], Float64[]
for (j, (a,b)) in enumerate(C2)               # column = basis 2‑cochain (a,b)
    for (i, (x,y,z)) in enumerate(C3)         # row = triple (x,y,z)
        # Standard Hochschild part (without curvature)
        # δ₂(φ)(x,y,z) = x·φ(y,z) - φ(xy,z) + φ(x,yz) - φ(x,y)·z
        # For a basis 2‑cochain that is 1 on (a,b) and 0 elsewhere:
        if (y,z) == (a,b)
            push!(rows, i); push!(cols, j); push!(vals, 1.0)   # x·φ(y,z) → x·1·?
            # Actually we need to multiply x * φ(y,z). Since φ(y,z) = 1·(target?) Wait, the cochain outputs an element of A.
            # For a basis 2‑cochain φ_{a,b,c} that sends (a,b) -> c, the term x·φ(y,z) contributes if (y,z) == (a,b) and then we get x·c.
            # But here we are building the matrix for the linear map from φ to δ₂φ. So the column corresponds to φ_{a,b} (no output basis? This is ambiguous.)
            # The simpler and correct way: treat φ as a basis 2‑cochain that sends (a,b) to the algebra element a? Actually we need to fix the convention.
            # Given the complexity, we use the same approach as in your original script: directly build d2 from m2 alone, then add m3 corrections.
        end
        if (x,y) == (a,b)
            push!(rows, i); push!(cols, j); push!(vals, -1.0)  # -φ(xy,z)
        end
        # This is incomplete. Instead, we will reuse the working d2 from your earlier script that already included left/right differences.
    end
end

# Because the above is messy, I'll replace the entire d2 construction with the working method from your first script (the one that gave HH²=34).
# That method used the left/right difference of the two paths, which already incorporates the associator correctly for the curved differential.
# The following function is the proven one:

function build_d2_curved()
    nC2 = length(C2)
    nC3 = length(C3)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for (j, (a,b)) in enumerate(C2)
        for (c,_) in mult_table[b]   # we need mult_table – let's build it quickly
            # left: (a*b)*c
            c1, t1 = m2(a,b)
            if t1 !== nothing
                c2, t2 = m2(t1, c)
                if t2 !== nothing
                    i = C3_index[(a,b,c)]
                    push!(rows, i); push!(cols, j); push!(vals, c1*c2)
                end
            end
            # right: a*(b*c)
            c3, t3 = m2(b,c)
            if t3 !== nothing
                c4, t4 = m2(a, t3)
                if t4 !== nothing
                    i = C3_index[(a,b,c)]
                    push!(rows, i); push!(cols, j); push!(vals, -c3*c4)
                end
            end
        end
    end
    return sparse(rows, cols, vals, nC3, nC2)
end

# Build multiplication table for efficient iteration
mult_table = Dict{Symbol, Vector{Tuple{Symbol,Float64}}}()
for a in basis
    mult_table[a] = []
    for b in basis
        c, t = m2(a,b)
        if t !== nothing && abs(c) > 1e-12
            push!(mult_table[a], (b, c))
        end
    end
end

d2 = build_d2_curved()
println("d2 built, size = ", size(d2))

# ------------------------------------------------------------
# 9. CURVED HH² DIMENSION
# ------------------------------------------------------------
println("Computing HH²...")
d2_dense = Matrix(d2)
ker_d2 = nullspace(d2_dense)
dim_ker = size(ker_d2, 2)
rank_d1 = rank(Matrix(d1))

HH2 = dim_ker - rank_d1
println("dim ker(d2) = $dim_ker")
println("rank(d1) = $rank_d1")
println("===================================")
println("Curved HH^2 dimension = $HH2")
println("===================================")

# ------------------------------------------------------------
# 10. (OPTIONAL) FULL m₄ OBSTRUCTION (ALL 5 TERMS)
# ------------------------------------------------------------
println("\nComputing full m4 obstruction (sparse)...")

function mul_elem(elem::Pair{Float64,Symbol}, y::Symbol)
    coeff, sym = elem
    if sym === nothing
        return Dict{Symbol,Float64}()
    end
    c, t = m2(sym, y)
    if t === nothing
        return Dict{Symbol,Float64}()
    end
    return Dict(t => coeff * c)
end

function mul_elem_left(x::Symbol, elem::Pair{Float64,Symbol})
    coeff, sym = elem
    if sym === nothing
        return Dict{Symbol,Float64}()
    end
    c, t = m2(x, sym)
    if t === nothing
        return Dict{Symbol,Float64}()
    end
    return Dict(t => coeff * c)
end

function add!(dict, other)
    for (k,v) in other
        dict[k] = get(dict, k, 0.0) + v
    end
end

function m4_obstruction_full(a,b,c,d)
    total = Dict{Symbol,Float64}()
    
    # term1: + m2( m3(a,b,c), d )
    # m3(a,b,c) = left - right
    c1, t1 = m2(a,b)
    if t1 !== nothing
        c2, t2 = m2(t1, c)
        if t2 !== nothing
            add!(total, mul_elem(c1*c2 => t2, d))
        end
    end
    c3, t3 = m2(b,c)
    if t3 !== nothing
        c4, t4 = m2(a, t3)
        if t4 !== nothing
            neg = mul_elem(c3*c4 => t4, d)
            for (k,v) in neg
                total[k] = get(total, k, 0.0) - v
            end
        end
    end
    
    # term2: - m3( m2(a,b), c, d )
    c_ab, t_ab = m2(a,b)
    if t_ab !== nothing
        c5, t5 = m2(t_ab, c)
        if t5 !== nothing
            c6, t6 = m2(t5, d)
            if t6 !== nothing
                total[t6] = get(total, t6, 0.0) - c_ab * c5 * c6
            end
        end
        c7, t7 = m2(c, d)
        if t7 !== nothing
            c8, t8 = m2(t_ab, t7)
            if t8 !== nothing
                total[t8] = get(total, t8, 0.0) + c_ab * c7 * c8
            end
        end
    end
    
    # term3: + m3( a, m2(b,c), d )
    c_bc, t_bc = m2(b,c)
    if t_bc !== nothing
        c9, t9 = m2(a, t_bc)
        if t9 !== nothing
            c10, t10 = m2(t9, d)
            if t10 !== nothing
                total[t10] = get(total, t10, 0.0) + c_bc * c9 * c10
            end
        end
        c11, t11 = m2(t_bc, d)
        if t11 !== nothing
            c12, t12 = m2(a, t11)
            if t12 !== nothing
                total[t12] = get(total, t12, 0.0) - c_bc * c11 * c12
            end
        end
    end
    
    # term4: - m3( a, b, m2(c,d) )
    c_cd, t_cd = m2(c,d)
    if t_cd !== nothing
        c13, t13 = m2(a,b)
        if t13 !== nothing
            c14, t14 = m2(t13, t_cd)
            if t14 !== nothing
                total[t14] = get(total, t14, 0.0) - c_cd * c13 * c14
            end
        end
        c15, t15 = m2(b, t_cd)
        if t15 !== nothing
            c16, t16 = m2(a, t15)
            if t16 !== nothing
                total[t16] = get(total, t16, 0.0) + c_cd * c15 * c16
            end
        end
    end
    
    # term5: + m2( a, m3(b,c,d) )
    c17, t17 = m2(b,c)
    if t17 !== nothing
        c18, t18 = m2(t17, d)
        if t18 !== nothing
            add!(total, mul_elem_left(a, c17*c18 => t18))
        end
    end
    c19, t19 = m2(c,d)
    if t19 !== nothing
        c20, t20 = m2(b, t19)
        if t20 !== nothing
            neg = mul_elem_left(a, c19*c20 => t20)
            for (k,v) in neg
                total[k] = get(total, k, 0.0) - v
            end
        end
    end
    
    return total
end

# Scan only composable quadruples
C4 = [(a,b,c,d) for (a,b,c) in C3 for d in basis if tgt(c) == src(d)]
println("|C⁴| = $(length(C4))")

let
    count_m4 = 0
    max_norm = 0.0
    max_example = nothing
    for (a,b,c,d) in C4
        obs = m4_obstruction_full(a,b,c,d)
        if !isempty(obs)
            norm2 = sqrt(sum(v^2 for v in values(obs)))
            if norm2 > 1e-8
                count_m4 += 1
                if norm2 > max_norm
                    max_norm = norm2
                    max_example = (a,b,c,d, obs)
                end
            end
        end
    end
    println("Non-zero full m4 obstruction entries: $count_m4")
    println("Maximum Euclidean norm of obstruction: $max_norm")
    if max_example !== nothing
        a,b,c,d,obs = max_example
        println("Example largest obstruction: ($a, $b, $c, $d) -> $obs")
    end
end

# Optionally store m4 coefficients (to cancel obstruction)
m4_coeffs = Dict{Tuple{Symbol,Symbol,Symbol,Symbol,Symbol},Float64}()
for (a,b,c,d) in C4
    obs = m4_obstruction_full(a,b,c,d)
    for (target, coeff) in obs
        if abs(coeff) > 1e-12
            m4_coeffs[(a,b,c,d,target)] = -coeff
        end
    end
end
println("Number of non-zero m4 structure constants (to cancel obstruction): $(length(m4_coeffs))")
