using LinearAlgebra, SparseArrays
using WriteVTK
using WriteVTK.VTKCellTypes: VTK_LINE
using DataFrames

println("=== Curved A∞ Hochschild HH² + m₄ Obstruction ===")
NODES = "./node_regions_clean.csv"
EDGES = "/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv"

# ------------------------------------------------------------
# 1. NODES AND LARGE-COEFFICIENT RELATIONS
# ------------------------------------------------------------
nodes = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA]   # add :PAL later
#nodes = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :PAL, :LSX, :RHP]

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
# Build m4_obs dictionary (quadruple -> dict of target -> coefficient)
m4_obs = Dict{Tuple{Symbol,Symbol,Symbol,Symbol}, Dict{Symbol,Float64}}()
for (a,b,c,d) in C4
    obs = m4_obstruction_full(a,b,c,d)
    if !isempty(obs)
        m4_obs[(a,b,c,d)] = obs
    end
end
println("Number of non-zero m4 obstructions (quadruples): ", length(m4_obs))
# ------------------------------------------------------------
# Region extraction helpers (tailored to your symbol naming)
# ------------------------------------------------------------
function extract_regions(sym::Symbol)
    s = String(sym)
    if startswith(s, "f_")
        parts = split(s, "_")
        # f_X_Y -> (X, Y)
        return (Symbol(parts[2]), Symbol(parts[3]))
    elseif startswith(s, "e_")
        # e_X -> (X,)
        return (Symbol(s[3:end]),)
    end
    return ()
end

function regions_in_tuple(tup)
    regs = Set{Symbol}()
    for x in tup
        for r in extract_regions(x)
            push!(regs, r)
        end
    end
    return regs
end

function compute_region_scores(m3_dict, m4_obs; α=1.0, β=2.0)
    score = Dict{Symbol, Float64}()
    
    # m3 contribution: each associator entry (a,b,c) -> (left, right)
    for (triple, (left, right)) in m3_dict
        regs = regions_in_tuple(triple)
        # weight = sum of absolute coefficients from left and right results
        weight = 0.0
        for (coeff, target) in (left, right)
            if target !== nothing
                weight += abs(coeff)
            end
        end
        for r in regs
            score[r] = get(score, r, 0.0) + α * weight
        end
    end
    
    # m4 contribution: each quadruple (a,b,c,d) -> dict of target -> coefficient
    for (quad, obs_dict) in m4_obs
        regs = regions_in_tuple(quad)
        weight = sum(abs(v) for v in values(obs_dict))
        for r in regs
            score[r] = get(score, r, 0.0) + β * weight
        end
    end
    
    return sort(collect(score), by=x->-x[2])
end

# ------------------------------------------------------------
# Build m4_obs dictionary for region scoring
# ------------------------------------------------------------
m4_obs = Dict{Tuple{Symbol,Symbol,Symbol,Symbol}, Dict{Symbol,Float64}}()
for (a,b,c,d) in C4
    obs = m4_obstruction_full(a,b,c,d)
    if !isempty(obs)
        m4_obs[(a,b,c,d)] = obs
    end
end

# ------------------------------------------------------------
# Compute region scores from m3 and m4
# ------------------------------------------------------------
println("\n--- Region scores (α=1, β=2) ---")
region_scores = compute_region_scores(m3, m4_obs; α=1.0, β=2.0)
for (reg, scr) in region_scores
    println("$reg : $scr")
end


function top_contributors(m3_dict, m4_obs; α=1.0, β=2.0, top_n=5)
    region_contrib = Dict{Symbol, Vector{Tuple{Any, Float64}}}()  # region -> list of (tuple, weight)
    
    # m3 contributions
    for (triple, (left, right)) in m3_dict
        weight = 0.0
        for (coeff, target) in (left, right)
            if target !== nothing
                weight += abs(coeff)
            end
        end
        regs = regions_in_tuple(triple)
        for r in regs
            push!(get!(region_contrib, r, []), (triple, α * weight))
        end
    end
    
    # m4 contributions
    for (quad, obs_dict) in m4_obs
        weight = sum(abs(v) for v in values(obs_dict))
        regs = regions_in_tuple(quad)
        for r in regs
            push!(get!(region_contrib, r, []), (quad, β * weight))
        end
    end
    
    # Sort and keep top_n per region
    top = Dict{Symbol, Vector{Tuple{Any, Float64}}}()
    for (r, contribs) in region_contrib
        sort!(contribs, by=x->x[2], rev=true)
        top[r] = contribs[1:min(top_n, end)]
    end
    return top
end

top_paths = top_contributors(m3, m4_obs; α=1.0, β=2.0, top_n=3)
println("\n--- Top 3 paths per region ---")
for (reg, paths) in sort(collect(top_paths), by=x->region_scores[findfirst(==(x[1]), first.(region_scores))][2], rev=true)
    println("$reg:")
    for (tup, w) in paths
        println("   $tup : $w")
    end
end


function top_contributors_topN(m3_dict, m4_obs; α=1.0, β=2.0, top_n=4)
    # Accumulate contributions per region
    region_contrib = Dict{Symbol, Vector{Tuple{Any, Float64}}}()
    
    # Process m3 (triples)
    for (triple, (left, right)) in m3_dict
        weight = 0.0
        for (coeff, target) in (left, right)
            if target !== nothing
                weight += abs(coeff)
            end
        end
        regs = regions_in_tuple(triple)
        for r in regs
            push!(get!(region_contrib, r, []), (triple, α * weight))
        end
    end
    
    # Process m4 (quadruples)
    for (quad, obs_dict) in m4_obs
        weight = sum(abs(v) for v in values(obs_dict))
        regs = regions_in_tuple(quad)
        for r in regs
            push!(get!(region_contrib, r, []), (quad, β * weight))
        end
    end
    
    # Keep only top_n per region (sorted by weight descending)
    top = Dict{Symbol, Vector{Tuple{Any, Float64}}}()
    for (r, contribs) in region_contrib
        sort!(contribs, by=x->x[2], rev=true)
        top[r] = contribs[1:min(top_n, end)]
    end
    return top
end

# After m4_obs is built
top_paths_N = top_contributors_topN(m3, m4_obs; α=1.0, β=2.0, top_n=4)

println("\n--- Top 4 paths per region ---")
for (reg, paths) in sort(collect(top_paths_N), by=x->region_scores[findfirst(==(x[1]), first.(region_scores))][2], rev=true)
    println("$reg:")
    for (tup, w) in paths
        # Pretty print: show tuple and weight
        println("   $tup : $w")
    end
end

function top_m4_obstructions(m4_obs; β=2.0, top_n=5)
    # Compute weight per quadruple: β * sum of absolute coefficients
    weighted = [(quad, β * sum(abs(v) for v in values(obs_dict))) for (quad, obs_dict) in m4_obs]
    sort!(weighted, by=x->x[2], rev=true)
    println("\n--- Top $top_n m₄ obstructions (by weight) ---")
    for (quad, w) in weighted[1:min(top_n, end)]
        println("$quad : $w")
        # Optionally show the internal dictionary
        println("   -> $(m4_obs[quad])")
    end
end

top_m4_obstructions(m4_obs; β=2.0, top_n=5)
# investigate paths
target_quad = (:f_BLA_LA, :f_LA_BLA, :f_BLA_sAMY, :f_sAMY_HPF)
if haskey(m4_obs, target_quad)
    println("Obstruction for $target_quad: ", m4_obs[target_quad])
end

m4_only_scores = compute_region_scores(Dict(), m4_obs; α=0.0, β=2.0)
println("\n--- Region scores from m₄ only ---")
for (reg, scr) in m4_only_scores
    println("$reg : $scr")
end


# ------------------------------------------------------------
# Local stalk and restriction helpers
# ------------------------------------------------------------
function build_local_stalk(basis, target_region::Symbol; depth=2)
    local_basis = Set{Symbol}()
    seen_regions = Set{Symbol}([target_region])
    
    # initial collection
    for b in basis
        if target_region in extract_regions(b)
            push!(local_basis, b)
            for r in extract_regions(b)
                push!(seen_regions, r)
            end
        end
    end
    
    # expand neighborhood by region adjacency
    for _ in 1:depth
        added = false
        for b in basis
            b in local_basis && continue
            regs = extract_regions(b)
            if any(r -> r in seen_regions, regs)
                push!(local_basis, b)
                for r in regs
                    push!(seen_regions, r)
                end
                added = true
            end
        end
        !added && break
    end
    
    return collect(local_basis)
end

# Now local_stalk uses the keyword argument correctly
function local_stalk(basis, region_name::String, depth::Int)
    region_sym = Symbol(region_name)
    return build_local_stalk(basis, region_sym; depth=depth)
end

function restrict_dict(dict, local_basis)
    # dict can be m3 (keys are triples) or m4_obs (keys are quadruples)
    return Dict(
        k => v for (k,v) in dict
        if all(x -> x in local_basis, k)
    )
end

function build_local_m5(m4_obs_local)
    m5 = Dict{Tuple{Symbol,Symbol,Symbol,Symbol,Symbol}, Float64}()
    for (quad, valdict) in m4_obs_local
        for (target, val) in valdict
            if abs(val) > 1e-12
                m5[(quad..., target)] = -val
            end
        end
    end
    return m5
end

function score_paths(m3, m4, m5)
    scores = Dict{Any, Float64}()
    for (k, v) in m3
        # v is (left, right) tuple; weight = |left| + |right|
        weight = 0.0
        for (coeff, target) in (v[1], v[2])
            if target !== nothing
                weight += abs(coeff)
            end
        end
        scores[k] = get(scores, k, 0.0) + weight
    end
    for (k, v) in m4
        # v is dict of target -> coeff
        weight = sum(abs, values(v))
        scores[k] = get(scores, k, 0.0) + weight
    end
    for (k, v) in m5
        # v is a scalar coefficient
        scores[k] = get(scores, k, 0.0) + abs(v)
    end
    return sort(collect(scores), by=x->-x[2])
end

normalize_path(p) = p isa Symbol ? (p,) :
                    p isa AbstractVector{Symbol} ? Tuple(p) :
                    p isa Tuple{Vararg{Symbol}} ? p :
                    error("Invalid path type: $(typeof(p))")

# Choose a region of interest, e.g., :sAMY
target = :sAMY
local_basis = build_local_stalk(basis, target, depth=2)
println("\n--- Local stalk for $target (depth=2) ---")
println("Size: $(length(local_basis))")
println("First 10 elements: $(local_basis[1:min(10, end)])")

# Restrict m3 and m4_obs to this stalk
m3_local = restrict_dict(m3, local_basis)
m4_local = restrict_dict(m4_obs, local_basis)
println("Restricted m3 entries: $(length(m3_local))")
println("Restricted m4 entries: $(length(m4_local))")

# Build local m5 to cancel obstruction
m5_local = build_local_m5(m4_local)
println("Local m5 entries (to cancel obstruction): $(length(m5_local))")

# Score paths inside the stalk
local_path_scores = score_paths(m3_local, m4_local, m5_local)
println("\n--- Top 10 path scores within $target stalk ---")
for (path, scr) in local_path_scores[1:min(10, end)]
    println("$path : $scr")
end

# ------------------------------------------------------------
# Convert m2 to dict format
# ------------------------------------------------------------
function mult_dict(x::Symbol, y::Symbol)
    c, t = m2(x, y)
    if t === nothing || abs(c) < 1e-12
        return Dict{Symbol,Float64}()
    else
        return Dict(t => c)
    end
end

# ------------------------------------------------------------
# Convert m3 associator (left, right) to a single element: left - right
# ------------------------------------------------------------
m3_element = Dict{Tuple{Symbol,Symbol,Symbol}, Dict{Symbol,Float64}}()
for (triple, (left, right)) in m3
    # left is (coeff, target), right is (coeff, target)
    diff = Dict{Symbol,Float64}()
    if left[2] !== nothing
        diff[left[2]] = get(diff, left[2], 0.0) + left[1]
    end
    if right[2] !== nothing
        diff[right[2]] = get(diff, right[2], 0.0) - right[1]
    end
    # filter near-zero
    for (k, v) in diff
        if abs(v) < 1e-12
            delete!(diff, k)
        end
    end
    if !isempty(diff)
        m3_element[triple] = diff
    end
end
println("Converted m3 to element form: $(length(m3_element)) entries")

# ------------------------------------------------------------
# Provided helper functions (adapted to use mult_dict)
# ------------------------------------------------------------
function add_dict!(A::Dict{Symbol,Float64}, B::Dict{Symbol,Float64}, scale=1.0)
    for (k,v) in B
        A[k] = get(A,k,0.0) + scale*v
    end
end

function src(x::Symbol)
    s = String(x)
    if startswith(s, "e_")
        return Symbol(s[3:end])
    else
        parts = split(s, "_")
        return Symbol(parts[2])
    end
end

function tgt(x::Symbol)
    s = String(x)
    if startswith(s, "e_")
        return Symbol(s[3:end])
    else
        parts = split(s, "_")
        return Symbol(parts[3])
    end
end

function is_composable(x::Symbol, y::Symbol)
    return tgt(x) == src(y)
end


function norm_dict(D::Dict{Symbol,Float64})
    isempty(D) && return 0.0
    sqrt(sum(v^2 for v in values(D)))
end

function check_Ainf_level4_consistent(basis, m4_corrected; tol=1e-6)
    println("Checking A∞ identity at level 4 (consistent with m4_obstruction_full)...")
    max_violation = 0.0
    violations = []
    # Only iterate over composable quadruples (C4) to save time
    for (a,b,c,d) in C4
        # Compute δm₃ using the exact same function that computed the obstruction
        total = m4_obstruction_full(a,b,c,d)   # this is δm₃ (terms 1-5)
        # Add the corrected m₄ (which should be -obstruction)
        if haskey(m4_corrected, (a,b,c,d))
            for (k, v) in m4_corrected[(a,b,c,d)]
                total[k] = get(total, k, 0.0) + v
            end
        end
        nrm = norm_dict(total)
        if nrm > tol
            push!(violations, ((a,b,c,d), nrm))
            max_violation = max(max_violation, nrm)
        end
    end
    println("Max violation (δm₃ + m₄_corrected): $max_violation")
    println("Number of violating quadruples: ", length(violations))
    return violations
end

function restrict_tensor(op, stalk)
    out = Dict()
    for (k,v) in op
        if all(x in stalk for x in k)
            out[k] = v
        end
    end
    return out
end

function build_sheaf(basis, mult, m3, m4, regions)
    sheaf = Dict()
    for r in regions
        println("\nBuilding stalk for $r")
        stalk = local_stalk(basis, r, 2)
        sheaf[r] = (
            basis = stalk,
            m3 = restrict_tensor(m3, stalk),
            m4 = restrict_tensor(m4, stalk)
        )
        println("Stalk size: ", length(stalk))
        println("m3 size: ", length(sheaf[r].m3))
        println("m4 size: ", length(sheaf[r].m4))
    end
    return sheaf
end

function check_gluing(sheaf, r1, r2)
    B1 = Set(sheaf[r1].basis)
    B2 = Set(sheaf[r2].basis)
    inter = intersect(B1, B2)
    println("Intersection size: ", length(inter))
    mismatches = 0
    for k in keys(sheaf[r1].m3)
        if all(x in inter for x in k)
            if haskey(sheaf[r2].m3, k)
                d1 = sheaf[r1].m3[k]
                d2 = sheaf[r2].m3[k]
                if norm_dict(d1) != norm_dict(d2)
                    mismatches += 1
                end
            end
        end
    end
    println("m3 mismatches: ", mismatches)
end

function mult_expand(mult, A::Dict{Symbol,Float64}, b::Symbol)
    out = Dict{Symbol,Float64}()
    for (k, v) in A
        tmp = mult(k, b)
        for (kk, vv) in tmp
            out[kk] = get(out, kk, 0.0) + v * vv
        end
    end
    return out
end

function mult_expand_left(mult, a::Symbol, B::Dict{Symbol,Float64})
    out = Dict{Symbol,Float64}()
    for (k, v) in B
        tmp = mult(a, k)
        for (kk, vv) in tmp
            out[kk] = get(out, kk, 0.0) + v * vv
        end
    end
    return out
end

function mult_expand_left(f::Any, sym::Symbol, d::Dict{Any, Any})
    # Convert on the fly
    converted = Dict{Symbol, Float64}(key => float(value) for (key, value) in d if key isa Symbol)
    # Then call the real method
    mult_expand_left(f, sym, converted)
end


function build_m5_local(stalk, mult, m3, m4; tol=1e-6)
    m5 = Dict{Tuple{Symbol,Symbol,Symbol,Symbol,Symbol}, Dict{Symbol,Float64}}()
    
    for a in stalk, b in stalk, c in stalk, d in stalk, e in stalk
        total = Dict{Symbol,Float64}()
        
        # ----- δm4 terms (curvature of m4) -----
        # + m2(m4(a,b,c,d), e)
        if haskey(m4, (a,b,c,d))
            tmp = mult_expand(mult, m4[(a,b,c,d)], e)
            add_dict!(total, tmp, +1)
        end
        
        # - m4(m2(a,b), c, d, e)
        for (k, v) in mult(a,b)
            if haskey(m4, (k,c,d,e))
                add_dict!(total, m4[(k,c,d,e)], -v)
            end
        end
        
        # + m4(a, m2(b,c), d, e)
        for (k, v) in mult(b,c)
            if haskey(m4, (a,k,d,e))
                add_dict!(total, m4[(a,k,d,e)], +v)
            end
        end
        
        # - m4(a, b, m2(c,d), e)
        for (k, v) in mult(c,d)
            if haskey(m4, (a,b,k,e))
                add_dict!(total, m4[(a,b,k,e)], -v)
            end
        end
        
        # + m4(a, b, c, m2(d,e))
        for (k, v) in mult(d,e)
            if haskey(m4, (a,b,c,k))
                add_dict!(total, m4[(a,b,c,k)], +v)
            end
        end
        
        # ----- m3∘m3 terms (associator of associator) -----
        # + m3(m3(a,b,c), d, e)
        if haskey(m3, (a,b,c))
            tmp = mult_expand(mult, m3[(a,b,c)], d)   # m3(a,b,c) * d  (as algebra element)
            # then apply m3( ..., d, e ) – but careful: m3 expects three arguments, not an element.
            # The correct term is: m3( m3(a,b,c) , d, e ), where the first argument is the output of m3 (an algebra element).
            # We need a helper that applies m3 to an algebra element (dict) and two symbols.
            # Simpler: iterate over the support of m3(a,b,c)
            for (x, coeff) in m3[(a,b,c)]
                if haskey(m3, (x, d, e))
                    add_dict!(total, m3[(x,d,e)], coeff)
                end
            end
        end
        
        # - m3(a, m3(b,c,d), e)
        if haskey(m3, (b,c,d))
            for (x, coeff) in m3[(b,c,d)]
                if haskey(m3, (a, x, e))
                    add_dict!(total, m3[(a,x,e)], -coeff)
                end
            end
        end
        
        # + m3(a, b, m3(c,d,e))
        if haskey(m3, (c,d,e))
            for (x, coeff) in m3[(c,d,e)]
                if haskey(m3, (a, b, x))
                    add_dict!(total, m3[(a,b,x)], +coeff)
                end
            end
        end
        
        # + m2(a, m3(b,c,d,e)) – but we don't have m5 yet! This term will be used to solve for m5.
        # In the A∞ relation, the term involving m5 is: + m5(a,b,c,d,e) (no m2 wrapping).
        # Actually, the level‑5 relation is:
        #   δm4 + m3(m3) + m5 = 0
        # So we set m5 = - (δm4 + m3(m3)).
        
        # Therefore, total already contains δm4 + m3(m3). Then m5 = -total.
        
        nrm = norm_dict(total)
        if nrm > tol
            # m5 should cancel the remaining obstruction
            m5_cancel = Dict{Symbol,Float64}()
            for (k, v) in total
                m5_cancel[k] = -v
            end
            m5[(a,b,c,d,e)] = m5_cancel
        end
    end
    
    println("Number of non‑zero m5 entries: ", length(m5))
    return m5
end

# ------------------------------------------------------------
# Run checks
# ------------------------------------------------------------
# Now check A∞ with the corrected m4
# Build the corrected m4 (the actual multiplication map that cancels the obstruction)
m4_corrected = Dict{Tuple{Symbol,Symbol,Symbol,Symbol}, Dict{Symbol,Float64}}()
for (quad, obs_dict) in m4_obs
    m4_corrected[quad] = Dict(target => -coeff for (target, coeff) in obs_dict)
end

# Now check A∞ with the corrected m4
violations = check_Ainf_level4_consistent(basis, m4_corrected; tol=1e-6)

# Viloations are large so double cehck -- Verify that m4_corrected cancels the obstruction
function verify_m4_correction(m4_obs, m4_corrected, C4; tol=1e-12)
    println("\nVerifying that m4_corrected cancels the obstruction...")
    max_residual = 0.0
    for (a,b,c,d) in C4
        # Compute the δm₃ part (the obstruction)
        total = m4_obstruction_full(a,b,c,d)
        # Subtract the correction (which is the negative of the obstruction)
        if haskey(m4_corrected, (a,b,c,d))
            for (k, v) in m4_corrected[(a,b,c,d)]
                total[k] = get(total, k, 0.0) + v
            end
        end
        nrm = norm_dict(total)
        if nrm > max_residual
            max_residual = nrm
        end
    end
    println("Max residual after adding m4_corrected: $max_residual")
    if max_residual < tol
        println("✅ m4_corrected successfully cancels the obstruction (residual < $tol).")
    else
        println("⚠️ Residual is larger than tolerance; check consistency.")
    end
    return max_residual
end

# Then call it after building m4_corrected:
residual = verify_m4_correction(m4_obs, m4_corrected, C4)
sheaf = build_sheaf(basis, mult_dict, m3_element, m4_obs,
    ["sAMY","HPF","BLA","CA1sp","HY","LA"])

# Example: build m5 for the sAMY stalk
stalk_sAMY = sheaf["sAMY"].basis
m3_local = sheaf["sAMY"].m3
m4_local = sheaf["sAMY"].m4

m5_local = build_m5_local(stalk_sAMY, mult_dict, m3_local, m4_local; tol=1e-6)

# Print a few entries
println("\n--- Sample m5 entries (sAMY stalk) ---")
for (k, v) in collect(m5_local)[1:min(5, end)]
    println("$k => $(collect(v))")
end
check_gluing(sheaf, "sAMY", "HPF")


# Zero dict
const ZERO = Dict{Symbol,Float64}()

# Add two dicts
function add_dict!(A::Dict{Symbol,Float64}, B::Dict{Symbol,Float64}, scale=1.0)
    for (k,v) in B
        A[k] = get(A,k,0.0) + scale*v
    end
end

# Norm
norm_dict(A) = sqrt(sum(v^2 for v in values(A)))

# Safe multiplication (Symbol × Symbol → Dict)
function safe_mult(mult, x::Symbol, y::Symbol)
    if !is_composable(x,y)
        return ZERO
    end
    return mult(x,y)
end

# Expand left: a * (linear combo)
function mult_expand_left(mult, a::Symbol, B::Dict{Symbol,Float64})
    out = Dict{Symbol,Float64}()
    for (k,v) in B
        tmp = safe_mult(mult, a, k)
        add_dict!(out, tmp, v)
    end
    return out
end

# Expand right: (linear combo) * b
function mult_expand_right(mult, A::Dict{Symbol,Float64}, b::Symbol)
    out = Dict{Symbol,Float64}()
    for (k,v) in A
        tmp = safe_mult(mult, k, b)
        add_dict!(out, tmp, v)
    end
    return out
end

function mult_expand_right(f::Any, d::Dict{Any, Any}, sym::Symbol)
    # Convert on the fly: ensure keys are Symbol, values are Float64
    converted = Dict{Symbol, Float64}(key => float(value) for (key, value) in d if key isa Symbol)
    # Call the real method (expects Dict{Symbol, Float64})
    return mult_expand_right(f, converted, sym)
end
# ============================================================
# MISSING HELPER FUNCTIONS
# ============================================================

function merge_dicts!(target::Dict{Symbol,Float64}, source::Dict{Symbol,Float64}, scale=1.0)
    for (k,v) in source
        target[k] = get(target, k, 0.0) + scale * v
    end
    return target
end

function mult_expand_middle_4(mult, m4, a::Symbol, bc::Dict{Symbol,Float64}, d::Symbol, e::Symbol)
    # m4(a, bc, d, e) where bc is a linear combination (from m2(b,c))
    total = Dict{Symbol,Float64}()
    for (x, coeff) in bc
        if haskey(m4, (a, x, d, e))
            merge_dicts!(total, m4[(a, x, d, e)], coeff)
        end
    end
    return total
end

function compose_m3_m3(a,b,c,d,e, m3, mult)
    # Terms: m3(m3(a,b,c), d, e) - m3(a, m3(b,c,d), e) + m3(a, b, m3(c,d,e))
    total = Dict{Symbol,Float64}()
    # + m3(m3(a,b,c), d, e)
    if haskey(m3, (a,b,c))
        for (x, coeff) in m3[(a,b,c)]
            if haskey(m3, (x, d, e))
                merge_dicts!(total, m3[(x, d, e)], coeff)
            end
        end
    end
    # - m3(a, m3(b,c,d), e)
    if haskey(m3, (b,c,d))
        for (x, coeff) in m3[(b,c,d)]
            if haskey(m3, (a, x, e))
                merge_dicts!(total, m3[(a, x, e)], -coeff)
            end
        end
    end
    # + m3(a, b, m3(c,d,e))
    if haskey(m3, (c,d,e))
        for (x, coeff) in m3[(c,d,e)]
            if haskey(m3, (a, b, x))
                merge_dicts!(total, m3[(a, b, x)], coeff)
            end
        end
    end
    return total
end

function build_stalk(region::Symbol, basis::Vector{Symbol}, depth::Int)
    return build_local_stalk(basis, region, depth=depth)
end

# ============================================================
# CORRECTED m5 COMPUTATION (A∞ IDENTITY AT LEVEL 5)
# ============================================================
function compute_local_m5(C5, m3, m4, mult)
    m5 = Dict{NTuple{5,Symbol}, Dict{Symbol,Float64}}()
    for (a,b,c,d,e) in C5
        total = Dict{Symbol,Float64}()
        
        # + a·m4(b,c,d,e)
        d1 = get(m4, (b,c,d,e), Dict{Symbol,Float64}())
        if !isempty(d1)
            tmp = mult_expand_left(mult, a, Dict{Symbol,Float64}(d1))
            merge_dicts!(total, tmp)
        end
        
        # + m4(a,b,c,d)·e
        d2 = get(m4, (a,b,c,d), Dict{Symbol,Float64}())
        if !isempty(d2)
            tmp = mult_expand_right(mult, Dict{Symbol,Float64}(d2), e)
            merge_dicts!(total, tmp)
        end
        
        # + m4(a, m2(b,c), d, e)
        bc = mult(b,c)
        if !isempty(bc)
            tmp = mult_expand_middle_4(mult, m4, a, bc, d, e)
            merge_dicts!(total, tmp)
        end
        
        # + m4(a, b, m2(c,d), e)
        cd = mult(c,d)
        if !isempty(cd)
            for (x, coeff) in cd
                d3 = get(m4, (a, b, x, e), Dict{Symbol,Float64}())
                if !isempty(d3)
                    tmp = Dict{Symbol,Float64}()
                    merge_dicts!(tmp, Dict{Symbol,Float64}(d3), coeff)
                    merge_dicts!(total, tmp)
                end
            end
        end
        
        # + m4(a, b, c, m2(d,e))
        de = mult(d,e)
        if !isempty(de) 
            for (x, coeff) in de
                d4 = get(m4, (a, b, c, x), Dict{Symbol,Float64}())
                if !isempty(d4)
                    merge_dicts!(total, Dict{Symbol,Float64}(d4), coeff)
                end
            end 
        end
        
        # m3∘m3 terms
        m3term = compose_m3_m3(a,b,c,d,e, m3, mult)
        merge_dicts!(total, m3term)
        
        if !isempty(total)
            m5_cancel = Dict{Symbol,Float64}()
            for (k, v) in total
                m5_cancel[k] = -v
            end
            m5[(a,b,c,d,e)] = m5_cancel
        end
    end
    println("Number of non‑zero m5 entries: ", length(m5))
    return m5
end

# ============================================================
# CORRECTED m6 COMPUTATION (A∞ IDENTITY AT LEVEL 6)
# ============================================================
# ----------------------------------------------------------------------
# m6 obstruction (lower terms of the A∞ identity at level 6)
# Returns a dictionary mapping output basis element -> coefficient
# for the sum of all terms that do NOT involve m6 itself.
#
# Arguments:
#   a,b,c,d,e,f : Symbol – the six input arguments
#   m2          : function (Symbol, Symbol) -> Dict{Symbol,Float64}
#   m3          : Dict{NTuple{3,Symbol}, Dict{Symbol,Float64}}
#   m4          : Dict{NTuple{4,Symbol}, Dict{Symbol,Float64}}
#   m5          : Dict{NTuple{5,Symbol}, Dict{Symbol,Float64}}
# ----------------------------------------------------------------------
function m6_obstruction_full(a,b,c,d,e,f, m2, m3, m4, m5)
    total = Dict{Symbol,Float64}()

    # ------------------------------------------------------------------
    # 1. j = 5, i = 2   ->   m2( m5(...), ... )   (2 terms)
    # ------------------------------------------------------------------
    # term: + m2( m5(a,b,c,d,e), f )
    if haskey(m5, (a,b,c,d,e))
        for (x, coeff) in m5[(a,b,c,d,e)]
            tmp = m2(x, f)
            add_dict!(total, tmp, coeff)
        end
    end
    # term: - m2( a, m5(b,c,d,e,f) )
    if haskey(m5, (b,c,d,e,f))
        for (x, coeff) in m5[(b,c,d,e,f)]
            tmp = m2(a, x)
            add_dict!(total, tmp, -coeff)
        end
    end

    # ------------------------------------------------------------------
    # 2. j = 4, i = 3   ->   m3( ..., m4(...), ... )   (3 terms)
    # ------------------------------------------------------------------
    # k=0: + m3( m4(a,b,c,d), e, f )
    if haskey(m4, (a,b,c,d))
        for (x, coeff) in m4[(a,b,c,d)]
            if haskey(m3, (x, e, f))
                add_dict!(total, m3[(x, e, f)], coeff)
            end
        end
    end
    # k=1: - m3( a, m4(b,c,d,e), f )
    if haskey(m4, (b,c,d,e))
        for (x, coeff) in m4[(b,c,d,e)]
            if haskey(m3, (a, x, f))
                add_dict!(total, m3[(a, x, f)], -coeff)
            end
        end
    end
    # k=2: + m3( a, b, m4(c,d,e,f) )
    if haskey(m4, (c,d,e,f))
        for (x, coeff) in m4[(c,d,e,f)]
            if haskey(m3, (a, b, x))
                add_dict!(total, m3[(a, b, x)], coeff)
            end
        end
    end

    # ------------------------------------------------------------------
    # 3. j = 3, i = 4   ->   m4( ..., m3(...), ... )   (4 terms)
    # ------------------------------------------------------------------
    # k=0: + m4( m3(a,b,c), d, e, f )
    if haskey(m3, (a,b,c))
        for (x, coeff) in m3[(a,b,c)]
            if haskey(m4, (x, d, e, f))
                add_dict!(total, m4[(x, d, e, f)], coeff)
            end
        end
    end
    # k=1: - m4( a, m3(b,c,d), e, f )
    if haskey(m3, (b,c,d))
        for (x, coeff) in m3[(b,c,d)]
            if haskey(m4, (a, x, e, f))
                add_dict!(total, m4[(a, x, e, f)], -coeff)
            end
        end
    end
    # k=2: + m4( a, b, m3(c,d,e), f )
    if haskey(m3, (c,d,e))
        for (x, coeff) in m3[(c,d,e)]
            if haskey(m4, (a, b, x, f))
                add_dict!(total, m4[(a, b, x, f)], coeff)
            end
        end
    end
    # k=3: - m4( a, b, c, m3(d,e,f) )
    if haskey(m3, (d,e,f))
        for (x, coeff) in m3[(d,e,f)]
            if haskey(m4, (a, b, c, x))
                add_dict!(total, m4[(a, b, c, x)], -coeff)
            end
        end
    end

    # ------------------------------------------------------------------
    # 4. j = 2, i = 5   ->   m5( ..., m2(...), ... )   (5 terms)
    # ------------------------------------------------------------------
    # k=0: + m5( m2(a,b), c, d, e, f )
    ab = m2(a,b)
    for (x, coeff) in ab
        if haskey(m5, (x, c, d, e, f))
            add_dict!(total, m5[(x, c, d, e, f)], coeff)
        end
    end
    # k=1: - m5( a, m2(b,c), d, e, f )
    bc = m2(b,c)
    for (x, coeff) in bc
        if haskey(m5, (a, x, d, e, f))
            add_dict!(total, m5[(a, x, d, e, f)], -coeff)
        end
    end
    # k=2: + m5( a, b, m2(c,d), e, f )
    cd = m2(c,d)
    for (x, coeff) in cd
        if haskey(m5, (a, b, x, e, f))
            add_dict!(total, m5[(a, b, x, e, f)], coeff)
        end
    end
    # k=3: - m5( a, b, c, m2(d,e), f )
    de = m2(d,e)
    for (x, coeff) in de
        if haskey(m5, (a, b, c, x, f))
            add_dict!(total, m5[(a, b, c, x, f)], -coeff)
        end
    end
    # k=4: + m5( a, b, c, d, m2(e,f) )
    ef = m2(e,f)
    for (x, coeff) in ef
        if haskey(m5, (a, b, c, d, x))
            add_dict!(total, m5[(a, b, c, d, x)], coeff)
        end
    end

    return total
end


function compute_m6_selective(C6, mult, m3, m4, m5; tol=1e-6)
    m6 = Dict{NTuple{6,Symbol}, Dict{Symbol,Float64}}()
    max_norm = 0.0
    for (a,b,c,d,e,f) in C6
        #total = m6_obstruction(a,b,c,d,e,f, mult, m3, m4, m5)
        total = m6_obstruction_full(a,b,c,d,e,f, mult, m3, m4, m5)
        nrm = norm_dict(total)
        if nrm > tol
            m6_cancel = Dict{Symbol,Float64}()
            for (k, v) in total
                m6_cancel[k] = -v
            end
            m6[(a,b,c,d,e,f)] = m6_cancel
            max_norm = max(max_norm, nrm)
        end
    end
    println("Nonzero m6 entries: ", length(m6))
    println("Max m6 obstruction: ", max_norm)
    return m6
end

# ============================================================
# FIX TYPOS AND MISSING EXPORTS
# ============================================================

# Ensure `C5_stalk` uses the correct function name.
function build_Ck(basis::Vector{Symbol}, k::Int)

    if k == 1
        return [(x,) for x in basis]
    end

    Cprev = build_Ck(basis, k-1)
    Ck = Vector{NTuple{k,Symbol}}()

    for tup in Cprev
        last = tup[end]
        for x in basis
            if is_composable(last, x)
                push!(Ck, (tup..., x))
            end
        end
    end

    return Ck
end
function build_Ck_limited(basis, k; max_size=50000)

    Ck = build_Ck(basis, k)

    if length(Ck) > max_size
        println("Truncating C$k from $(length(Ck)) to $max_size")
        return Ck[1:max_size]
    end

    return Ck
end

# Example usage after stalk is built:
stalk_basis = build_stalk(:sAMY, basis, 2)
C5_stalk = build_Ck_limited(stalk_basis, 5)   # corrected name
m5_stalk = compute_local_m5(C5_stalk, m3_element, m4_corrected, mult_dict)

# For C6, you need to define C6 from C5 (composable chains). Add:
function build_C6_from_C5(C5, basis)
    C6 = []
    for (a,b,c,d,e) in C5
        for f in basis
            if is_composable(e, f)
                push!(C6, (a,b,c,d,e,f))
            end
        end
    end
    return C6
end
C6_stalk = build_C6_from_C5(C5_stalk, stalk_basis)
m6_stalk = compute_m6_selective(C6_stalk, mult_dict, m3_element, m4_corrected, m5_stalk)

# Build C6 (composable 6‑tuples) with a size limit.
# If C5 is provided, extend it; otherwise build C6 from basis directly.
function build_C6_limited(basis, max_size::Int=50000; C5=nothing)
    C6 = NTuple{6,Symbol}[]
    
    if C5 !== nothing
        # Extend from existing C5 (faster)
        for (a,b,c,d,e) in C5
            for f in basis
                if is_composable(e, f)
                    push!(C6, (a,b,c,d,e,f))
                    if length(C6) >= max_size
                        println("Truncating C6 at $max_size")
                        return C6
                    end
                end
            end
        end
    else
        # Build from scratch: first build C2, C3, C4, C5, then C6
        C2 = [(x,y) for x in basis for y in basis if is_composable(x,y)]
        C3 = [(x,y,z) for (x,y) in C2 for z in basis if is_composable(y,z)]
        C4 = [(x,y,z,w) for (x,y,z) in C3 for w in basis if is_composable(z,w)]
        C5 = [(x,y,z,w,u) for (x,y,z,w) in C4 for u in basis if is_composable(w,u)]
        
        for (a,b,c,d,e) in C5
            for f in basis
                if is_composable(e, f)
                    push!(C6, (a,b,c,d,e,f))
                    if length(C6) >= max_size
                        println("Truncating C6 at $max_size")
                        return C6
                    end
                end
            end
        end
    end
    
    println("Built C6 with $(length(C6)) entries")
    return C6
end

# Example usage:
# C6 = build_C6_limited(basis, 50000)               # from scratch
# C6 = build_C6_limited(stalk_basis, 10000, C5=C5_stalk)  # from existing C5


function build_C6_selective(C5, mult; max_size=5000)
    C6 = NTuple{6,Symbol}[]
    for (a,b,c,d,e) in C5
        for f in basis
            if is_composable(e,f)
                push!(C6, (a,b,c,d,e,f))
                if length(C6) > max_size
                    return C6
                end
            end
        end
    end
    return C6
end
# a · m5(b,c,d,e,f)
# m5(a,b,c,d,e) · f
# m5(a, m2(b,c), d,e,f)
# m5(a, b, m2(c,d), e,f)
# m5(a, b, c, m2(d,e), f)
# m5(a, b, c, d, m2(e,f))

#m4(m3(a,b,c), d, e, f)
#m4(a, m3(b,c,d), e, f)
#m4(a, b, m3(c,d,e), f)
#m4(a, b, c, m3(d,e,f))

#m3(m4(a,b,c,d), e, f)
#m3(a, m4(b,c,d,e), f)
#m3(a, b, m4(c,d,e,f))

#m6 = - (δm5 + m4∘m3 + m3∘m4)
# Expensive -- Builds Full.
function compute_m6(C6, m2, m3, m4, m5; tol=1e-12)
    m6 = Dict{NTuple{6,Symbol}, Dict{Symbol,Float64}}()
    
    for tup in C6
        a,b,c,d,e,f = tup
        obs = m6_obstruction_full(a,b,c,d,e,f, m2, m3, m4, m5)
        
        if norm_dict(obs) > tol
            # m6 must cancel the obstruction
            m6[tup] = Dict(k => -v for (k,v) in obs)
        end
    end
    
    println("Computed $(length(m6)) non‑zero m6 entries")
    return m6
end

# Local Stalk based building for composable paths only
# Build composable 5‑tuples and 6‑tuples on the stalk
function build_C5_local(stalk_basis)
    C2 = [(x,y) for x in stalk_basis for y in stalk_basis if is_composable(x,y)]
    C3 = [(x,y,z) for (x,y) in C2 for z in stalk_basis if is_composable(y,z)]
    C4 = [(x,y,z,w) for (x,y,z) in C3 for w in stalk_basis if is_composable(z,w)]
    C5 = [(x,y,z,w,u) for (x,y,z,w) in C4 for u in stalk_basis if is_composable(w,u)]
    return C5
end

function build_C6_local(C5, stalk_basis; max_size=50000)
    C6 = NTuple{6,Symbol}[]
    for (a,b,c,d,e) in C5
        for f in stalk_basis
            if is_composable(e, f)
                push!(C6, (a,b,c,d,e,f))
                length(C6) >= max_size && return C6
            end
        end
    end
    return C6
end

# Helper: add contents of dict B into dict A with an optional scale factor
function add_dict!(A::Dict{Symbol,Float64}, B::Dict{Symbol,Float64}, scale::Float64=1.0)
    for (k, v) in B
        A[k] = get(A, k, 0.0) + scale * v
    end
    return A
end



function m6_obstruction(a,b,c,d,e,f, mult, m3, m4, m5)
    total = Dict{Symbol,Float64}()

    # --- (1) m2 ∘ m4 terms ---
    if haskey(m4, (a,b,c,d))
        tmp = mult_expand_left(mult, a, m4[(b,c,d,e)])
        add_dict!(total, tmp, +1.0)
    end

    if haskey(m4, (b,c,d,e))
        tmp = mult_expand_left(mult, a, m4[(b,c,d,e)])
        add_dict!(total, tmp, -1.0)
    end

    # --- (2) m3 ∘ m3 terms ---
    if haskey(m3, (a,b,c)) && haskey(m3, (d,e,f))
        left = m3[(a,b,c)]
        tmp = mult_expand_right(mult, left, d)
        tmp2 = mult_expand_right(mult, tmp, e)
        tmp3 = mult_expand_right(mult, tmp2, f)
        add_dict!(total, tmp3, +1.0)
    end

    if haskey(m3, (c,d,e)) && haskey(m3, (a,b,c))
        right = m3[(c,d,e)]
        tmp = mult_expand_left(mult, a, right)
        tmp2 = mult_expand_right(mult, tmp, f)
        add_dict!(total, tmp2, -1.0)
    end

    # --- (3) m4 ∘ m2 terms ---
    if haskey(m4, (b,c,d,e))
        tmp = mult_expand_right(mult, m4[(b,c,d,e)], f)
        add_dict!(total, tmp, +1.0)
    end

    if haskey(m4, (a,b,c,d))
        tmp = mult_expand_right(mult, m4[(a,b,c,d)], e)
        tmp2 = mult_expand_right(mult, tmp, f)
        add_dict!(total, tmp2, -1.0)
    end

    # --- (4) m5 differential term (optional if available) ---
    if haskey(m5, (a,b,c,d,e))
        tmp = mult_expand_right(mult, m5[(a,b,c,d,e)], f)
        add_dict!(total, tmp, +1.0)
    end

    if haskey(m5, (b,c,d,e,f))
        tmp = mult_expand_left(mult, a, m5[(b,c,d,e,f)])
        add_dict!(total, tmp, -1.0)
    end

    return total
end




function compute_local_m5(C5, m3, m4, mult)

    m5 = Dict{NTuple{5,Symbol}, Dict{Symbol,Float64}}()

    for (a,b,c,d,e) in C5
        total = Dict{Symbol,Float64}()

        # --- A∞ identity at level 5 ---
        # δm4 + m3∘m3 terms

        # (1) a·m4(b,c,d,e)
        tmp1 = mult_expand_left(mult, a, get(m4, (b,c,d,e), Dict()))
        merge_dicts!(total, tmp1)

        # (2) m4(a,b,c,d)·e
        tmp2 = mult_expand_right(mult, get(m4, (a,b,c,d), Dict()), e)
        merge_dicts!(total, tmp2)

        # (3) m4(a, m2(b,c), d, e)
        bc = mult(b,c)
        tmp3 = mult_expand_middle_4(mult, m4, a, bc, d, e)
        merge_dicts!(total, tmp3)

        # (4) m3∘m3 terms (CRITICAL)
        tmp4 = compose_m3_m3(a,b,c,d,e, m3, mult)
        merge_dicts!(total, tmp4)

        if !isempty(total)
            m5[(a,b,c,d,e)] = total
        end
    end

    return m5
end

function get_region(sym::Symbol)
    s = String(sym)

    if startswith(s, "e_")
        return Symbol(s[3:end])
    elseif startswith(s, "f_")
        parts = split(s, "_")
        return Symbol(parts[2])   # source
    else
        return nothing
    end
end

function get_regions(sym::Symbol)
    s = String(sym)

    if startswith(s, "e_")
        r = Symbol(s[3:end])
        return (r, r)
    elseif startswith(s, "f_")
        parts = split(s, "_")
        return (Symbol(parts[2]), Symbol(parts[3]))
    else
        return nothing
    end
end

function compute_region_heatmap_m6(m6)
    region_score = Dict{Symbol,Float64}()

    for (path, val) in m6
        w = norm_dict(val)

        for sym in path
            r1, r2 = get_regions(sym)

            region_score[r1] = get(region_score, r1, 0.0) + w
            region_score[r2] = get(region_score, r2, 0.0) + w
        end
    end

    return region_score
end

function normalize_scores!(scores)
    maxv = maximum(values(scores))
    for k in keys(scores)
        scores[k] /= maxv
    end
end

function print_region_ranking(scores)
    sorted = sort(collect(scores), by=x->-x[2])
    println("\n--- m6 Region Heatmap ---")
    for (r,v) in sorted
        println(r, " : ", v)
    end
end

function compute_edge_heatmap_m6(m6)
    edge_score = Dict{Tuple{Symbol,Symbol},Float64}()

    for (path, val) in m6
        w = norm_dict(val)

        for sym in path
            r1, r2 = get_regions(sym)
            edge_score[(r1,r2)] = get(edge_score,(r1,r2),0.0) + w
        end
    end

    return edge_score
end

function top_m6_paths(m6; k=10)
    scored = []

    for (path, val) in m6
        push!(scored, (path, norm_dict(val)))
    end

    sort!(scored, by=x->-x[2])
    return scored[1:min(k,length(scored))]
end

function path_energy(path, m4, m5, m6; α=1.0, β=2.0, γ=3.0)
    score = 0.0

    # m4 contributions (sub-4 paths)
    for i in 1:length(path)-3
        key = (path[i],path[i+1],path[i+2],path[i+3])
        if haskey(m4, key)
            score += α * norm_dict(m4[key])
        end
    end

    # m5 contributions
    for i in 1:length(path)-4
        key = (path[i],path[i+1],path[i+2],path[i+3],path[i+4])
        if haskey(m5, key)
            score += β * norm_dict(m5[key])
        end
    end

    # m6 contributions
    for i in 1:length(path)-5
        key = (path[i],path[i+1],path[i+2],path[i+3],path[i+4],path[i+5])
        if haskey(m6, key)
            score += γ * norm_dict(m6[key])
        end
    end

    return score
end

function generate_paths(C3, C4, C5, C6)
    paths = Vector{Tuple{Vararg{Symbol}}}()

    for C in (C3, C4, C5, C6), t in C
        push!(paths, normalize_path(t))
    end

    return paths
end

function score_paths(paths, m4, m5, m6)
    scored = []

    for p in paths
        s = path_energy(p, m4, m5, m6)
        if s > 0
            push!(scored, (p, s))
        end
    end

    sort!(scored, by=x->-x[2])
    return scored
end

function path_to_regions(path)
    regions = []

    for sym in path
        r1, r2 = get_regions(sym)
        push!(regions, r1)
    end

    return regions
end

function top_region_pathways(scored_paths; k=20)
    results = []

    for (p,s) in scored_paths[1:min(k,length(scored_paths))]
        push!(results, (path_to_regions(p), s))
    end

    return results
end

function is_septohippocampal(path)
    return (:MS in path) && (:CA1sp in path || :CA3 in path || :DG in path)
end

function cluster_paths(paths)
    clusters = Dict{Set{Symbol}, Float64}()

    for (p,s) in paths
        key = Set(p)
        clusters[key] = get(clusters,key,0.0) + s
    end

    return clusters
end

function cluster_paths_exact(paths)
    clusters = Dict{Vector{Symbol}, Float64}()

    for (p,s) in paths
        clusters[p] = get(clusters, p, 0.0) + s
    end

    return clusters
end

function cluster_paths_by_regions(paths)
    clusters = Dict{Vector{Symbol}, Float64}()
    for (p,s) in paths
        rpath = path_to_regions(p)          # returns Vector{String}
        rpath_sym = Symbol.(rpath)          # convert to Vector{Symbol}
        clusters[rpath_sym] = get(clusters, rpath_sym, 0.0) + s
    end
    return clusters
end

function extract_motifs(path, k)
    motifs = []
    for i in 1:length(path)-k+1
        push!(motifs, path[i:i+k-1])
    end
    return motifs
end

function canonical_path(p)
    return min(p, reverse(p))
end



using CSV, DataFrames

df = CSV.read(NODES, DataFrame)
# Safe parser for region strings like "['bgr']" or "['bgr', 'cortex']"
function parse_regions(s::String)
    # Remove outer brackets
    s = strip(s)
    if startswith(s, '[') && endswith(s, ']')
        s = s[2:end-1]
    else
        return String[]
    end
    # Split by commas
    parts = split(s, ',')
    regions = String[]
    for p in parts
        p = strip(p)
        # Remove surrounding single or double quotes
        if startswith(p, '\'') && endswith(p, '\'')
            p = p[2:end-1]
        elseif startswith(p, '"') && endswith(p, '"')
            p = p[2:end-1]
        end
        if !isempty(p)
            push!(regions, p)
        end
    end
    return regions
end

node_regions = Dict{Int, Vector{String}}()

for r in eachrow(df)
    node_regions[r.id] = parse_regions(String(r.regions))   # use the safe parser
end

function compute_node_scores(node_regions, region_scores)
    node_score = Dict{Int, Float64}()

    for (nid, regs) in node_regions
        s = 0.0
        for r in regs
            if haskey(region_scores, Symbol(r))
                s += region_scores[Symbol(r)]
            end
        end
        node_score[nid] = s
    end

    return node_score
end

function add_path_scores!(node_score, node_regions, scored_paths)
    for (path, weight) in scored_paths   # path is a Vector{Symbol} or Tuple
        # extract regions involved in path
        regs = Set{Symbol}()
        for sym in path
            sx = String(sym)
            if startswith(sx, "f_")
                parts = split(sx, "_")
                push!(regs, Symbol(parts[2]))
                push!(regs, Symbol(parts[3]))
            elseif startswith(sx, "e_")
                push!(regs, Symbol(sx[3:end]))
            end
        end
        # add weight to each node that belongs to any of these regions
        for (nid, rlist) in node_regions
            for r in rlist
                if Symbol(r) in regs
                    node_score[nid] = get(node_score, nid, 0.0) + weight
                end
            end
        end
    end
end



function write_vtk_fast(nodes_df, edges_df, node_score, filename; 
                        node_cluster=nothing, node_motif_score=nothing)
    
    # 1. Extract coordinates into a 3xN matrix (Required by WriteVTK)
    # Assuming nodes_df has columns x, y, z
    points = hcat(nodes_df.pos_x, nodes_df.pos_y, nodes_df.pos_z)'
    
    # 2. Create the grid (UnstructuredGrid for nodes and edges)
    # Map edges to VTK_LINE cells
    cells = [
        MeshCell(VTK_LINE, [edges_df.node1id[i] - 1, edges_df.node2id[i] - 1]) 
        for i in 1:nrow(edges_df)
    ]    
    # 3. Initialize the VTK file (This uses binary XML by default)
    vtk_grid(filename, points, cells) do vtk
        
        # 4. Add Point Data (Optimized binary transfer)
        intensity = [get(node_score, id, 0.0) for id in nodes_df.id]
        vtk["intensity"] = intensity
        
        if node_cluster !== nothing
            clusters = [get(node_cluster, id, -1) for id in nodes_df.id]
            vtk["cluster_id"] = clusters
        end
        
        if node_motif_score !== nothing
            motifs = [get(node_motif_score, id, 0.0) for id in nodes_df.id]
            vtk["motif_score"] = motifs
        end
    end
end

function write_vtk(nodes_df, edges_df, node_score, filename;
    node_cluster=nothing, node_motif_score=nothing)
    open(filename, "w") do io
        # ... (same header, POINTS, LINES as before) ...

        # Primary intensity
        println(io, "POINT_DATA $(nrow(nodes_df))")
        println(io, "SCALARS intensity float 1")
        println(io, "LOOKUP_TABLE default")
        for r in eachrow(nodes_df)
            println(io, get(node_score, r.id, 0.0))
        end

        # Optional cluster ID (integer)
        if node_cluster !== nothing
            println(io, "SCALARS cluster_id int 1")
            println(io, "LOOKUP_TABLE default")
            for r in eachrow(nodes_df)
                println(io, get(node_cluster, r.id, -1))
            end
        end

        # Optional motif score
        if node_motif_score !== nothing
            println(io, "SCALARS motif_score float 1")
            println(io, "LOOKUP_TABLE default")
            for r in eachrow(nodes_df)
                println(io, get(node_motif_score, r.id, 0.0))
            end
        end
    end
end

function dict_norm(D::Dict{Symbol,Float64})
    s = 0.0
    for v in values(D)
        s += v*v
    end
    return sqrt(s)
end

function node_score_m3(m3)
    score = Dict{Symbol,Float64}()

    for ((a,b,c), val) in m3
        w = dict_norm(val)
        for x in (a,b,c)
            score[x] = get(score,x,0.0) + w
        end
    end

    return score
end

function node_score_m4(m4)
    score = Dict{Symbol,Float64}()

    for ((a,b,c,d), val) in m4
        w = dict_norm(val)
        for x in (a,b,c,d)
            score[x] = get(score,x,0.0) + w
        end
    end

    return score
end

function node_score_m5(m5)
    score = Dict{Symbol,Float64}()

    for ((a,b,c,d,e), val) in m5
        w = dict_norm(val)
        for x in (a,b,c,d,e)
            score[x] = get(score,x,0.0) + w
        end
    end

    return score
end

function combine_scores(m3s, m4s, m5s; α=1.0, β=2.0, γ=3.0)
    total = Dict{Symbol,Float64}()

    nodes = union(keys(m3s), keys(m4s), keys(m5s))

    for n in nodes
        total[n] =
            α * get(m3s,n,0.0) +
            β * get(m4s,n,0.0) +
            γ * get(m5s,n,0.0)
    end

    return total
end

function region_scores_hm(node_scores, node_to_region)
    reg = Dict{String,Float64}()

    for (node, val) in node_scores
        if haskey(node_to_region, node)
            for r in node_to_region[node]
                reg[r] = get(reg,r,0.0) + val
            end
        end
    end

    return reg
end

#(a,b,c,d,e,f) => Dict(target => weight)

function extract_m6_paths(m6)
    paths = Dict{NTuple{6,Symbol}, Float64}()

    for (tuple, out) in m6
        total = 0.0
        for (_, v) in out
            total += abs(v)
        end
        paths[tuple] = total
    end

    return paths
end

function symbol_to_region(sym::Symbol)
    s = String(sym)

    if startswith(s, "e_")
        return s[3:end]
    elseif startswith(s, "f_")
        parts = split(s, "_")
        return parts[2]   # source region
    end

    return "UNK"
end

function path_to_regions(path)
    return [symbol_to_region(p) for p in path]
end

function compute_region_scores(paths)
    scores = Dict{String, Float64}()

    for (p, w) in paths
        regions = path_to_regions(p)
        for r in regions
            scores[r] = get(scores, r, 0.0) + w
        end
    end

    return scores
end

function compute_edge_scores(paths)
    edges = Dict{Tuple{String,String}, Float64}()

    for (p, w) in paths
        regions = path_to_regions(p)

        for i in 1:length(regions)-1
            e = (regions[i], regions[i+1])
            edges[e] = get(edges, e, 0.0) + w
        end
    end

    return edges
end

function build_node_region_map(node_table)
    node_regions = Dict{Int, Vector{String}}()

    for row in node_table
        id = row.id
        regions = row.regions  # already parsed
        node_regions[id] = regions
    end

    return node_regions
end

function compute_node_heat(node_regions, region_scores)
    node_heat = Dict{Int, Float64}()

    for (nid, regs) in node_regions
        h = 0.0
        for r in regs
            h += get(region_scores, r, 0.0)
        end
        node_heat[nid] = h
    end

    return node_heat
end

function compute_voxel_edge_heat(edges_region, edge_table, node_regions)
    edge_heat = Dict{Int, Float64}()

    for edge in edge_table
        r1 = node_regions[edge.node1id]
        r2 = node_regions[edge.node2id]

        total = 0.0
        for a in r1, b in r2
            total += get(edges_region, (a,b), 0.0)
            total += get(edges_region, (b,a), 0.0)
        end

        edge_heat[edge.id] = total
    end

    return edge_heat
end

function top_k_paths(paths, k=10)
    sorted = sort(collect(paths), by=x->-x[2])
    return sorted[1:min(k, length(sorted))]
end

#for (p, w) in top_k_paths(m6_paths, 10)
#    println(path_to_regions(p), " : ", w)
#end


function singular_support(defects; threshold=1e6)
    return [k for (k,v) in defects if v > threshold]
end

function pathway_singular_overlap(paths, singular_edges)
    selected = []

    for (p,w) in paths
        regions = path_to_regions(p)

        for i in 1:length(regions)-1
            if (regions[i], regions[i+1]) in singular_edges
                push!(selected, (p,w))
                break
            end
        end
    end

    return selected
end

# ----------------------------------------------------------------------
# Verify A∞ identity at level 6:
# For every composable 6‑tuple (a,b,c,d,e,f), compute:
#   total = δm₅ + m₄∘m₃ + m₃∘m₄ + m₆
# where δm₅ is computed by m6_obstruction_full (which already includes all
# terms that do NOT contain m6). Then add m6 explicitly.
# The identity requires total == 0.
# ----------------------------------------------------------------------
function check_Ainf_level6_consistent(C6, m2, m3, m4, m5, m6; tol=1e-6)
    println("Checking A∞ identity at level 6...")
    max_violation = 0.0
    violations = 0
    
    for tup in C6
        a,b,c,d,e,f = tup
        
        # Compute the lower-degree obstruction (δm₅ + m₄∘m₃ + m₃∘m₄)
        total = m6_obstruction_full(a,b,c,d,e,f, m2, m3, m4, m5)
        
        # Add m6 term (which should cancel the obstruction)
        if haskey(m6, tup)
            for (k, v) in m6[tup]
                total[k] = get(total, k, 0.0) + v
            end
        end
        
        nrm = norm_dict(total)
        if nrm > tol
            violations += 1
            max_violation = max(max_violation, nrm)
            if violations <= 5  # print first few violations
                println("Violation at $tup : norm = $nrm")
                println("  total dictionary: $total")
            end
        end
    end
    
    println("Max violation (δm₅ + m₄∘m₃ + m₃∘m₄ + m₆): $max_violation")
    println("Number of violating 6‑tuples: $violations out of $(length(C6))")
    
    if max_violation < tol
        println("✅ A∞ identity at level 6 is satisfied (residual < $tol).")
    else
        println("⚠️ A∞ identity at level 6 has violations above tolerance.")
    end
    
    return violations
end
#=
Perverse sheaf ≈
“tracking failure of local algebra to glue globally”
m₆-high paths = geodesics through singular support
high m₆ weight
+
crosses singular support

singularity → local blowup (stalk)
→ track resolution via A∞ corrections

microlocal geometry of brain dynamics via A∞-sheaf theory

=#

# ============================================================
# 0. ASSUMPTIONS (already computed in your pipeline)
# ============================================================
# basis :: Vector{Symbol}
# m3    :: Dict{NTuple{3,Symbol}, Dict{Symbol,Float64}}
# m4    :: Dict{NTuple{4,Symbol}, Dict{Symbol,Float64}}
# m5    :: Dict{NTuple{5,Symbol}, Dict{Symbol,Float64}}
# m6    :: Dict{NTuple{6,Symbol}, Dict{Symbol,Float64}}
# mult  :: function returning Dict{Symbol,Float64}

# regions list (IMPORTANT: define explicitly)
regions = ["sAMY", "HPF", "BLA", "CA1sp", "HY", "LA"]

# ============================================================
# 1. BUILD STALKS
# ============================================================

function build_all_stalks(regions, basis, m3, m4, m5, m6; depth=2)
    stalks = Dict{String, Any}()

    for r in regions
        println("\n--- Building stalk for $r ---")

        nodes = build_stalk(r, depth)   # you already have this

        m3_s = restrict_m3(m3, nodes)
        m4_s = restrict_m4(m4, nodes)

        # Build local C5/C6 ONLY for stalk
        C5_s = build_C5(nodes)
        C6_s = build_C6(nodes)

        m5_s = compute_local_m5(C5_s, m3_s, m4_s, mult)
        m6_s = compute_m6_selective(C6_s, mult, m3_s, m4_s, m5_s)

        stalks[r] = Dict(
            :basis => nodes,
            :m3 => m3_s,
            :m4 => m4_s,
            :m5 => m5_s,
            :m6 => m6_s
        )

        println("  size: ", length(nodes))
        println("  m3: ", length(m3_s), " m4: ", length(m4_s),
                " m5: ", length(m5_s), " m6: ", length(m6_s))
    end

    return stalks
end

# ============================================================
# 2. BUILD RESTRICTIONS
# ============================================================

function build_all_restrictions(stalks)
    restrictions = Dict{Tuple{String,String}, Any}()

    for r1 in keys(stalks), r2 in keys(stalks)
        if r1 == r2; continue; end

        A = Set(stalks[r1][:basis])
        B = Set(stalks[r2][:basis])

        inter = intersect(A,B)

        restrictions[(r1,r2)] = Dict(
            :intersection => inter
        )
    end

    return restrictions
end

# ============================================================
# 3. GLUING DEFECT
# ============================================================

function diff_dict(d1, d2)
    keys_all = union(keys(d1), keys(d2))
    out = Dict{Symbol,Float64}()

    for k in keys_all
        out[k] = get(d1,k,0.0) - get(d2,k,0.0)
    end

    return out
end

function norm_dict(d)
    s = 0.0
    for v in values(d)
        s += v*v
    end
    return sqrt(s)
end

function gluing_defect(stalkA, stalkB, inter)
    defect = 0.0

    # --- m3 ---
    for (k,v) in stalkA[:m3]
        if all(x -> x in inter, k)
            v2 = get(stalkB[:m3], k, Dict{Symbol,Float64}())
            defect += norm_dict(diff_dict(v, v2))
        end
    end

    # --- m4 ---
    for (k,v) in stalkA[:m4]
        if all(x -> x in inter, k)
            v2 = get(stalkB[:m4], k, Dict{Symbol,Float64}())
            defect += norm_dict(diff_dict(v, v2))
        end
    end

    return defect
end

# ============================================================
# 4. BUILD PERVERSE SHEAF OBJECT
# ============================================================

function build_perverse_sheaf(stalks, restrictions)
    defects = Dict{Tuple{String,String}, Float64}()

    println("\n--- Computing gluing defects ---")

    for ((r1,r2), data) in restrictions
        inter = data[:intersection]

        d = gluing_defect(stalks[r1], stalks[r2], inter)

        defects[(r1,r2)] = d

        println("$r1 ↔ $r2 : $d")
    end

    return Dict(
        :stalks => stalks,
        :restrictions => restrictions,
        :defects => defects
    )
end

# ============================================================
# 5. EXTRACT SINGULAR SUPPORT
# ============================================================

function singular_support(defects; threshold=1e6)
    return [k for (k,v) in defects if v > threshold]
end

# ============================================================
# 6. PATH EXTRACTION FROM m6
# ============================================================

function extract_m6_paths(m6)
    paths = Dict{NTuple{6,Symbol}, Float64}()

    for (p, out) in m6
        total = 0.0
        for (_,v) in out
            total += abs(v)
        end
        paths[p] = total
    end

    return paths
end

function symbol_to_region(sym::Symbol)
    s = String(sym)

    if startswith(s, "e_")
        return s[3:end]
    elseif startswith(s, "f_")
        return split(s, "_")[2]
    end

    return "UNK"
end

function path_to_regions(path)
    return [symbol_to_region(p) for p in path]
end

# ============================================================
# 7. FILTER PATHS THROUGH SINGULARITIES
# ============================================================

function filter_singular_paths(paths, singular_edges)
    selected = []

    for (p,w) in paths
        regs = path_to_regions(p)

        for i in 1:length(regs)-1
            if (regs[i], regs[i+1]) in singular_edges
                push!(selected, (p,w))
                break
            end
        end
    end

    return selected
end










# ----------------------------------------------------------------------------
# 4. Main driver: compute stalks, m5/m6, perverse sheaf, and pathways
# ----------------------------------------------------------------------------
stalk_data = Dict{String, Dict{Symbol, Any}}()
global_m6_paths = Dict{NTuple{6,Symbol}, Float64}()   # aggregate weights across stalks

for region in regions
    println("\n--- Processing region: $region ---")
    
    local stalk_basis = build_stalk(Symbol(region), basis, 2)
    println("Stalk basis size: $(length(stalk_basis))")
    
    local m3_stalk = restrict_dict(m3_element, stalk_basis)
    local m4_stalk = restrict_dict(m4_corrected, stalk_basis)
    println("m3 entries: $(length(m3_stalk)), m4 entries: $(length(m4_stalk))")
    
    local C5_stalk = build_Ck_limited(stalk_basis, 5; max_size=20000)
    local C6_stalk = build_C6_from_C5(C5_stalk, stalk_basis)
    println("C5 size: $(length(C5_stalk)), C6 size: $(length(C6_stalk))")
    
    local m5_stalk = compute_local_m5(C5_stalk, m3_stalk, m4_stalk, mult_dict)
    local m6_stalk = compute_m6_selective(C6_stalk, mult_dict, m3_stalk, m4_stalk, m5_stalk)
    
    stalk_data[region] = Dict(
        :basis => stalk_basis,
        :m3 => m3_stalk,
        :m4 => m4_stalk,
        :m5 => m5_stalk,
        :m6 => m6_stalk
    )
    println("m5 entries: $(length(m5_stalk)), m6 entries: $(length(m6_stalk))")
    
    # ----- Path scoring and clustering for this stalk -----
    # Build all composable paths within this stalk (C3 to C6)
    C3_local = [(a,b,c) for a in stalk_basis for b in stalk_basis for c in stalk_basis if is_composable(a,b) && is_composable(b,c)]
    C4_local = [(a,b,c,d) for (a,b,c) in C3_local for d in stalk_basis if is_composable(c,d)]
    C5_local = [(a,b,c,d,e) for (a,b,c,d) in C4_local for e in stalk_basis if is_composable(d,e)]
    C6_local = [(a,b,c,d,e,f) for (a,b,c,d,e) in C5_local for f in stalk_basis if is_composable(e,f)]
    
    # verify_m6
    # Assuming you have C6_stalk (composable 6‑tuples on the stalk)
    # and m2 = mult_dict, m3 = m3_stalk, m4 = m4_stalk, m5 = m5_stalk, m6 = m6_stalk
    check_Ainf_level6_consistent(C6_stalk, mult_dict, m3_stalk, m4_stalk, m5_stalk, m6_stalk; tol=1e-6)
    all_paths_local = generate_paths(C3_local, C4_local, C5_local, C6_local)
    scored_local = score_paths(all_paths_local, m4_stalk, m5_stalk, m6_stalk)
    top_local = scored_local[1:min(100, end)]
    
    println("Top 5 paths in $region stalk:")
    for (p, w) in top_local[1:min(5, end)]
        avg = w / length(p)
        println("   $(path_to_regions(p)) : total=$w, avg=$avg")
    end
    
    # Aggregate global m6 weights (for later singular pathway filtering)
    for (path, coeff_dict) in m6_stalk
        weight = sum(abs(v) for v in values(coeff_dict))
        global_m6_paths[path] = get(global_m6_paths, path, 0.0) + weight
    end
end

# ----------------------------------------------------------------------------
# 5. Build restrictions and gluing defects
# ----------------------------------------------------------------------------
function build_restriction(stalkA::Dict, stalkB::Dict)
    A = Set(stalkA[:basis])
    B = Set(stalkB[:basis])
    inter = intersect(A, B)
    return Dict(:intersection => inter)
end

restrictions = Dict{Tuple{String,String}, Dict}()
for r1 in regions, r2 in regions
    r1 == r2 && continue
    restrictions[(r1, r2)] = build_restriction(stalk_data[r1], stalk_data[r2])
end

function gluing_defect(stalkA::Dict, stalkB::Dict, inter::Set{Symbol})
    defect = 0.0
    for (k, v1) in stalkA[:m3]
        if all(x -> x in inter, k)
            v2 = get(stalkB[:m3], k, Dict{Symbol,Float64}())
            defect += norm_dict(diff_dict(v1, v2))
        end
    end
    for (k, v1) in stalkA[:m4]
        if all(x -> x in inter, k)
            v2 = get(stalkB[:m4], k, Dict{Symbol,Float64}())
            defect += norm_dict(diff_dict(v1, v2))
        end
    end
    return defect
end

defects = Dict{Tuple{String,String}, Float64}()
for (r1, r2) in keys(restrictions)
    inter = restrictions[(r1, r2)][:intersection]
    d = gluing_defect(stalk_data[r1], stalk_data[r2], inter)
    defects[(r1, r2)] = d
    println("Gluing defect $r1 → $r2 : $d")
end

perverse_sheaf = Dict(
    :stalks => stalk_data,
    :restrictions => restrictions,
    :defects => defects
)


# ----------------------------------------------------------------------------
# 6. Singular support and pathway filtering
# ----------------------------------------------------------------------------
function singular_support(defects; threshold=1e6)
    return [k for (k, v) in defects if v > threshold]
end

sing_edges = singular_support(defects, threshold=1e6)
println("\n--- Singular support (defect > 1e6) ---")
for (r1, r2) in sing_edges
    println("$r1 ↔ $r2")
end

# Filter global m6 paths that cross a singular edge
function path_crosses_singular_edge(path, sing_edges)
    regs = path_to_regions(path)
    for i in 1:length(regs)-1
        (regs[i], regs[i+1]) in sing_edges && return true
    end
    return false
end

singular_paths = [(p, w) for (p, w) in global_m6_paths if path_crosses_singular_edge(p, sing_edges)]
sorted_singular = sort(singular_paths, by=x->-x[2])

println("\n--- Top 10 singular m6 pathways (crossing singular support) ---")
for i in 1:min(10, length(sorted_singular))
    path, w = sorted_singular[i]
    println("$(path_to_regions(path)) : $w")
end

println("\nDriver finished. Perverse sheaf and pathways computed.")
C4 = [(a,b,c,d) for (a,b,c) in C3 for d in basis if tgt(c) == src(d)]
C5 = [(a,b,c,d,e) for (a,b,c,d) in C4 for e in basis if tgt(d) == src(e)]
C6 = [(a,b,c,d,e,f) for (a,b,c,d,e) in C5 for f in basis if tgt(e) == src(f)]
# 1. Generate all composable paths from C3, C4, C5, C6 (if you have them)
#    If you don't have global C4, C5, C6, you can build them from the full basis:
# ----------------------------------------------------------------------------
# Ensure global composable chains C4, C5, C6 exist (with truncation)
# ----------------------------------------------------------------------------
if !@isdefined(C4)
    println("Building global C4...")
    C4 = [(a,b,c,d) for (a,b,c) in C3 for d in basis if tgt(c) == src(d)]
    println("C4 size: $(length(C4))")
end

if !@isdefined(C5)
    println("Building global C5 (may be large, truncating to 50000)...")
    C5_full = [(a,b,c,d,e) for (a,b,c,d) in C4 for e in basis if tgt(d) == src(e)]
    if length(C5_full) > 50000
        C5 = C5_full[1:50000]
        println("Truncated C5 from $(length(C5_full)) to 50000")
    else
        C5 = C5_full
        println("C5 size: $(length(C5))")
    end
end

if !@isdefined(C6)
    println("Building global C6 (may be huge, truncating to 100000)...")
    C6_full = [(a,b,c,d,e,f) for (a,b,c,d,e) in C5 for f in basis if tgt(e) == src(f)]
    if length(C6_full) > 100000
        C6 = C6_full[1:100000]
        println("Truncated C6 from $(length(C6_full)) to 100000")
    else
        C6 = C6_full
        println("C6 size: $(length(C6))")
    end
end

const Path = Tuple{Vararg{Symbol}}

function cluster_motifs(scored_paths, k)
    motifs = Dict{Path, Float64}()

    for (path, score) in scored_paths
        p = normalize_path(path)

        # Example: truncate to k-length motif
        if length(p) ≥ k
            motif = ntuple(i -> p[i], k)   # keeps tuple type
            motifs[motif] = get(motifs, motif, 0.0) + score
        end
    end

    return motifs
end

# 2. Combine all paths into a single list (each as a Vector{Symbol})
all_paths = generate_paths(C3, C4, C5, C6)   # defined earlier in your script

# 3. Score each path using m4, m5, m6 (global or stalk‑wise)
#    Note: score_paths expects (paths, m4, m5, m6) and returns (path, total_weight)
scored = score_paths(all_paths, m4_corrected, m5_stalk, m6_stalk)  # adjust dicts as needed

# 4. Keep top 500 paths by weight
top_paths = scored[1:min(500, end)]

# 5. Optionally compute average weight per path length
for (p, w) in top_paths
    avg = w / length(p)
    println("$(path_to_regions(p)) : total=$w, avg=$avg")
end

# Step 1: region clustering
clusters = cluster_paths_by_regions(top_paths)
#println("typeof(keys(clusters)) = ", typeof(keys(clusters)))

# Step 2: motif extraction
motifs = cluster_motifs(top_paths, 3)
#println("motifs type: ", typeof(motifs))
#println("motifs value: ", motifs)

# Step 3: sort
sorted_motifs = sort(collect(motifs), by=x->-x[2])

# Assign a cluster ID to each node (based on the first region path it belongs to)
node_cluster = Dict{Int, Int}()
# Precompute once
region_to_nodes = Dict{Symbol, Vector{Int}}()
for (nid, regs) in node_regions
    for r in regs
        push!(get!(region_to_nodes, Symbol(r), Int[]), nid)
    end
end

# Fast assignment
for (i, (region_path, _)) in enumerate(clusters)
    for reg in region_path
        for nid in get(region_to_nodes, Symbol(reg), Int[])
            node_cluster[nid] = i
        end
    end
end

# Assign motif score per node (sum of weights of all motifs containing that node)
node_motif_score = Dict{Int, Float64}()

for (motif, weight) in motifs
    for sym in motif
        r = symbol_to_region(sym)
        for nid in get(region_to_nodes, Symbol(r), Int[])
            node_motif_score[nid] = get(node_motif_score, nid, 0.0) + weight
        end
    end
end

# ------------------------------------------------------------------
# 4. (Optional) Write results to VTK for visualization
# ------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Build global m5 and m6 by merging all stalk data
# ----------------------------------------------------------------------------
m5_global = Dict{NTuple{5,Symbol}, Dict{Symbol,Float64}}()
m6_global = Dict{NTuple{6,Symbol}, Dict{Symbol,Float64}}()

for (region, data) in stalk_data
    merge!(m5_global, data[:m5])
    merge!(m6_global, data[:m6])
end

# ----------------------------------------------------------------------------
# Compute global path scores (if you have global C3..C6)
# ----------------------------------------------------------------------------
# Build global composable chains (C3..C6) from the full basis.
# (These are already defined earlier in your script: C3, C4, C5, C6)
all_paths = generate_paths(C3, C4, C5, C6)

# Score each path using the merged m4_corrected, m5_global, m6_global
scored = score_paths(all_paths, m4_corrected, m5_global, m6_global)

# Keep top 500 paths (or all, depending on memory)
top_paths = scored[1:min(500, end)]

region_scores_dict = Dict{Symbol, Float64}(region_scores)
node_score = compute_node_scores(node_regions, region_scores_dict)
add_path_scores!(node_score, node_regions, top_paths)

# Compute node scores using the region scores and node-region mapping
node_score = compute_node_scores(node_regions, region_scores_dict)

# Read the node and edge CSV files into DataFrames
nodes_df = CSV.read(NODES, DataFrame)   # NODES should be defined as the path string
edges_df = CSV.read(EDGES, DataFrame) 

function write_vtk_enhanced(nodes_df, edges_df, node_score, filename; 
    stalk_data=nothing, node_regions=nothing)
    
    open(filename, "w") do io
        # 1. HEADER
        println(io, "# vtk DataFile Version 3.0")
        println(io, "Stasheff Tower Connectome - A-infinity visualization")
        println(io, "ASCII")
        println(io, "DATASET POLYDATA")

        # 2. POINTS (Geometry)
        println(io, "POINTS $(nrow(nodes_df)) float")
        for r in eachrow(nodes_df)
            println(io, "$(r.pos_x) $(r.pos_y) $(r.pos_z)")
        end

        # 3. LINES (Connectome)
        println(io, "LINES $(nrow(edges_df)) $(3 * nrow(edges_df))")
        for r in eachrow(edges_df)
            # Find indices of node1id and node2id in nodes_df
            # Note: Assuming 'id' in nodes_df matches the ids in edges_df
            idx1 = findfirst(==(r.node1id), nodes_df.id) - 1
            idx2 = findfirst(==(r.node2id), nodes_df.id) - 1
            println(io, "2 $idx1 $idx2")
        end

        # 4. POINT DATA (The Stasheff Tower levels)
        println(io, "POINT_DATA $(nrow(nodes_df))")
        
        # Total Global Score (The "Snap" Probability)
        println(io, "SCALARS total_obstruction float 1")
        println(io, "LOOKUP_TABLE default")
        for r in eachrow(nodes_df)
            println(io, get(node_score, r.id, 0.0))
        end

        # If we have the stalk data, we can decompose the Tower
        if stalk_data !== nothing && node_regions !== nothing
            for level in ["m3", "m4", "m5", "m6"]
                println(io, "SCALARS $level float 1")
                println(io, "LOOKUP_TABLE default")
                for r in eachrow(nodes_df)
                    # Get region for this node
                    reg_sym = get(node_regions, r.id, :none)
                    val = 0.0
                    if haskey(stalk_data, reg_sym)
                        # Summing the norms of the local m_n dictionary
                        # This represents local complexity/tension at this Stasheff level
                        dict_mn = stalk_data[reg_sym][Symbol(level)]
                        val = isempty(dict_mn) ? 0.0 : sum(abs.(values(dict_mn)))
                    end
                    println(io, val)
                end
            end
        end

        # 5. CELL DATA (Edge Curvature)
        println(io, "CELL_DATA $(nrow(edges_df))")
        println(io, "SCALARS edge_curveness float 1")
        println(io, "LOOKUP_TABLE default")
        for r in eachrow(edges_df)
            println(io, r.curveness)
        end
    end
end

function write_vtk_fast(nodes_df, edges_df, node_score, filename; 
                        node_cluster=nothing, node_motif_score=nothing)
    
    # 1. Extract coordinates into a 3xN matrix (Required by WriteVTK)
    # Assuming nodes_df has columns x, y, z
    points = hcat(nodes_df.x, nodes_df.y, nodes_df.z)'
    
    # 2. Create the grid (UnstructuredGrid for nodes and edges)
    # Map edges to VTK_LINE cells
    cells = [MeshCell(VTKCellTypes.VTK_LINE, [e.from_idx, e.to_idx]) for e in eachrow(edges_df)]
    
    # 3. Initialize the VTK file (This uses binary XML by default)
    vtk_grid(filename, points, cells) do vtk
        
        # 4. Add Point Data (Optimized binary transfer)
        intensity = [get(node_score, id, 0.0) for id in nodes_df.id]
        vtk["intensity"] = intensity
        
        if node_cluster !== nothing
            clusters = [get(node_cluster, id, -1) for id in nodes_df.id]
            vtk["cluster_id"] = clusters
        end
        
        if node_motif_score !== nothing
            motifs = [get(node_motif_score, id, 0.0) for id in nodes_df.id]
            vtk["motif_score"] = motifs
        end
    end
end




function write_vtk_enhanced_fast(nodes_df, edges_df, node_score, filename; 
    stalk_data=nothing, node_regions=nothing)
    
    # 1. Faster ID Mapping (O(N) lookup)
    # Using Int32 saves significant disk space and RAM for 3.5M points
    id_to_idx = Dict{Int64, Int32}(id => Int32(i - 1) for (i, id) in enumerate(nodes_df.id))
    
    # 2. Vectorized Coordinates (3 x N Matrix)
    points = hcat(nodes_df.pos_x, nodes_df.pos_y, nodes_df.pos_z)'
    
    # 3. Fast Cell Creation
    # This avoids the ASCII "LINES" formatting that caused your error
    cells = [MeshCell(VTKCellTypes.VTK_LINE, [id_to_idx[edges_df.node1id[i]], 
                                              id_to_idx[edges_df.node2id[i]]]) 
             for i in 1:nrow(edges_df)]

    # 4. Write Binary XML (The .vtu format)
    # Use 'append=true' for raw binary stream (fastest)
    vtk_grid(filename, points, cells; append=true, ascii=false) do vtk
        
        # POINT DATA
        vtk["total_obstruction"] = [Float32(get(node_score, id, 0.0)) for id in nodes_df.id]

        # Process higher-order Stasheff levels (m3-m6)
        if stalk_data !== nothing && node_regions !== nothing
            for level in ["m3", "m4", "m5", "m6"]
                level_sym = Symbol(level)
                region_sums = Dict{Symbol, Float32}(
                    reg => isempty(data[level_sym]) ? 0.0f0 : Float32(sum(abs.(values(data[level_sym]))))
                    for (reg, data) in stalk_data
                )
                vtk[level] = [region_sums[get(node_regions, id, :none)] for id in nodes_df.id]
            end
        end

        # CELL DATA (Curvature)
        vtk["edge_curveness"] = Float32.(edges_df.curveness)
    end
end

# Optimized state-to-score for 3.5M nodes
function fast_state_to_scores(state, nodes_df, node_regions)
    # 1. Define region names locally to ensure the function is self-contained
    # Adjust this list to match your A-infinity basis exactly
    local_region_names = [:sAMY, :HPF, :BLA, :CA1sp, :HY, :LA]
    
    # 2. Map the state dict to a float array
    region_val_map = Dict(reg => get(state, Symbol("e_$reg"), 0.0) for reg in local_region_names)
    
    # 3. Pre-allocate the scores array (much faster for 3.5M nodes)
    n_nodes = nrow(nodes_df)
    scores = zeros(Float64, n_nodes)
    
    # 4. Use a standard for-loop (easier to debug and memory efficient)
    for i in 1:n_nodes
        # Use nodes_df.id[i] to avoid triggering column-searches
        node_id = nodes_df.id[i]
        
        # Get the region associated with this node ID
        # Note: if node_regions[id] returns a list, we take the first element
        if haskey(node_regions, node_id)
            reg_raw = node_regions[node_id]
            reg_sym = Symbol(isa(reg_raw, Vector) ? reg_raw[1] : reg_raw)
            scores[i] = get(region_val_map, reg_sym, 0.0)
        end
    end
    
    return scores
end

# Add the 'Residue' to the map
function dynamical_map_with_obstruction(state, mult, m4_residue)
    new_state = dynamical_map(state, mult)
    # Apply a "topological friction" based on the obstruction norm
    for k in keys(new_state)
        new_state[k] *= (1.0 - m4_residue * 0.01) 
    end
    return new_state
end

# ============================================================
# 5. Time evolution and perturbation simulation
# ============================================================
using WriteVTK   # make sure you have added WriteVTK.jl

println("\n--- Simulating time evolution using m₂ ---")

# ------------------------------------------------------------
# Helper: convert state (basis coefficients) to node scores (per node)
# ------------------------------------------------------------
function state_to_node_scores(state::Dict{Symbol,Float64},
    node_regions::Dict{Int,Vector{String}})
    max_id = maximum(keys(node_regions))
    node_scores = zeros(Float64, max_id + 1)   # allocate space for 0..max_id
    for (nid, regs) in node_regions
        idx = nid + 1   # convert 0‑based to 1‑based
        total = 0.0
        for r in regs
            e_sym = Symbol("e_$r")
            total += get(state, e_sym, 0.0)
        end
        node_scores[idx] = total
    end
    return node_scores
end

# ------------------------------------------------------------
# Dynamical map: x(t+1) = m₂(x(t), x(t))   (quadratic)
# ------------------------------------------------------------
function dynamical_map(state::Dict{Symbol,Float64}, 
                       mult::Function)::Dict{Symbol,Float64}
    new_state = Dict{Symbol,Float64}()
    for (a, va) in state
        for (b, vb) in state
            prod = mult(a, b)   # returns Dict{Symbol,Float64}
            for (c, vc) in prod
                new_state[c] = get(new_state, c, 0.0) + va * vb * vc
            end
        end
    end
    return new_state
end

# ------------------------------------------------------------
# Normalize state to avoid numerical explosion
# ------------------------------------------------------------
function normalize_state!(state::Dict{Symbol,Float64}, target_norm::Float64=1.0)
    nrm = sqrt(sum(v^2 for v in values(state)))
    if nrm > 0
        scale = target_norm / nrm
        for k in keys(state)
            state[k] *= scale
        end
    end
end

# ------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------
region_names = ["sAMY", "HPF", "BLA", "CA1sp", "HY", "LA"]
nt = 20                 # number of time steps
perturb_time = 10       # step at which to perturb
perturb_strength = 0.5
initial_state = Dict{Symbol,Float64}(:e_sAMY => 1.0)

# ------------------------------------------------------------
# Baseline trajectory (unperturbed)
# ------------------------------------------------------------
baseline_traj = []
let state = copy(initial_state)
    for t in 1:nt
        push!(baseline_traj, copy(state))
        state = dynamical_map(state, mult_dict)
        normalize_state!(state, 1.0)
    end
end

# ------------------------------------------------------------
# Perturbed trajectory (add e_LA at perturb_time)
# ------------------------------------------------------------
perturbed_traj = []
let state = copy(initial_state)
    for t in 1:nt
        if t == perturb_time
            state[:e_LA] = get(state, :e_LA, 0.0) + perturb_strength
        end
        push!(perturbed_traj, copy(state))
        state = dynamical_map(state, mult_dict)
        normalize_state!(state, 1.0)
    end
end

# ------------------------------------------------------------
# Plot region activities over time (using Plots.jl)
# ------------------------------------------------------------
using Plots

# Baseline plot
baseline_activities = Dict(r => Float64[] for r in region_names)
for t in 1:nt
    for r in region_names
        e_sym = Symbol("e_$r")
        push!(baseline_activities[r], get(baseline_traj[t], e_sym, 0.0))
    end
end
p_base = plot(1:nt, [baseline_activities[r] for r in region_names],
              label=region_names, title="Baseline dynamics (no perturbation)",
              xlabel="Time step", ylabel="Activity", legend=:outertopright)
savefig(p_base, "baseline_dynamics.png")

# Perturbed plot
pert_activities = Dict(r => Float64[] for r in region_names)
for t in 1:nt
    for r in region_names
        e_sym = Symbol("e_$r")
        push!(pert_activities[r], get(perturbed_traj[t], e_sym, 0.0))
    end
end
p_pert = plot(1:nt, [pert_activities[r] for r in region_names],
              label=region_names, title="Perturbed dynamics (add LA at t=10)",
              xlabel="Time step", ylabel="Activity", legend=:outertopright)
vline!([perturb_time], linestyle=:dash, color=:black, label="Perturbation")
savefig(p_pert, "perturbed_dynamics.png")

println("Plots saved: baseline_dynamics.png, perturbed_dynamics.png")

# ------------------------------------------------------------
# Write VTU files for time series (using WriteVTK.jl)
# ------------------------------------------------------------
# Prepare points and cells once (for all time steps)
points = hcat(nodes_df.pos_x, nodes_df.pos_y, nodes_df.pos_z)'
cells = [MeshCell(VTK_LINE, [edges_df.node1id[i]-1, edges_df.node2id[i]-1]) 
         for i in 1:nrow(edges_df)]
num_nodes = nrow(nodes_df)
println("Writing baseline VTU files...")
for t in 1:nt
    node_scores = state_to_node_scores(baseline_traj[t], node_regions)
    # Ensure correct length
    if length(node_scores) != nrow(nodes_df)
        node_scores = node_scores[1:nrow(nodes_df)]
    end
    # Apply log10 transform for better visualisation (avoid huge numbers)
    viewable = log10.(abs.(node_scores) .+ 1.0)
    filename = "baseline_t$(lpad(t,3,"0"))"
    vtk_grid(filename, points, cells) do vtk
        vtk["activity"] = viewable
    end
end

println("Writing perturbed VTU files...")
for t in 1:nt
    node_scores = state_to_node_scores(perturbed_traj[t], node_regions)
    # Ensure correct length
    if length(node_scores) != nrow(nodes_df)
        node_scores = node_scores[1:nrow(nodes_df)]
    end
    viewable = log10.(abs.(node_scores) .+ 1.0)
    filename = "perturbed_t$(lpad(t,3,"0"))"
    vtk_grid(filename, points, cells) do vtk
        vtk["activity"] = viewable
    end
end

# ------------------------------------------------------------
# Write static reference file with initial node scores (from region_scores)
# ------------------------------------------------------------
println("Writing static reference VTU file...")
# node_score is a Dict{Int,Float64} from earlier (region_scores mapped to nodes)
static_scores = zeros(Float64, nrow(nodes_df))
# Defensive check for the static node_score dictionary
if @isdefined(node_score) && node_score isa Dict
    for (nid, val) in node_score
        idx = nid + 1   # convert 0‑based to 1‑based
        if 1 <= idx <= length(static_scores)
            static_scores[idx] = val
        end
    end
end
# Apply log transform for visualisation
static_viewable = log10.(abs.(static_scores) .+ 1.0)

vtk_grid("static_brain_reference", points, cells) do vtk
    vtk["region_score"] = static_viewable
    vtk["degree"] = nodes_df.degree
    # add any other node attributes you have
end

println("Time evolution simulation and VTK export completed.")

# ----------------------------------------------------------------------------
# Functions to compare stalks at all levels (m3, m4, m5, m6)
# ----------------------------------------------------------------------------
function diff_dict(d1, d2)
    keys_all = union(keys(d1), keys(d2))
    out = Dict{Symbol,Float64}()
    for k in keys_all
        out[k] = get(d1, k, 0.0) - get(d2, k, 0.0)
    end
    return out
end

function norm_dict(d)
    return sqrt(sum(v^2 for v in values(d)))
end

function compare_stalk_m3(stalkA, stalkB, inter)
    defect = 0.0
    for (k, v1) in stalkA[:m3]
        if all(x -> x in inter, k)
            v2 = get(stalkB[:m3], k, Dict{Symbol,Float64}())
            defect += norm_dict(diff_dict(v1, v2))
        end
    end
    return defect
end

function compare_stalk_m4(stalkA, stalkB, inter)
    defect = 0.0
    for (k, v1) in stalkA[:m4]
        if all(x -> x in inter, k)
            v2 = get(stalkB[:m4], k, Dict{Symbol,Float64}())
            defect += norm_dict(diff_dict(v1, v2))
        end
    end
    return defect
end

function compare_stalk_m5(stalkA, stalkB, inter)
    defect = 0.0
    for (k, v1) in stalkA[:m5]
        if all(x -> x in inter, k)
            v2 = get(stalkB[:m5], k, Dict{Symbol,Float64}())
            defect += norm_dict(diff_dict(v1, v2))
        end
    end
    return defect
end

function compare_stalk_m6(stalkA, stalkB, inter)
    defect = 0.0
    for (k, v1) in stalkA[:m6]
        if all(x -> x in inter, k)
            v2 = get(stalkB[:m6], k, Dict{Symbol,Float64}())
            defect += norm_dict(diff_dict(v1, v2))
        end
    end
    return defect
end

function gluing_defect_full(stalkA, stalkB, inter)
    return (
        m3 = compare_stalk_m3(stalkA, stalkB, inter),
        m4 = compare_stalk_m4(stalkA, stalkB, inter),
        m5 = compare_stalk_m5(stalkA, stalkB, inter),
        m6 = compare_stalk_m6(stalkA, stalkB, inter),
        total = compare_stalk_m3(stalkA, stalkB, inter) +
                compare_stalk_m4(stalkA, stalkB, inter) +
                compare_stalk_m5(stalkA, stalkB, inter) +
                compare_stalk_m6(stalkA, stalkB, inter)
    )
end

println("\n=== Full gluing defects (m3+m4+m5+m6) ===")
for r1 in regions, r2 in regions
    r1 == r2 && continue
    inter = intersect(Set(stalk_data[r1][:basis]), Set(stalk_data[r2][:basis]))
    local defects = gluing_defect_full(stalk_data[r1], stalk_data[r2], inter)
    println("$r1 ↔ $r2 : m3=$(defects.m3), m4=$(defects.m4), m5=$(defects.m5), m6=$(defects.m6), total=$(defects.total)")
end

# Example: compare first common triple between sAMY and HPF
r1, r2 = "sAMY", "HPF"
inter = intersect(Set(stalk_data[r1][:basis]), Set(stalk_data[r2][:basis]))
for triple in keys(stalk_data[r1][:m3])
    if all(x -> x in inter, triple)
        println("\nExample m3 comparison for triple $triple:")
        println("  $r1 : ", stalk_data[r1][:m3][triple])
        println("  $r2 : ", stalk_data[r2][:m3][triple])
        break
    end
end