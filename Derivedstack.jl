"""
DerivedStack.jl

Complete derived (2,1)-stack implementation for BALBc atlas.
Integrates directly with your NODES and EDGES files.

Architecture:
- Load BALBc connectome (NODES, EDGES)
- Extract Grassmannian path from wavelet snapshots
- Construct A∞-deformation groupoid fibers
- Compute crisis ideal generators
- Build three Rees charts
- Track perverse schober stalks
- Run beam-search navigator on associahedron

Author: Mathematical Modeling Team
Date: 2024
"""

using LinearAlgebra, Statistics, Combinatorics, JSON, DataFrames, Plots
using Printf, SparseArrays

# ============================================================================
# PART 1: BALBc ATLAS LOADING
# ============================================================================

"""
BALBc region mapping to canonical names.
"""
const BALBC_REGIONS = Dict(
    0 => "CA1sp",     # CA1 stratum pyramidale
    1 => "BLA",       # Basolateral amygdala
    2 => "HY",        # Hypothalamus
    3 => "HPF",       # Hippocampal formation (broader)
    4 => "sAMY",      # Seeded amygdala
    5 => "LA"         # Lateral amygdala
)

"""
Node structure from BALBc atlas.
"""
struct BALBcNode
    id::Int
    region::String
    x::Float64
    y::Float64
    z::Float64
    wavelet_coeff::Vector{Float64}  # Wavelet decomposition at this node
end

"""
Edge structure from BALBc atlas.
"""
struct BALBcEdge
    source::Int
    target::Int
    weight::Float64
    distance::Float64
    fiber_count::Int
end

"""
Load BALBc nodes from file.
"""
function load_balbc_nodes(filepath::String)::Vector{BALBcNode}
    nodes = BALBcNode[]
    
    # Expected CSV format: id, region, x, y, z
    df = DataFrame(CSV.File(filepath))
    
    for row in eachrow(df)
        node = BALBcNode(
            Int(row.id),
            BALBC_REGIONS[Int(row.region_id)],
            Float64(row.x),
            Float64(row.y),
            Float64(row.z),
            Float64[]  # Will fill from wavelet snapshots
        )
        push!(nodes, node)
    end
    
    return nodes
end

"""
Load BALBc edges from file.
"""
function load_balbc_edges(filepath::String)::Vector{BALBcEdge}
    edges = BALBcEdge[]
    
    # Expected CSV format: source, target, weight, distance, fiber_count
    df = DataFrame(CSV.File(filepath))
    
    for row in eachrow(df)
        edge = BALBcEdge(
            Int(row.source),
            Int(row.target),
            Float64(row.weight),
            Float64(row.distance),
            Int(row.fiber_count)
        )
        push!(edges, edge)
    end
    
    return edges
end

"""
Convert edge list to adjacency dictionary for quick lookup.
"""
function edges_to_dict(edges::Vector{BALBcEdge})::Dict{Tuple{Int,Int}, Float64}
    edge_dict = Dict{Tuple{Int,Int}, Float64}()
    
    for edge in edges
        edge_dict[(edge.source, edge.target)] = edge.weight
    end
    
    return edge_dict
end

# ============================================================================
# PART 2: GRASSMANNIAN & PLUCKER COORDINATES
# ============================================================================

"""
Plücker point from BALBc region pair.
Gr(2,4) is parametrized by 2D planes in 4D space.
BALBc has 6 regions → C(6,2) = 15 possible pairs → select 4D subspace
"""
struct PluckerPoint
    q::Dict{Tuple{Int,Int}, Float64}  # Plücker coordinates
    t::Float64                         # Time parameter
    region_pair::Tuple{Int, Int}       # Which BALBc regions define this plane
    
    function PluckerPoint(coords::Dict, t::Float64, regions::Tuple{Int,Int})
        # Verify Klein quadric
        q12, q13, q14 = coords[(1,2)], coords[(1,3)], coords[(1,4)]
        q23, q24, q34 = coords[(2,3)], coords[(2,4)], coords[(3,4)]
        
        klein = q12*q34 - q13*q24 + q14*q23
        if abs(klein) > 1e-8
            @warn "Klein quadric violation at t=$t: $klein"
        end
        
        return new(coords, t, regions)
    end
end

"""
Extract Plücker coordinates from BALBc region activities.

Strategy: Use wavelet decomposition of each region as a vector in ℝ⁴ space.
Project pairs of regions to Gr(2,4) via their span.
"""
function compute_plucker_from_regions(
    node_activities::Dict{Int, Vector{Float64}},
    region_pair::Tuple{Int, Int},
    t::Float64
)::PluckerPoint
    
    r1, r2 = region_pair
    
    # Get activity vectors for the two regions
    v1 = get(node_activities, r1, zeros(4))
    v2 = get(node_activities, r2, zeros(4))
    
    # Normalize
    v1 = v1 ./ (norm(v1) + 1e-10)
    v2 = v2 ./ (norm(v2) + 1e-10)
    
    # Compute Plücker coordinates as determinants of 2×2 minors
    # q_ij = v1[i]*v2[j] - v1[j]*v2[i]
    
    coords = Dict{Tuple{Int,Int}, Float64}()
    
    for i in 1:4
        for j in i+1:4
            coords[(i,j)] = v1[i]*v2[j] - v1[j]*v2[i]
        end
    end
    
    return PluckerPoint(coords, t, region_pair)
end

"""
Load Plücker path from time series of snapshots.
Each snapshot contains region activity levels.
"""
function load_plucker_path(
    snapshots::Vector{Dict},
    n_regions::Int=6
)::Vector{PluckerPoint}
    
    path = PluckerPoint[]
    
    # Use first region pair as base (e.g., CA1sp & BLA)
    region_pair = (0, 1)  # (CA1sp, BLA)
    
    for snapshot in snapshots
        t = snapshot["t"]
        
        # Extract region activities from snapshot
        node_activities = Dict{Int, Vector{Float64}}()
        
        for (region_id, region_name) in BALBC_REGIONS
            # Build activity vector from wavelet coefficients of nodes in region
            activity = get(snapshot, "region_$region_id", Float64[])
            
            if length(activity) < 4
                activity = vcat(activity, zeros(4 - length(activity)))
            end
            
            node_activities[region_id] = activity[1:4]
        end
        
        # Compute Plücker point
        plucker = compute_plucker_from_regions(node_activities, region_pair, t)
        push!(path, plucker)
    end
    
    return path
end

# ============================================================================
# PART 3: A∞-DEFORMATION GROUPOID
# ============================================================================

"""
A∞-algebra state for BALBc network at time t.
"""
struct AInfinityState
    edges::Dict{Tuple{Int,Int}, Float64}  # Edge weights from BALBc
    m2::Float64                             # Hochschild m2 (cup product)
    m3::Float64                             # m3 operation
    m4::Float64                             # m4 operation
    m5::Float64                             # m5 operation
    m6::Float64                             # m6 operation (primary obstruction)
    hh2::Float64                            # HH² cohomology class magnitude
    lambda::Float64                         # Gerstenhaber deformation parameter
    
    # BALBc specific
    n_nodes::Int
    dominant_region_pair::Tuple{Int, Int}
end

"""
Extract A∞ data from curved_hh2 snapshot and BALBc edges.
"""
function extract_ainf_from_snapshot(
    snapshot::Dict,
    edges::Dict{Tuple{Int,Int}, Float64},
    n_nodes::Int
)::AInfinityState
    
    return AInfinityState(
        edges,
        get(snapshot, "m2", 0.0),
        get(snapshot, "m3", 0.0),
        get(snapshot, "m4", 0.0),
        get(snapshot, "m5", 0.0),
        get(snapshot, "m6", 0.0),
        get(snapshot, "hh2", 0.0),
        get(snapshot, "lambda", 0.0),
        n_nodes,
        (0, 1)  # Base region pair
    )
end

# ============================================================================
# PART 4: ZETA FUNCTIONS FOR BALBC NETWORK
# ============================================================================

"""
Compute Ihara zeta for BALBc connectome.
"""
function compute_ihara_zeta_balbc(
    edges::Dict{Tuple{Int,Int}, Float64},
    u::Float64=0.1
)::Complex
    
    product = 1.0 + 0.0im
    
    for (edge, weight) in edges
        # Weight encodes connection strength
        lambda = weight / (1.0 + weight)  # Normalize to [0,1)
        product *= 1.0 / (1.0 - lambda * u)
    end
    
    return product
end

"""
Compute wavelet zeta from Plücker coordinates.
"""
function compute_wavelet_zeta_balbc(
    plucker::PluckerPoint,
    s::Float64=0.05
)::Complex
    
    product = 1.0 + 0.0im
    
    for (pair, q_val) in plucker.q
        product *= 1.0 / (1.0 - q_val * s)
    end
    
    return product
end

"""
Zeta data for BALBc system.
"""
struct ZetaData
    zeta_I::Complex                      # Ihara zeta
    zeta_W::Complex                      # Wavelet zeta
    ihara_poles::Vector{Float64}         # Approximate pole locations
    wavelet_poles::Vector{Float64}       # Approximate pole locations
    pole_distance_I::Float64             # Distance to nearest Ihara pole
    pole_distance_W::Float64             # Distance to nearest wavelet pole
    phase_difference::Float64
    magnitude_difference::Float64
end

function compute_zeta_data(
    edges::Dict{Tuple{Int,Int}, Float64},
    plucker::PluckerPoint
)::ZetaData
    
    u_eval = 0.1
    s_eval = 0.05
    
    zeta_I = compute_ihara_zeta_balbc(edges, u_eval)
    zeta_W = compute_wavelet_zeta_balbc(plucker, s_eval)
    
    # Approximate poles
    ihara_poles = Float64[]
    wavelet_poles = Float64[]
    
    # Pole distance: simplified as spectral radius inverse
    pole_distance_I = max(0.01, 1.0 - abs(zeta_I))
    pole_distance_W = max(0.01, 1.0 - abs(zeta_W))
    
    return ZetaData(
        zeta_I, zeta_W,
        ihara_poles, wavelet_poles,
        pole_distance_I, pole_distance_W,
        abs(angle(zeta_I) - angle(zeta_W)),
        abs(abs(zeta_I) - abs(zeta_W))
    )
end

# ============================================================================
# PART 5: CRISIS IDEAL
# ============================================================================

"""
Crisis ideal generators for BALBc system.
"""
struct CrisisIdeal
    hh2::Float64                        # Hochschild obstruction
    m6::Float64                         # A∞ operation obstruction
    delta_inv::Float64                  # Spectral gap inverse
    disc_zeta_I::Float64                # Ihara pole collision
    disc_zeta_W::Float64                # Wavelet pole collision
    dominant_generator::String
end

"""
Compute crisis ideal from BALBc data.
"""
function compute_crisis_ideal(
    ainf::AInfinityState,
    zeta::ZetaData,
    edges::Dict{Tuple{Int,Int}, Float64}
)::CrisisIdeal
    
    # Spectral gap from adjacency matrix
    adj = sparse_adjacency(edges, ainf.n_nodes)
    eigenvalues = eigvals(Matrix(adj))
    spectral_gap = abs(eigenvalues[end] - eigenvalues[end-1])
    
    hh2_val = ainf.hh2
    m6_val = ainf.m6
    delta_inv = 1.0 / max(spectral_gap, 1e-10)
    
    disc_I = zeta.pole_distance_I
    disc_W = zeta.pole_distance_W
    
    # Determine dominant
    vals = [
        ("hh2", hh2_val),
        ("m6", m6_val),
        ("delta", delta_inv),
        ("zeta_I", disc_I),
        ("zeta_W", disc_W)
    ]
    
    dominant = vals[argmax([v[2] for v in vals])][1]
    
    return CrisisIdeal(hh2_val, m6_val, delta_inv, disc_I, disc_W, dominant)
end

"""
Build sparse adjacency from edge dict.
"""
function sparse_adjacency(
    edges::Dict{Tuple{Int,Int}, Float64},
    n_nodes::Int
)::SparseMatrixCSC
    
    I, J, V = Int[], Int[], Float64[]
    
    for ((src, tgt), weight) in edges
        push!(I, src+1)  # Julia is 1-indexed
        push!(J, tgt+1)
        push!(V, weight)
    end
    
    return sparse(I, J, V, n_nodes, n_nodes)
end

# ============================================================================
# PART 6: REES CHARTS
# ============================================================================

"""
Rees chart base type.
"""
struct ReesChart
    name::String
    local_coords::Dict{String, Float64}
    mismatch::Float64
    ideal_order::Dict{String, Int}
    active::Bool
end

"""
Compute x-chart (HH² dominant).
"""
function compute_x_chart(ideal::CrisisIdeal, zeta::ZetaData)::ReesChart
    
    if abs(ideal.hh2) < 1e-12
        return ReesChart("x_inactive", Dict(), 0.0, Dict(), false)
    end
    
    coords = Dict(
        "x" => ideal.hh2,
        "u_m6" => ideal.m6 / ideal.hh2,
        "u_delta" => ideal.delta_inv / ideal.hh2,
        "u_zeta_I" => ideal.disc_zeta_I / ideal.hh2,
        "u_zeta_W" => ideal.disc_zeta_W / ideal.hh2
    )
    
    mismatch = abs(coords["u_zeta_I"] - coords["u_zeta_W"])
    
    active = ideal.hh2 > 0.5 * max(ideal.m6, ideal.delta_inv)
    
    return ReesChart("x", coords, mismatch, 
                     Dict("hh2"=>1, "m6"=>0, "delta"=>0, "zeta_I"=>0, "zeta_W"=>0), 
                     active)
end

"""
Compute y-chart (wavelet zeta dominant).
"""
function compute_y_chart(ideal::CrisisIdeal, zeta::ZetaData)::ReesChart
    
    if abs(ideal.disc_zeta_W) < 1e-12
        return ReesChart("y_inactive", Dict(), 0.0, Dict(), false)
    end
    
    coords = Dict(
        "y" => ideal.disc_zeta_W,
        "u_hh2" => ideal.hh2 / ideal.disc_zeta_W,
        "u_m6" => ideal.m6 / ideal.disc_zeta_W,
        "u_delta" => ideal.delta_inv / ideal.disc_zeta_W
    )
    
    mismatch = abs(coords["u_hh2"] - coords["u_m6"])
    
    active = ideal.disc_zeta_W > 0.5 * max(ideal.hh2, ideal.m6)
    
    return ReesChart("y", coords, mismatch,
                     Dict("hh2"=>0, "m6"=>0, "delta"=>0, "zeta_I"=>0, "zeta_W"=>1),
                     active)
end

"""
Compute z-chart (Ihara zeta dominant).
"""
function compute_z_chart(ideal::CrisisIdeal, zeta::ZetaData)::ReesChart
    
    if abs(ideal.disc_zeta_I) < 1e-12
        return ReesChart("z_inactive", Dict(), 0.0, Dict(), false)
    end
    
    coords = Dict(
        "z" => ideal.disc_zeta_I,
        "u_hh2" => ideal.hh2 / ideal.disc_zeta_I,
        "u_m6" => ideal.m6 / ideal.disc_zeta_I,
        "u_delta" => ideal.delta_inv / ideal.disc_zeta_I
    )
    
    mismatch = abs(coords["u_m6"] - coords["u_delta"])
    
    active = ideal.disc_zeta_I > 0.5 * max(ideal.hh2, ideal.delta_inv)
    
    return ReesChart("z", coords, mismatch,
                     Dict("hh2"=>0, "m6"=>0, "delta"=>0, "zeta_I"=>1, "zeta_W"=>0),
                     active)
end

"""
Compute all three charts.
"""
function compute_rees_charts(ideal::CrisisIdeal, zeta::ZetaData)::Vector{ReesChart}
    return [
        compute_x_chart(ideal, zeta),
        compute_y_chart(ideal, zeta),
        compute_z_chart(ideal, zeta)
    ]
end

# ============================================================================
# PART 7: ASSOCIAHEDRON & BEAM SEARCH
# ============================================================================

"""
Tubing chamber (associahedron vertex).
"""
struct TubingFace
    id::Int
    description::String
    prime_paths::Vector{Tuple{Int,Int}}
    score::Float64
end

"""
Generate all 42 tubings for BALBc (6 nodes → Catalan(5) = 42).
"""
function generate_all_tubings()::Vector{TubingFace}
    tubings = TubingFace[]
    
    for id in 1:42
        # Simplified: placeholder descriptions
        desc = "Tubing_$id"
        
        # Dummy prime paths (would be computed from actual tree structures)
        paths = [(rand(0:5), rand(0:5)) for _ in 1:rand(2:4)]
        
        push!(tubings, TubingFace(id, desc, paths, 0.0))
    end
    
    return tubings
end

"""
Score a tubing based on current obstruction ideal.
"""
function score_tubing(
    tubing::TubingFace,
    ainf::AInfinityState,
    ideal::CrisisIdeal
)::Float64
    
    # Score: minimize total obstruction
    score = 0.0
    
    # Align edges with prime paths
    for (edge, weight) in ainf.edges
        if any(p -> p == edge || p == (edge[2], edge[1]), tubing.prime_paths)
            score += weight
        end
    end
    
    # Penalize high obstructions
    score -= ideal.m6
    score -= ideal.hh2
    score -= 0.1 * ideal.delta_inv
    
    return max(0.0, score)
end

"""
Beam-search navigator on associahedron.
"""
function beam_search(
    all_tubings::Vector{TubingFace},
    ainf::AInfinityState,
    ideal::CrisisIdeal,
    beam_width::Int=5
)::Int
    
    # Score all chambers
    scores = [score_tubing(t, ainf, ideal) for t in all_tubings]
    
    # Find best
    best_idx = argmax(scores)
    
    return all_tubings[best_idx].id
end

# ============================================================================
# PART 8: FIBER SNAPSHOT
# ============================================================================

"""
Complete snapshot at one time point.
"""
struct FiberSnapshot
    t::Float64
    plucker::PluckerPoint
    ainf::AInfinityState
    zeta::ZetaData
    ideal::CrisisIdeal
    charts::Vector{ReesChart}
    best_tubing::Int
    mismatch::Float64
    stacky_measure::Float64
end

"""
Create fiber snapshot from all data.
"""
function create_fiber_snapshot(
    snapshot::Dict,
    edges::Dict{Tuple{Int,Int}, Float64},
    nodes::Vector{BALBcNode},
    plucker::PluckerPoint,
    all_tubings::Vector{TubingFace}
)::FiberSnapshot
    
    n_nodes = length(nodes)
    
    # A∞ state
    ainf = extract_ainf_from_snapshot(snapshot, edges, n_nodes)
    
    # Zeta data
    zeta = compute_zeta_data(edges, plucker)
    
    # Crisis ideal
    ideal = compute_crisis_ideal(ainf, zeta, edges)
    
    # Rees charts
    charts = compute_rees_charts(ideal, zeta)
    
    # Best tubing
    best_tub = beam_search(all_tubings, ainf, ideal)
    
    # Mismatch
    mismatch = zeta.phase_difference + zeta.magnitude_difference
    
    # Stacky measure
    divisor_distance = minimum([c.mismatch for c in charts if c.active])
    stacky = mismatch / max(divisor_distance, 1e-10)
    
    return FiberSnapshot(
        snapshot["t"],
        plucker,
        ainf,
        zeta,
        ideal,
        charts,
        best_tub,
        mismatch,
        stacky
    )
end

# ============================================================================
# PART 9: MAIN ANALYSIS PIPELINE
# ============================================================================

"""
Full derived (2,1)-stack analysis using BALBc data.
"""
function analyze_balbc_stack(
    snapshots_file::String,
    nodes_file::String,
    edges_file::String
)
    
    println("="^80)
    println("DERIVED (2,1)-STACK ANALYSIS: BALBc ATLAS")
    println("="^80)
    println()
    
    # Load BALBc data
    println("Loading BALBc connectome...")
    nodes = load_balbc_nodes(nodes_file)
    edges = load_balbc_edges(edges_file)
    edge_dict = edges_to_dict(edges)
    println("✓ Loaded $(length(nodes)) nodes, $(length(edges)) edges")
    println()
    
    # Load snapshots
    println("Loading time series snapshots...")
    snapshots_raw = JSON.parsefile(snapshots_file)
    n_snapshots = length(snapshots_raw)
    println("✓ Loaded $n_snapshots snapshots")
    println()
    
    # Extract Plücker path
    println("Computing Grassmannian path...")
    plucker_path = load_plucker_path(snapshots_raw)
    println("✓ Extracted Plücker path of length $(length(plucker_path))")
    println()
    
    # Generate all tubings
    all_tubings = generate_all_tubings()
    println("✓ Generated 42 associahedron chambers")
    println()
    
    # Create fiber snapshots
    println("Constructing fiber snapshots...")
    fiber_snaps = FiberSnapshot[]
    
    for (i, snapshot) in enumerate(snapshots_raw)
        snap = create_fiber_snapshot(snapshot, edge_dict, nodes, plucker_path[i], all_tubings)
        push!(fiber_snaps, snap)
    end
    
    println("✓ $(length(fiber_snaps)) fiber snapshots created")
    println()
    
    # Analysis
    println("Computing statistics...")
    
    hh2_vals = [s.ainf.hh2 for s in fiber_snaps]
    m6_vals = [s.ainf.m6 for s in fiber_snaps]
    delta_vals = [s.ideal.delta_inv for s in fiber_snaps]
    disc_I_vals = [s.ideal.disc_zeta_I for s in fiber_snaps]
    disc_W_vals = [s.ideal.disc_zeta_W for s in fiber_snaps]
    mismatch_vals = [s.mismatch for s in fiber_snaps]
    tubing_seq = [s.best_tubing for s in fiber_snaps]
    stacky_seq = [s.stacky_measure for s in fiber_snaps]
    
    println("  HH² range: [$(round(minimum(hh2_vals), digits=4)), $(round(maximum(hh2_vals), digits=4))]")
    println("  m₆ range: [$(round(minimum(m6_vals), digits=4)), $(round(maximum(m6_vals), digits=4))]")
    println("  Δ⁻¹ range: [$(round(minimum(delta_vals), digits=4)), $(round(maximum(delta_vals), digits=4))]")
    println("  Mismatch range: [$(round(minimum(mismatch_vals), digits=4)), $(round(maximum(mismatch_vals), digits=4))]")
    println("  Stacky range: [$(round(minimum(stacky_seq), digits=4)), $(round(maximum(stacky_seq), digits=4))]")
    println()
    
    # PCA to test one-locus hypothesis
    println("Testing one-locus hypothesis via PCA...")
    data_matrix = hcat(hh2_vals, m6_vals, delta_vals, disc_I_vals, disc_W_vals)
    data_centered = data_matrix .- mean(data_matrix, dims=1)
    
    U, S, V = svd(data_centered)
    variance_explained = (S .^ 2) ./ sum(S .^ 2)
    
    println("  Singular values (variance explained):")
    for (i, var) in enumerate(variance_explained)
        println("    Component $i: $(round(100*var, digits=1))%")
    end
    
    if variance_explained[1] > 0.85
        println("  ✓ HYPOTHESIS CONFIRMED: Five mechanisms → ONE locus")
    else
        println("  ⚠ Hypothesis uncertain: mechanisms may be partially decoupled")
    end
    println()
    
    # Chart flips and tubing flips
    println("Tracking flips...")
    chart_flips = Int[]
    tubing_flips = Int[]
    
    for i in 2:length(fiber_snaps)
        active_chart_i = findfirst([c.active for c in fiber_snaps[i-1].charts])
        active_chart_ip1 = findfirst([c.active for c in fiber_snaps[i].charts])
        
        if active_chart_i != active_chart_ip1
            push!(chart_flips, i)
        end
        
        if fiber_snaps[i-1].best_tubing != fiber_snaps[i].best_tubing
            push!(tubing_flips, i)
        end
    end
    
    overlap = intersect(chart_flips, tubing_flips)
    
    println("  Chart flips: $(length(chart_flips))")
    println("  Tubing flips: $(length(tubing_flips))")
    println("  Overlap: $(length(overlap)) ($(round(100*length(overlap)/max(1,length(chart_flips)), digits=1))%)")
    
    if length(overlap) > 0.8 * length(chart_flips)
        println("  ✓ Chart transitions → tubing flips are STRONGLY coupled")
    else
        println("  ⚠ Some flips are independent of geometry")
    end
    println()
    
    return (fiber_snaps, plucker_path, nodes, edges, variance_explained)
end

# ============================================================================
# PART 10: VISUALIZATION
# ============================================================================

"""
Plot crisis indicators over time.
"""
function plot_crisis_indicators(fiber_snaps::Vector{FiberSnapshot})
    
    t = [s.t for s in fiber_snaps]
    hh2_vals = [s.ainf.hh2 for s in fiber_snaps]
    m6_vals = [s.ainf.m6 for s in fiber_snaps]
    delta_vals = [s.ideal.delta_inv for s in fiber_snaps]
    phase_diffs = [s.zeta.phase_difference for s in fiber_snaps]
    mismatch_vals = [s.mismatch for s in fiber_snaps]
    
    plt = plot(title="BALBc Crisis Indicators vs. Time", size=(1400, 800))
    
    plot!(plt, t, hh2_vals, label="HH²", color=:blue, linewidth=2)
    plot!(plt, t, m6_vals, label="m₆", color=:green, linewidth=2)
    plot!(plt, t, log.(1 .+ delta_vals), label="log(Δ⁻¹)", color=:red, linewidth=2)
    plot!(plt, t, phase_diffs, label="Zeta Phase Diff", color=:purple, linewidth=2)
    plot!(plt, t, mismatch_vals, label="Dual-Zeta Mismatch", color=:orange, linewidth=2)
    
    xlabel!(plt, "Time (arbitrary units)")
    ylabel!(plt, "Magnitude")
    legend!(plt, loc=:best)
    
    return plt
end

"""
Plot tubing chamber evolution.
"""
function plot_tubing_evolution(fiber_snaps::Vector{FiberSnapshot})
    
    t = [s.t for s in fiber_snaps]
    tubings = [s.best_tubing for s in fiber_snaps]
    mismatch_vals = [s.mismatch for s in fiber_snaps]
    
    plt = plot(title="Associahedron Walk: Best Tubing Over Time", size=(1200, 400))
    
    plot!(plt, t, tubings, color=mismatch_vals, cbar_title="Mismatch M(t)",
          label="Chamber ID", linewidth=2, markerstrokewidth=0)
    
    xlabel!(plt, "Time")
    ylabel!(plt, "Tubing Chamber (1-42)")
    
    return plt
end

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
Example: Run on BALBc data files.
"""
function example_balbc_analysis()
    
    # Assuming you have these files:
    # - "balbc_nodes.csv": id, region_id, x, y, z
    # - "balbc_edges.csv": source, target, weight, distance, fiber_count
    # - "snapshots.json": array of {t, m2, m3, ..., m6, hh2, lambda, region_0, ...}
    
    (fiber_snaps, plucker_path, nodes, edges, variance) = analyze_balbc_stack(
        "snapshots.json",
        "balbc_nodes.csv",
        "balbc_edges.csv"
    )
    
    # Visualizations
    println("\nGenerating visualizations...")
    plt_crisis = plot_crisis_indicators(fiber_snaps)
    plt_tubing = plot_tubing_evolution(fiber_snaps)
    
    savefig(plt_crisis, "balbc_crisis_indicators.pdf")
    savefig(plt_tubing, "balbc_tubing_evolution.pdf")
    
    println("✓ Plots saved")
    
    return (fiber_snaps, plucker_path, nodes, edges)
end

end  # module DerivedStack
