using LinearAlgebra
using Statistics
using Plots
using SparseArrays
using IterativeSolvers
using SparseArrays

NODES = 200

# === Node and Edge Structures ===
mutable struct BandHopf
    state::Float64
    ω::Float64
    r::Float64
end

struct RestrictionMap
    matrix::Matrix{Float64}   # dim_V × dim_U
end

mutable struct AUNodes
    state::Float64
    bands::Dict{Symbol, BandHopf}  # :alpha, :beta, :gamma, :theta
    c2::Float64
    region::Symbol
end

struct DGSpan
    π_U::RestrictionMap      # U <- E
    π_V::RestrictionMap      # E -> V
    weight::Float64          # half-life / concentration
    hopf_freq::Float64       # optional
end

mutable struct EdgeHopf
    weight::Float64
    delay::Float64
end

# === Initialize Nodes ===

function init_nodes(N::Int, region_map::Dict{Symbol, Vector{Int}})
    nodes = Vector{AUNodes}(undef, N)

    # Create a reverse map from node index → region symbol
    node_to_region = Dict{Int, Symbol}()
    for (region, idxs) in region_map
        for i in idxs
            node_to_region[i] = region
        end
    end

    for i in 1:N
        bands = Dict(
            :alpha => BandHopf(rand(), 10.0, 1.0),
            :beta  => BandHopf(rand(), 20.0, 1.0),
            :gamma => BandHopf(rand(), 40.0, 1.0),
            :theta => BandHopf(rand(), 5.0, 1.0)
        )
        state = rand()
        c2 = rand()
        region = node_to_region[i]  # assign sequentially from region_map
        nodes[i] = AUNodes(state, bands, c2, region)
    end

    return nodes
end

# === Initialize Edges ===
function init_edges(N, avg_deg)
    edges = Dict{Tuple{Int, Int}, EdgeHopf}()
    for i in 1:N
        for _ in 1:avg_deg
            target = rand(1:N)
            if target != i
                edges[(i,target)] = EdgeHopf(rand(), rand())
            end
        end
    end
    return edges
end

# === Pruning to HH2 aware trimming === #
function extremal_rays(nodes, edges; bands=[:alpha,:beta,:gamma,:theta], τ=0.5)
    rays = Set{Tuple{Int,Int}}()

    for ((i,j), e1) in edges
        for ((j2,k), e2) in edges
            j == j2 || continue

            for b in bands
                xi = nodes[i].bands[b].state
                xj = nodes[j].bands[b].state
                xk = nodes[k].bands[b].state

                φ = abs(xk - xj) * e1.weight * e2.weight
                if φ > τ
                    push!(rays, (i,j))
                    push!(rays, (j,k))
                end
            end
        end
    end
    return rays
end

# === Derived Contraction === #
function categorical_blowdown!(edges, extremal_edges; factor=0.1)
    for e in extremal_edges
        if haskey(edges, e)
            edges[e].weight *= factor
        end
    end
end

function kodaira_dimension(edges; ε=1e-2)
    return count(e -> e.weight > ε, values(edges))
end

function sparse_pinv(A::SparseMatrixCSC; abstol=1e-8, reltol=1e-6, maxiter=1000)
    n, m = size(A)
    X = zeros(m, n)

    for i in 1:n
        b = zeros(n); b[i] = 1.0
        x = zeros(m)
        # call cg! without 'tol'
        cg!(x, A, b; abstol=abstol, reltol=reltol, maxiter=maxiter, initially_zero=true)
        X[:,i] = x
    end
    return sparse(X)
end

function compute_cohomology_transfer(ρ::RestrictionMap,
    HH_domain::Dict{Int,Array{Float64}},
    HH_codomain::Dict{Int,Array{Float64}})
    transfer_maps = Dict{Int,Matrix{Float64}}()
    extension_maps = Dict{Int,Matrix{Float64}}()

    for k in 0:3
        if haskey(HH_domain, k) && haskey(HH_codomain, k)
            M = ρ.matrix
            if k == 0
                ρ_k = M
            elseif k == 1
                ρ_k = kron(M, M)
            elseif k == 2
                d = size(M,1)
                # reduce to manage memory explosion.
                small_dim = min(d, 5)
                ρ_k = kron(sparse(M[1:small_dim,1:small_dim]), sparse(M[1:small_dim,1:small_dim]))
                # dim_E = 300, that’s 300⁴ = 8.1×10⁹ entries → ~64 GB if dense.
                #ρ_k = kron(kron(M, M), kron(M, M))
            elseif k == 3
                #ρ_k = kron(kron(kron(M, M), M), M)
                ρ_k = zeros(1,1)
            end
            transfer_maps[k] = ρ_k
            # Choose pseudo-inverse based on type
            if issparse(ρ_k)
                extension_maps[k] = sparse_pinv(ρ_k)
            else
                extension_maps[k] = pinv(ρ_k)
            end
        end
    end

    return transfer_maps, extension_maps
end

function compute_span_HH_transfer(span::DGSpan,
    HH_U::Dict{Int,Array{Float64}},
    HH_E::Dict{Int,Array{Float64}},
    HH_V::Dict{Int,Array{Float64}})
    # U -> E
    ρ_UE, _ = compute_cohomology_transfer(span.π_U, HH_U, HH_E)

    # E -> V
    ρ_EV, _ = compute_cohomology_transfer(span.π_V, HH_E, HH_V)

    # Compose with weighting
    span_transfer = Dict{Int,Matrix{Float64}}()

    for k in keys(ρ_UE)
        if haskey(ρ_EV, k)
            span_transfer[k] =
                span.weight * (ρ_EV[k] * ρ_UE[k])
        end
    end

    return span_transfer
end
# Force T_span to match p_direct[k]
function merge_transfers!(ρ_direct, span_transfer)
    for (k, T_span) in span_transfer
        if haskey(ρ_direct, k)
            # resize if needed
            if size(ρ_direct[k]) != size(T_span)
                min_rows = min(size(ρ_direct[k],1), size(T_span,1))
                min_cols = min(size(ρ_direct[k],2), size(T_span,2))
                ρ_direct[k][1:min_rows, 1:min_cols] .+= T_span[1:min_rows, 1:min_cols]
            else
                ρ_direct[k] .+= T_span
            end
        else
            ρ_direct[k] = T_span
        end
    end
end

function prune_spans!(spans::Vector{DGSpan},
    HH_U::Dict{Int,Array{Float64}},
    HH_Es::Vector{Dict{Int,Array{Float64}}},
    HH_V::Dict{Int,Array{Float64}};
    ε::Float64=1e-4,
    δ::Float64=1e-6)

    keep = BitVector(undef, length(spans))

    for i in eachindex(spans)
        span = spans[i]
        HH_E = HH_Es[i]

        # quick weight test
        if span.weight < ε
            keep[i] = false
            continue
        end

        span_transfer = compute_span_HH_transfer(span, HH_U, HH_E, HH_V)

        # check if span contributes anything meaningful
        survives = false
        for (k, T) in span_transfer
            if norm(T) ≥ δ
                survives = true
                break
            end
        end

        keep[i] = survives
    end

    # destructive prune
    spans[:] = spans[keep]
    HH_Es[:] = HH_Es[keep]

    return nothing
end

# === Hopf Update per Node per Band ===
function update_hopf!(
    band::BandHopf,
    bandname::Symbol,
    node_idx::Int,
    edges::Dict{Tuple{Int,Int},EdgeHopf},
    nodes::Vector{AUNodes};
    dt = 0.01
)
    # --- coupling only from SAME band ---
    coupling = 0.0
    for ((src, dst), e) in edges
        if dst == node_idx
            coupling += e.weight * nodes[src].bands[bandname].state
        end
    end

    # --- Hopf normal form (amplitude only) ---
    x = band.state
    dx = x * (band.r - x^2) + 0.05 * coupling

    band.state += dt * dx

    # --- hard safety clamp ---
    band.state = clamp(band.state, -5.0, 5.0)
end
#=
function update_hopf!(band::BandHopf, node_idx::Int, edges::Dict{Tuple{Int,Int},EdgeHopf}, nodes::Vector{AUNodes}; dt=0.01)
    # Simple coupling: sum over incoming edges for this band
    coupling = 0.0
    for (src, dst) in keys(edges)
        if dst == node_idx
            coupling += edges[(src,dst)].weight * nodes[src].bands[:alpha].state
        end
    end

    # Euler step with small dt
    band.state += dt * (band.state*(band.r - band.state^2) + 0.01*coupling)
    # ω should rotate phase if you want oscillatory dynamics; for 1D amplitude only, skip adding ω directly
end
=#

# === Prolate Observer ===
function prolate_observer(nodes, edges)
    N = length(nodes)
    M = min(N, 50)
    diag = rand(M)
    offdiag = rand(M-1)
    
    # Construct a tridiagonal matrix
    T = Tridiagonal(offdiag, diag, offdiag)
    eigvals = eigen(T).values
    return eigvals
end


# === Gradual Region Pruning ===
function gradual_region_prune!(nodes::Vector{AUNodes}, edges::Dict{Tuple{Int,Int},EdgeHopf},
    region_map::Dict{Symbol,Vector{Int}}, min_fraction::Float64)
    for (region, idxs) in region_map
        # Only keep indices that still exist in nodes
        valid_idxs = filter(i -> i <= length(nodes), idxs)
        if isempty(valid_idxs)
            continue  # skip this region if no nodes left
        end

        N_keep = max(1, Int(floor(length(valid_idxs) * min_fraction)))
        scores = [nodes[i].c2 for i in valid_idxs]
        sorted_idxs = sortperm(scores; rev=true)
        keep_idxs = valid_idxs[sorted_idxs[1:min(N_keep, length(sorted_idxs))]]

        # Remove nodes not kept
        remove_idxs = setdiff(valid_idxs, keep_idxs)
        for r in sort(remove_idxs, rev=true)
            if r <= length(nodes)  # safety check
                deleteat!(nodes, r)
                # remove associated edges
                for k in keys(edges)
                    if k[1] == r || k[2] == r
                        delete!(edges, k)
                    end
                end
            end
        end
    end
end

"""
compute_HH(nodes, region_key)

Compute HH^0 to HH^3 for a set of nodes in a given region.

Returns:
    Dict{Int, Array{Float64}} mapping degree → cochain array.
"""
function compute_HH(nodes, region_key)
    # select nodes in region
    region_nodes = filter(n -> n.region == region_key, nodes)

    # placeholder dimensions
    dim = length(region_nodes)
    HH = Dict{Int, Array{Float64}}()
    if dim == 0
        # no nodes in this region → return empty arrays
        for k in 0:3
            HH[k] = zeros(0,0)
        end
        return HH
    end

    # HH^0: center elements (vector)
    HH[0] = sparse(randn(dim, 1))

    # HH^1: derivations (dim × dim matrix)
    HH[1] = sparse(randn(dim, dim))

    # HH^2: associators (4-tensor flattened to dim^2 × dim^2)
    #HH[2] = randn(dim^2, dim^2)

    # HH^3: higher operations (8-tensor flattened)
    #HH[3] = randn(dim^4, dim^4)
    # Reduce computation by approximation.
    small_dim = min(dim, 5)
    HH[2] = sprandn(small_dim, small_dim, 0.1)

    #HH[2] = randn(dim^2 ÷ 100, dim^2 ÷ 100)  # downsample
    # dim_E^4 ÷ 10_000 ≈ 8 × 10^7 → 80 million × 80 million float64 — impossible, that’s ~50+ TB.
    HH[3] = zeros(1,1) # randn(max(1, dim^4 ÷ 10_000), max(1, dim^4 ÷ 10_000))  # downsample HH3

    return HH
end

# === Dec 23, 2025 === #
# Build adjacency lists once per timestep
function build_in_edges(edges)
    in_edges = Dict{Int, Vector{Int}}()
    for (i,j) in keys(edges)
        push!(get!(in_edges, j, Int[]), i)
    end
    return in_edges
end

"""
Compute local Hochschild 2-cocycles φ_ijk^(b)
Returns a vector of (t, i, j, k, band, value)
"""
function compute_HH2_cocycles(nodes, edges; bands=[:alpha,:beta,:gamma,:theta])
    in_edges = build_in_edges(edges)
    cocycles = Float64[]

    for j in keys(in_edges)
        for i in in_edges[j]
            for k in get(in_edges, j, Int[])
                # require composable i → j → k
                if haskey(edges, (i,j)) && haskey(edges, (j,k))
                    w = edges[(i,j)].weight * edges[(j,k)].weight
                    for b in bands
                        xi = nodes[i].bands[b].state
                        xj = nodes[j].bands[b].state
                        xk = nodes[k].bands[b].state
                        push!(cocycles, abs(xk - xj) * w)
                    end
                end
            end
        end
    end
    return cocycles
end

"""
Effective Kodaira dimension proxy:
number of low-obstruction directions
"""
#function kodaira_dimension(cocycles; ε=1e-2)
#    return count(<(ε), cocycles)
#end
#function kodaira_dimension(edges; ε=1e-2)
#    return count(e -> e.weight > ε, values(edges))
#end
function kodaira_dimension(edges; ε=1e-2)
    return count(values(edges)) do w
        w isa EdgeHopf ? w.weight > ε : w > ε
    end
end

"""
Local effective Kodaira dimension κᵢ
derived from stalk-level HH² cocycle mass.

- NO global aggregation
- NO thresholds over all edges
- Depends only on node i and its incident structure
"""
function local_kodaira_dimension(
    i::Int,
    nodes::Vector{AUNodes},
    edges::Dict{Tuple{Int,Int},EdgeHopf};
    ε::Float64 = 1e-12
)
    node = nodes[i]

    # --- collect true local neighborhood (stalk support) ---
    nbrs = Int[]
    for (u, v) in keys(edges)
        if u == i
            push!(nbrs, v)
        elseif v == i
            push!(nbrs, u)
        end
    end

    # Fewer than 2 neighbors ⇒ no composable 2-simplices
    length(nbrs) < 2 && return -Inf

    # --- local HH² mass (associativity obstruction) ---
    hh2_mass = 0.0

    for a in 1:length(nbrs)
        for b in a+1:length(nbrs)
            j = nbrs[a]
            k = nbrs[b]

            eij = get(edges, (i, j), nothing)
            eik = get(edges, (i, k), nothing)

            # HH² only exists where both morphisms exist
            (eij === nothing || eik === nothing) && continue

            # band interaction = local nonlinearity (sheaf stalk data)
            band_mismatch = 0.0
            for band in values(node.bands)
                band_mismatch += abs(band.state) * band.r
            end

            # HH² cocycle contribution
            hh2_mass += abs(eij.weight * eik.weight) * band_mismatch
        end
    end

    # Kodaira dimension = logarithmic growth of HH² mass
    return log(hh2_mass + ε)
end


"""
Compute κᵢ(t) for all nodes, grouped by region.
Pure stalk evaluation; regions are just index sets.
"""
function compute_kappa_by_region(
    nodes::Vector{AUNodes},
    edges::Dict{Tuple{Int,Int},EdgeHopf},
    region_map::Dict{Symbol,Vector{Int}}
)
    κ = Dict{Symbol,Vector{Float64}}()

    for (region, idxs) in region_map
        κ[region] = Float64[
            local_kodaira_dimension(i, nodes, edges)
            for i in idxs
        ]
    end

    return κ
end

"""
Check if stalk i glues with neighbor j.
Failure means a jump locus.
"""
function gluable(
    κi::Float64,
    κj::Float64;
    tol = 0.5
)
    # logarithmic scale → additive comparison
    return abs(κi - κj) ≤ tol
end

"""
Detect jump loci as stalks failing gluing conditions.
Returns indices of non-gluable stalks.
"""
function detect_jump_loci(
    region_nodes::Vector{Int},
    κvals::Vector{Float64};
    τ::Float64 = 1.0
)
    n = length(κvals)
    n < 3 && return Int[]

    jumps = Int[]

    for i in 2:(n-1)
        κ_prev = κvals[i-1]
        κ_curr = κvals[i]
        κ_next = κvals[i+1]

        # discrete curvature / second difference
        Δ2 = κ_next - 2κ_curr + κ_prev

        if abs(Δ2) > τ
            push!(jumps, region_nodes[i])
        end
    end

    return jumps
end


"""
Prolate gap derived as colimit obstruction.
Uses only gluable stalks.
"""
function prolate_gap_from_colimit(
    κvals::Vector{Float64},
    jump_loci::Vector{Int}
)
    if isempty(κvals)
        return 0.0
    end

    # exclude non-gluable stalks
    good = setdiff(1:length(κvals), jump_loci)

    if isempty(good)
        return maximum(κvals)  # totally obstructed
    end

    κ_good = κvals[good]

    # colimit obstruction = spread of compatible deformations
    return maximum(κ_good) - minimum(κ_good)
end

# === Example Restriction Map Generator ===
function restriction_map(region::Symbol; dim_U=10, dim_V=10)
    # For simplicity, random linear map scaled by 0.1–1.0
    M = 0.1 .+ 0.9*rand(dim_V, dim_U)
    return RestrictionMap(M)
end

# === Initialize DGSpans ===
function init_spans(N_spans, HH_U, HH_V)
    spans = Vector{DGSpan}(undef, N_spans)
    HH_Es = Vector{Dict{Int, Array{Float64}}}(undef, N_spans)

    for i in 1:N_spans
        dim_E = size(HH_U[0], 1)
        HH_E = Dict{Int, Array{Float64}}()

        # k=0,1 safe
        HH_E[0] = randn(dim_E,1)
        HH_E[1] = randn(dim_E, dim_E)

        # k=2 downsample sparsely
        HH_E[2] = sprandn(max(1, dim_E^2 ÷ 100), max(1, dim_E^2 ÷ 100), 0.01)

        # k=3 placeholder
        HH_E[3] = zeros(1,1)

        HH_Es[i] = HH_E

        π_U = RestrictionMap(randn(dim_E, size(HH_U[0],1)))
        π_V = RestrictionMap(randn(size(HH_V[0],1), dim_E))
        weight = rand()
        spans[i] = DGSpan(π_U, π_V, weight, 0.0)
    end

    return spans, HH_Es
end

"""
Detect jump times in Kodaira dimension
"""
function detect_kodaira_jumps(kappa_hist; tol=1e-6)
    jumps = Int[]
    for t in 2:length(kappa_hist)
        if abs(kappa_hist[t] - kappa_hist[t-1]) > tol
            push!(jumps, t)
        end
    end
    return jumps
end

kappa_hist = Int[]
hh2_mass   = Float64[]
# === Example Region Map ===


region_map = Dict(
    :PFC => collect(1:floor(Int, 0.3*NODES)),
    :BG => collect(floor(Int,0.3*NODES)+1:floor(Int,0.6*NODES)),
    :Amygdala => collect(floor(Int,0.6*NODES)+1:NODES)
)
# [local evolution] → [simultaneous snapshot] → [sheaf logic]
const N_spans = 10  # moderate number for memory control
# === Simulation Loop ===
# === Simulation Loop ===
function run_simulation(N, avg_deg, region_map, tmax)

    nodes = init_nodes(N, region_map)
    edges = init_edges(N, avg_deg)

    # --- Histories ---
    alpha_hist   = zeros(tmax)
    beta_hist    = zeros(tmax)
    gamma_hist   = zeros(tmax)
    theta_hist   = zeros(tmax)

    kappa_hist   = Float64[]
    prolate_hist = Float64[]

    hh2_direct_hist = Float64[]
    hh2_span_hist   = Float64[]
    hh2_total_hist  = Float64[]

    for t in 1:tmax

        # --------------------------------------------------
        # 1. Local stalk evolution (Hopf dynamics only)
        # --------------------------------------------------
        for (i, n) in enumerate(nodes)
            for (bandname, band) in n.bands
                update_hopf!(band, bandname, i, edges, nodes)
            end
        end

        # --------------------------------------------------
        # 2. Stalkwise sheaf invariants (HH² → κᵢ)
        # --------------------------------------------------
        κ_by_region = compute_kappa_by_region(nodes, edges, region_map)

        region_PFC = region_map[:PFC]
        κvals_PFC = κ_by_region[:PFC]
        jumps_PFC = detect_jump_loci(region_PFC, κvals_PFC)
        # detect categorical jump loci
        jump_loci  = detect_kodaira_jumps(κvals_PFC)

        κ_eff = mean(κvals_PFC)               # regional effective κ
        gap   = prolate_gap_from_colimit(κvals_PFC, jumps_PFC)

        push!(kappa_hist, κ_eff)
        push!(prolate_hist, gap)

        #=
        # ==================================================
        # 3. HH transfer (direct + span-induced)
        # ==================================================
        HH_U = compute_HH(nodes, :PFC)      # source
        HH_V = HH_U                         # colimit target (regional)

        # --- Initialize spans and HH_Es ---
        spans, HH_Es = init_spans(N_spans, HH_U, HH_V)

        # direct restriction
        ρ_direct, _ = compute_cohomology_transfer(
            restriction_map(:PFC), HH_U, HH_V
        )

        # --- Measure HH² transfer strength ---
        direct_norm = haskey(ρ_direct, 2) ? norm(ρ_direct[2]) : 0.0

        span_norm = 0.0

        # span-induced contributions
        for i in eachindex(spans)
            span_transfer = compute_span_HH_transfer(
                spans[i], HH_U, HH_Es[i], HH_V
            )
            if haskey(span_transfer, 2)
                span_norm += norm(span_transfer[2])
            end
            merge_transfers!(ρ_direct, span_transfer)
        end
        push!(hh2_direct_hist, direct_norm)
        push!(hh2_span_hist, span_norm)
        push!(hh2_total_hist, direct_norm + span_norm)
        =#
        # --------------------------------------------------
        # 4. Observable summaries (purely diagnostic)
        # --------------------------------------------------
        alpha_hist[t] = mean(n.bands[:alpha].state for n in nodes)
        beta_hist[t]  = mean(n.bands[:beta].state  for n in nodes)
        gamma_hist[t] = mean(n.bands[:gamma].state for n in nodes)
        theta_hist[t] = mean(n.bands[:theta].state for n in nodes)

        # --------------------------------------------------
        # 5. Derived MMP step (categorical blow-down)
        # --------------------------------------------------
        rays = extremal_rays(nodes, edges; τ = 0.5)
        categorical_blowdown!(edges, rays)

        # ==================================================
        # 6. **Span blow-down (NEW — integrated from item 6)**
        # ==================================================
        #=
        prune_spans!(
            spans,
            HH_U,
            HH_Es,
            HH_V;
            ε = 1e-4,    # chemical half-life cutoff
            δ = 1e-6     # HH transfer significance
        )
        =#
        # ==================================================
        # Logging
        # ==================================================
        #=
        println(
            "t=$t | κ_eff=$(round(κ_eff, digits=3)) " *
            "| gap=$(round(gap, digits=3)) " *
            "| edges=$(length(edges)) " *
            "| spans=$(length(spans))"
        )
        =#
        println("t=$t | κ_eff=$(round(κ_eff, digits=3)) | gap=$(round(gap, digits=3)) | edges=$(length(edges))")
    end

    # ------------------------------------------------------
    # 5. Detect Kodaira jump loci (event-level)
    # ------------------------------------------------------
    jumps = detect_kodaira_jumps(kappa_hist)

    # ------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------
    t_vals = 1:tmax

    plot(t_vals, alpha_hist, label="Alpha")
    plot!(t_vals, beta_hist,  label="Beta")
    plot!(t_vals, gamma_hist, label="Gamma")
    plot!(t_vals, theta_hist, label="Theta")
    plot!(t_vals, prolate_hist, label="Prolate Gap (Derived colimit)", lw=2, ls=:dash)

    xlabel!("Time")
    ylabel!("Amplitude / Colimit Invariant")
    title!("Stalkwise Oscillations with Derived Colimit Collapse")
    savefig("simulation_output.png")

    plot(eachindex(kappa_hist), kappa_hist;
         lw = 2,
         label = "Effective Kodaira Dimension",
         xlabel = "Event Index",
         ylabel = "κ_eff")

    scatter!(jumps, kappa_hist[jumps];
             ms = 6,
             label = "Kodaira Jump Loci")

    savefig("simulation_output_jl.png")

    # === HH² Transfer Strength ===
    plot(t_vals, hh2_direct_hist;
    lw = 2,
    label = "HH² Direct Restriction")

    plot!(t_vals, hh2_span_hist;
    lw = 2,
    ls = :dot,
    label = "HH² Span-Induced (Chemical Access)")

    plot!(t_vals, hh2_total_hist;
    lw = 3,
    ls = :dash,
    label = "HH² Total Transfer")

    xlabel!("Time")
    ylabel!("‖HH² Transfer‖")
    title!("Derived Deformation Flow and Chemical Connectivity")
    savefig("simulation_HH_transfer.png")

    return (
        kappa_hist   = kappa_hist,
        prolate_hist = prolate_hist,
        alpha_hist   = alpha_hist,
        beta_hist    = beta_hist,
        gamma_hist   = gamma_hist,
        theta_hist   = theta_hist,
        hh2_direct_hist = hh2_direct_hist,
        hh2_span_hist   = hh2_span_hist,
        hh2_total_hist  = hh2_total_hist
    )

end




# Run Simulation
run_simulation(NODES, 5, region_map, 20)
