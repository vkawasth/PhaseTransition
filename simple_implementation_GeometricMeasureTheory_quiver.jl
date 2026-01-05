using LinearAlgebra
using Statistics
using SparseArrays
using IterativeSolvers
using Random
using Plots

# ------------------------
# === Node / Quiver Structures ===
# ------------------------
mutable struct MoritaAlgebra
    # Local data: (A, M, B) where M is A-B bimodule
    A::Matrix{ComplexF64}      # Endomorphism algebra at source
    B::Matrix{ComplexF64}      # Endomorphism algebra at target  
    M::Matrix{ComplexF64}      # Bimodule (stalk transport)
    
    # Derived invariants
    HH0::Float64               # Center/momentum map
    HH1::Float64               # Derivations/infinitesimal automorphisms
    HH2::Float64               # Deformations/obstructions
end

mutable struct BandHopf
    z::ComplexF64     # x + i y
    ω::Float64       # intrinsic frequency
end

mutable struct MolecularLoad
    A::Float64      # excitatory / stabilizing mass
    B::Float64      # inhibitory / destabilizing mass
end

mutable struct AUNode
    state::Float64
    bands::Dict{Symbol,BandHopf}
    c2::Float64
    region::Symbol
    # --- Missing Fields Added Below ---
    morita_data::MoritaAlgebra
    load::MolecularLoad
    is_singular::Bool
end

mutable struct MoritaEdge
    src::Int
    dst::Int
    weight::Float64
    delay::Float64
    A::SparseMatrixCSC{Float64,Int}
    B::SparseMatrixCSC{Float64,Int}
    M::SparseMatrixCSC{Float64,Int}
    load::MolecularLoad
end

mutable struct QuiverEdge
    src::Int             # Index of source AUNode
    dst::Int             # Index of target AUNode
    weight::Float64      # Synaptic/Structural coupling strength
    load::MolecularLoad  # Local molecular concentration on the edge
end
edge_stress(e::QuiverEdge) = e.load.B - e.load.A

# Quiver container
mutable struct MoritaQuiver
    nodes::Vector{AUNode}
    edges::Vector{MoritaEdge}
    adjacency::Vector{Vector{Int}}  # incoming edges indices
end

struct PoissonStalk
    position::Vector{Float64}     # Geometric position
    phase::Float64                # U(1) phase
    amplitude::Float64            # Oscillator amplitude
    momentum::Vector{Float64}     # Canonical momentum
    poisson_bracket::Matrix{Float64}  # Symplectic structure
end

# === VESSEL AS MORITA CONTEXT ===

struct VesselMorita
    id::Int
    source::PoissonStalk
    target::PoissonStalk
    bimodule::Matrix{ComplexF64}      # M: source → target
    bimodule_op::Matrix{ComplexF64}   # Mᴼᴾ: target → source
    morita_data::MoritaAlgebra
end

function rand_symplectic(n::Int)
    # Generate a random skew-symmetric matrix (Poisson tensor)
    # For n=6 (3 positions + 3 momenta)
    J = zeros(n, n)
    for i in 1:n
        for j in i+1:n
            val = randn()
            J[i,j] = val
            J[j,i] = -val
        end
    end
    # Ensure it's non-degenerate
    J += 0.1 * I  # Small regularization
    return J
end

function schur_complement(A::Matrix{Float64}, constraints::Matrix{Float64})
    # Compute Schur complement for Dirac reduction
    n = size(A, 1)
    m = size(constraints, 1)
    
    # Build extended matrix
    M = [A constraints';
         constraints zeros(m, m)]
    
    # Invert using block matrix formula
    if m > 0
        A_inv = pinv(A)
        S = constraints * A_inv * constraints'
        return A - A_inv * constraints' * pinv(S) * constraints * A_inv
    else
        return A
    end
end

function constraints(triangle::Vector{PoissonStalk})
    # Constraints for triangle reduction (closure conditions)
    n = length(triangle)
    dim = length(triangle[1].position)
    
    # Closure: sum of edge vectors = 0
    C = zeros(dim, n*dim)
    for i in 1:n
        C[:, (i-1)*dim+1:i*dim] = I(dim)
    end
    return C
end

"""
    compute_massey_product(A::Matrix{ComplexF64}, B::Matrix{ComplexF64}, M::Matrix{ComplexF64})

Compute the higher associativity obstruction (Massey triple product) for
a Morita context (A, M, B). Measures HH²-like deformations.

⟨[A], [B], [M]⟩ = AB*M - A*(B*M) + (M*A)*B
"""

function compute_massey_product(A::Matrix{ComplexF64}, 
                                B::Matrix{ComplexF64}, 
                                M::Matrix{ComplexF64})
    # Simplified Massey product for Morita context
    n = size(A, 1)
    AB = A * B
    MA = M * A
    BM = B * M
    
    # Massey triple product: ⟨[A], [B], [M]⟩
    massey = AB * M - A * BM + MA * B
    return massey
end

function compute_poisson_curvature(source::PoissonStalk, target::PoissonStalk)
    # Symplectic curvature = failure of Darboux theorem globally
    ω_source = pinv(source.poisson_bracket + 1e-8I)
    ω_target = pinv(target.poisson_bracket + 1e-8I)
    
    # Parallel transport curvature
    return norm(log(ω_source * ω_target'))
end

function poisson_bracket(stalk1::PoissonStalk, stalk2::PoissonStalk)
    # Compute Poisson bracket between two stalks
    Δx = stalk1.position - stalk2.position
    dist = norm(Δx)
    
    if dist < 1e-10
        return 0.0
    end
    
    # Symplectic form evaluated at difference
    ω1 = stalk1.poisson_bracket
    ω2 = stalk2.poisson_bracket
    ω_avg = (ω1 + ω2) / 2
    
    # Contract with position difference
    return Δx' * ω_avg * Δx / (dist^2 + 1e-8)
end

function ∂H_∂amplitude(stalk::PoissonStalk, H::Float64)
    # Derivative of Hamiltonian w.r.t amplitude
    return 2 * stalk.amplitude * H
end

function ∂H_∂phase(stalk::PoissonStalk, H::Float64)
    # Derivative of Hamiltonian w.r.t phase
    return 0.0  # Phase doesn't directly appear in simple H
end

function compute_noether_current(stalk::PoissonStalk, neighbors::Vector{PoissonStalk})
    # Compute Noether current from symmetries
    current = zeros(3)
    for nb in neighbors
        # Current proportional to phase difference
        phase_diff = nb.phase - stalk.phase
        pos_diff = nb.position - stalk.position
        current .+= sin(phase_diff) .* pos_diff
    end
    return current / (length(neighbors) + 1e-8)
end

function detect_kodaira_jumps(kappa_hist; tol=1e-6)
    jumps = Int[]
    for t in 2:length(kappa_hist)
        if abs(kappa_hist[t] - kappa_hist[t-1]) > tol
            push!(jumps, t)
        end
    end
    return jumps
end

function create_edge_morita(source::PoissonStalk, target::PoissonStalk)
    # Create a Morita algebra for an edge
    n = 3  # Dimension of simple algebra
    
    # Create random matrices for A and B (source and target algebras)
    A = randn(ComplexF64, n, n) + im * randn(n, n)
    B = randn(ComplexF64, n, n) + im * randn(n, n)
    
    # Make them approximately unitary
    A = A / norm(A)
    B = B / norm(B)
    
    # Bimodule M depends on stalk states
    phase_factor = exp(im * (target.phase - source.phase))
    amp_factor = sqrt(source.amplitude * target.amplitude)
    
    M = amp_factor * phase_factor * (A + B') / 2
    
    # Compute HH invariants
    HH0 = real(tr(A * B')) / (norm(A) * norm(B))
    
    crossed = [A M; M' B]
    HH1 = log(abs(det(crossed)) + 1e-10)
    
    massey = compute_massey_product(A, B, M)
    HH2 = norm(massey)
    
    return MoritaAlgebra(A, B, M, HH0, HH1, HH2)
end

function update_morita_from_stalks!(vessel::VesselMorita)
    # Update Morita algebra based on current stalk states
    phase_factor = exp(im * (vessel.target.phase - vessel.source.phase))
    amp_factor = sqrt(vessel.source.amplitude * vessel.target.amplitude)
    
    # Update bimodule
    vessel.bimodule .= amp_factor * phase_factor * 
                      (vessel.morita_data.A + vessel.morita_data.B') / 2
    vessel.bimodule_op .= conj(vessel.bimodule')
    
    # Recompute HH
    HH0, HH1, HH2 = compute_morita_HH(vessel)
    vessel.morita_data = MoritaAlgebra(
        vessel.morita_data.A,
        vessel.morita_data.B,
        vessel.bimodule,
        HH0, HH1, HH2
    )
end


# === MORITA ALGEBRA MASSEY PRODUCT === #

function compute_morita_HH(vessel::VesselMorita)
    A = vessel.morita_data.A
    B = vessel.morita_data.B
    M = vessel.bimodule
    
    # HH⁰ = Center of algebra (Casimirs)
    # HH⁰ = Center of algebra 
    HH0 = real(tr(A * B')) / (norm(A) * norm(B))
    
    # HH¹ = Outer derivations (infinitesimal symmetries)
    crossed = [A M; M' B]
    HH1 = log(abs(det(crossed)) + 1e-10)
    
    # HH² = Deformations (Poisson obstruction)
    HH2 = norm(compute_massey_product(A, B, M))
    
    return (HH0, HH1, HH2)
end

# ------------------------
# === Initialization ===
# ------------------------

function init_nodes(N::Int, region_map::Dict{Symbol, Vector{Int}})
    nodes = Vector{AUNode}(undef, N)
    node_to_region = Dict{Int, Symbol}()
    for (r, idxs) in region_map
        for i in idxs; node_to_region[i] = r; end
    end

    for i in 1:N
        bands = Dict(
            :alpha => BandHopf(0.01*(randn() + im*randn()), 10.0),
            :beta  => BandHopf(0.01*(randn() + im*randn()), 20.0),
            :gamma => BandHopf(0.01*(randn() + im*randn()), 40.0),
            :theta => BandHopf(0.01*(randn() + im*randn()), 5.0)
        )

        # Initial smooth local algebra (Stalk)
        dim = 3
        init_algebra = MoritaAlgebra(
            Matrix{ComplexF64}(I, dim, dim), # A
            Matrix{ComplexF64}(I, dim, dim), # B
            Matrix{ComplexF64}(I, dim, dim), # M
            1.0, 0.0, 0.0 # HH invariants
        )

        nodes[i] = AUNode(
            1.0,           # Initial state (smooth)
            bands,
            rand(),
            node_to_region[i],
            init_algebra,  # morita_data
            MolecularLoad(0.0, 0.0), # Initial load
            false          # Not singular
        )
    end
    return nodes
end

# Build adjacency list (incoming edges)
function build_adjacency(quiver::MoritaQuiver)
    N = length(quiver.nodes)
    edges = quiver.edges
    adj = [Int[] for _ in 1:N]
    for (idx, e) in enumerate(edges)
        push!(adj[e.dst], idx)
    end
    return adj
end

function init_edges!(quiver::MoritaQuiver, avg_deg::Int64)
    N = length(quiver.nodes)
    connected = Dict{Tuple{Int64,Int64}, Bool}()

    for i in 1:N
        deg = 0
        while deg < avg_deg
            j = rand(1:N)
            if i != j && !haskey(connected, (i,j)) && !haskey(connected, (j,i))
                d = 5
                edge = MoritaEdge(
                    i, j,
                    rand(), rand(),
                    sprand(d,d,0.2),
                    sprand(d,d,0.2),
                    sprand(d,d,0.2),
                    MolecularLoad(0.0, 0.0)
                )
                push!(quiver.edges, edge)  # add edge
                connected[(i,j)] = true
                deg += 1
            end
        end
    end

    # After all edges are added, build adjacency
    quiver.adjacency = build_adjacency(quiver)

    return quiver
end



# ------------------------
# === Molecule Schedules ===
# ------------------------

struct MoleculeSchedule
    t_on::Int
    t_off::Int
    amplitude::Float64
end

function molecule_level(t::Int, sched::MoleculeSchedule)
    if t < sched.t_on
        return 0.0
    elseif t <= sched.t_off
        return sched.amplitude
    else
        return 0.0
    end
end

molA = MoleculeSchedule(1, 5, 0.5) # Earlier, lower amplitude
molB = MoleculeSchedule(6, 10, 0.5)


# ------------------------
# === Region Sensitivity ===
# ------------------------

function region_sensitivity(region::Symbol)
    if region == :PFC
        return 1.0
    elseif region == :BG
        return 0.6
    elseif region == :Amygdala
        return 0.3
    else
        return 0.1
    end
end

# ------------------------
# === Morita Feedback Term ===
# ------------------------

function morita_feedback(node_idx::Int, quiver::MoritaQuiver)
    acc = 0.0
    for eidx in quiver.adjacency[node_idx]
        e = quiver.edges[eidx]
        # trace-like contraction: cheap proxy for bimodule activity
        if nnz(e.M) > 10  # proxy for nontrivial Massey
            acc += 0.01 * e.weight
        end
        acc += e.weight * nnz(e.M)
    end
    return 1e-4 * acc
end

function rouquier_dimension(quiver::MoritaQuiver)
    depths = Float64[]
    for i in 1:length(quiver.nodes)
        visited = Set([i])
        frontier = [i]
        depth = 0
        while !isempty(frontier)
            next = Int[]
            for u in frontier
                for eidx in quiver.adjacency[u]
                    v = quiver.edges[eidx].src
                    if v ∉ visited
                        push!(visited, v)
                        push!(next, v)
                    end
                end
            end
            isempty(next) && break
            frontier = next
            depth += 1
        end
        push!(depths, depth)
    end
    return mean(depths)
end

# ------------------------
# === Hopf Updates ===
# ------------------------
# THIS IS THE LOCAL PHYSICS (The Stalk)
function update_hopf!(band::BandHopf, node::AUNode, r0_base, coupling, dt)
    # Molecule A increases 'numbness' (makes r more negative)
    # Molecule B increases 'activity' (pushes r toward positive)
    r_eff = r0_base - (2.0 * node.load.A) + (1.5 * node.load.B)
    
    # Frequency modulation: Molecule A slows it down, B speeds it up
    ω_dyn = band.ω * (1.0 - 0.5 * node.load.A + 0.7 * node.load.B)
    
    z = band.z
    # If coupling is 0 (gate closed), and r_eff is negative, dz will be negative.
    # THIS is what creates the dip.
    dz = (r_eff + im * ω_dyn) * z - abs2(z) * z + coupling
    band.z += dt * dz
end

# ------------------------
# === HH² & Kodaira dimension ===
# ------------------------
# Compute instantaneous amplitude of a Hopf band
hopf_amplitude(band::BandHopf) = abs(band.z)

function local_kodaira_dimension(i::Int, quiver::MoritaQuiver; ε=1e-8)
    node = quiver.nodes[i]

    # collect neighbors of i
    nbrs = Int[]
    for e in quiver.edges
        if e.src == i
            push!(nbrs, e.dst)
        elseif e.dst == i
            push!(nbrs, e.src)
        end
    end

    # need at least two neighbors to form a 2-simplex / Massey-like interaction
    length(nbrs) < 2 && return -Inf

    # total oscillatory mass at node i (state-dependent, not control-dependent)
    band_mass = sum(abs(b.z) * b.ω for b in values(node.bands))

    # count "active" bands based on Hopf amplitude
    active_bands = count(b -> hopf_amplitude(b) > ε, values(node.bands))

    hh2_mass = 0.0
    for a in 1:length(nbrs)
        for b in (a+1):length(nbrs)
            j, k = nbrs[a], nbrs[b]

            eij = findfirst(e -> e.src == i && e.dst == j, quiver.edges)
            eik = findfirst(e -> e.src == i && e.dst == k, quiver.edges)

            (eij === nothing || eik === nothing) && continue

            # Massey / HH² proxy via span-induced interaction, we used band_mass 
            # Massey / HH² proxy scaled by active bands instead of total z*ω, we are using active_bands
            hh2_mass += abs(
                quiver.edges[eij].weight *
                quiver.edges[eik].weight
            ) * active_bands #band_mass
        end
    end

    return log(hh2_mass + ε)
end

function compute_kappa_by_region(quiver::MoritaQuiver, region_map::Dict{Symbol,Vector{Int}})
    κ = Dict{Symbol, Vector{Float64}}()
    for (r, idxs) in region_map
        κ[r] = [local_kodaira_dimension(i, quiver) for i in idxs]
    end
    return κ
end

# ---- Kodaira dimension measures what geometry survives.
# ---- Rouquier dimension measures how deep interaction must go to coordinate it.
# ---- Molecules decide both — observables merely witness.
# ------------------------
# === Prolate Observer ===
# ------------------------

function prolate_gap(κvals::Vector{Float64}, jump_loci::Vector{Int})
    good = setdiff(1:length(κvals), jump_loci)
    isempty(good) && return maximum(κvals)
    κ_good = κvals[good]
    return maximum(κ_good)-minimum(κ_good)
end

function step_band!(b::BandHopf, r::Float64, dt)
    z = b.z
    b.z += dt * ((r + im*b.ω)*z - abs(z)^2*z)
end


function update_edge_load!(e::MoritaEdge, t, molA, molB)
    e.load.A += molecule_level(t, molA)
    e.load.B += molecule_level(t, molB)
end

function edge_stress(e::MoritaEdge)
    hh2_proxy = nnz(e.M)
    return hh2_proxy * abs(e.load.A - e.load.B)
end



function rebuild_adjacency!(quiver::MoritaQuiver)
    N = length(quiver.nodes)
    adj = [Int[] for _ in 1:N]  # new adjacency list
    for (idx, e) in enumerate(quiver.edges)
        push!(adj[e.dst], idx)
    end
    quiver.adjacency = adj        # overwrite in-place
    return nothing
end


function prune_edges!(
    quiver::MoritaQuiver;
    stress_threshold = 0.15
)
    before = length(quiver.edges)

    quiver.edges = [
        e for e in quiver.edges
        if edge_stress(e) < stress_threshold
    ]

    if length(quiver.edges) < before
        rebuild_adjacency!(quiver)
    end
end

# 1. Molecular Dynamics with Half-Life
function apply_half_life!(node::AUNode, λ=0.05)
    # Each node locally dissipates its load independently
    for band in values(node.bands)
        # Molecules impacting the Hopf radius decay
        node.c2 *= exp(-λ) 
    end
end

function apply_molecular_stress!(node::AUNode)
    # The load acts as a 'singularizing' parameter on the algebra's generators
    # We rotate the local algebra matrices based on the A/B ratio
    stress_factor = node.load.B / (node.load.A + 1e-8)
    
    # Deform the local algebra A by injecting 'noise' into its structure
    # This represents the local 'Tower' becoming unstable
    for i in 1:size(node.morita_data.A, 1)
        node.morita_data.A[i,i] *= (1.0 + 0.05 * stress_factor * randn())
    end
    
    # Apply Half-Life: Molecules dissipate locally (The 'dissipation' mentioned)
    node.load.A *= 0.95 
    node.load.B *= 0.92 # Slightly faster decay for B to allow recovery
end

function calculate_massey_obstruction(node::AUNode)
    # The Massey Triple Product: ⟨[A], [B], [M]⟩
    # Measures the higher associativity obstruction in the Morita context
    A = node.morita_data.A
    B = node.morita_data.B
    M = node.morita_data.M
    
    massey = (A * B) * M - A * (B * M) + (M * A) * B
    
    # The norm of this tensor is our 'Singularity Metric'
    return norm(massey)
end

function perform_hironaka_blowup!(node::AUNode)
    # Categorical Blow-up:
    # 1. Flag the node as 'Singular' (In the exceptional divisor phase)
    node.is_singular = true
    
    # 2. Increase the 'internal resolution'
    # This simulates replacing the node with a projective line of virtual nodes
    # Effectively dampens the amplitude to account for destructive interference in the divisor
    node.state *= 0.2 
    
    # 3. The Prolate Gap spikes here because the local symmetry is broken
    # (Observable: Prolate Gap spike at Time 3)
end

function perform_hironaka_blowdown!(node::AUNode)
    # Categorical Blow-down:
    # 1. Remove the singularity flag
    node.is_singular = false
    
    # 2. Restitute the 'Ladder' (The Morita Equivalence is restored)
    # The node state (amplitude) can now rise to its steady state
    node.state = 1.0 
    
    # 3. Regional waves (Alpha/Beta) emerge post-Time 3
end



function local_arithmetic_step!(i::Int, quiver::MoritaQuiver)
    node = quiver.nodes[i]
    
    # APPLY DEFORMATION: Molecular load changes the 'Towers'
    # This is the 'deform_algebra' action
    apply_molecular_stress!(node) 

    # MEASURE SINGULARITY: HH2 measures the failure of the 'Ladder'
    hh2_local = calculate_massey_obstruction(node)

    if hh2_local > BLOWUP_THRESHOLD
        # CATEGORICAL BLOW-UP
        # The singularity is resolved by expanding the local category
        # This causes the Prolate Gap spike
        perform_hironaka_blowup!(node)
    elseif half_life_decayed(node)
        # CATEGORICAL BLOW-DOWN
        # Molecular dissipation allows the category to collapse to smooth base
        # This allows regional waves to emerge
        perform_hironaka_blowdown!(node)
    end
end

function update_edge_load!(
    e::QuiverEdge,
    t::Int,
    molA,
    molB;
    decay = 0.01
)
    # local accumulation
    e.load.A += molA(e, t)
    e.load.B += molB(e, t)

    # slow dissipation (optional but realistic)
    e.load.A *= (1 - decay)
    e.load.B *= (1 - decay)
end

function update_local_topology!(node::AUNode, t::Int)
    # 1. Faster Molecular Dissipation
    node.load.A *= 0.65 
    node.load.B *= 0.7 

    # 2. Algebraic Relaxation (The Hironaka Blow-down)
    dim = size(node.morita_data.A, 1)
    target = Matrix{ComplexF64}(I, dim, dim)
    node.morita_data.A += 0.15 * (target - node.morita_data.A)

    # 3. Apply Stress-Induced Deformation (Numbness Logic)
    # Molecule A disrupts the "Smoothness" (Identity Matrix)
    deformation_stress = node.load.A * 0.5 
    if deformation_stress > 0.01
        node.morita_data.A += deformation_stress * randn(ComplexF64, dim, dim)
    end

    # 4. Measure Stress (Map HH² to the available 'c2' field)
    # This measures the obstruction to the associative identity
    node.c2 = norm(node.morita_data.A * node.morita_data.A - node.morita_data.A)
    
    # 5. The Categorical Gate (Morita Valve)
    # Use 'c2' to determine the state. When c2 spikes, state crashes to 0.
    # Threshold 0.25, Sharpness power 8
    node.state = 1.0 / (1.0 + (node.c2 / 0.25)^8) 

    # 6. Update singularity flag for Prolate tracking
    node.is_singular = node.state < 0.1
end

# Parameters

A0 = 0.3
B0 = 0.2
morita_feedback_scale = 0.01
ε = 1e-8
dt = 0.02
tmax = 50           # longer to see transitions
r0_global = -0.15     # slightly subcritical baseline
A0 = 0.3       # amplitude of excitatory modulation
B0 = 0.2       # amplitude of inhibitory modulation
fA = 2.0       # oscillation frequency for lambdaA (cycles over tmax)
τB = 100.0     # decay constant for inhibitory input
coupling_strength = 0.6
function λA(t, tmax)
    region_factor = 1
    #region_factor = region == :PFC ? 1.0 : 0.5
    return region_factor * A0 * sin(2π * fA * t / tmax)
end
function λB(t)
    # Slowly decaying inhibitory input
    return B0 * exp(-t / τB)
end
# ------------------------
# === Simulation Loop ===
# ------------------------
#=
function run_quiver_sim(N::Int, avg_deg::Int, region_map::Dict{Symbol,Vector{Int}}, tmax::Int)
    nodes = init_nodes(N, region_map)
    quiver = MoritaQuiver(nodes, MoritaEdge[], [])
    
    println("Quiver type: ", typeof(quiver))
    println("Quiver fields: ", fieldnames(typeof(quiver)))

    # Populate edges and adjacency
    init_edges!(quiver, avg_deg)

    alpha_hist, beta_hist, gamma_hist, theta_hist = zeros(tmax), zeros(tmax), zeros(tmax), zeros(tmax)
    kappa_hist, prolate_hist = Float64[], Float64[]
    rouquier_hist = Float64[]
    
    for t in 1:tmax
        #r_noise = 0.005 * randn()
        #r = r0 + λA(t, tmax) - λB(t) + r_noise
        # Hopf dynamics
        for e in quiver.edges
            update_edge_load!(e, t, molA, molB)
        end

        prune_edges!(quiver)

        for (i, n) in enumerate(quiver.nodes)
            for band in values(n.bands)
                update_hopf!(
                    band,
                    quiver.nodes[i],
                    i,
                    quiver
                )
                
            end
        end

        @info "edges", length(quiver.edges)
        @info "mean edge stress",
            mean(edge_stress(e) for e in quiver.edges)
        # Kodaira dimension
        κ_by_region = compute_kappa_by_region(quiver, region_map)
        κvals_PFC = κ_by_region[:PFC]
        κ_eff = mean(κvals_PFC)
        jump_loci = detect_kodaira_jumps(κvals_PFC)
        gap = prolate_gap(κvals_PFC, jump_loci)
        push!(kappa_hist, κ_eff)
        push!(prolate_hist, gap)
        push!(rouquier_hist, rouquier_dimension(quiver))

        # Record Hopf averages
        alpha_hist[t] = mean(abs(n.bands[:alpha].z) for n in quiver.nodes)
        beta_hist[t]  = mean(abs(n.bands[:beta].z) for n in quiver.nodes)
        gamma_hist[t] = mean(abs(n.bands[:gamma].z) for n in quiver.nodes)
        theta_hist[t] = mean(abs(n.bands[:theta].z) for n in quiver.nodes)

        println("κ spread = ",
            quantile(κvals_PFC, [0.1, 0.5, 0.9]))
        
        println("t=$t | κ_eff=$(round(κ_eff, digits=3)) | gap=$(round(gap, digits=3)) | edges=$(length(quiver.edges))")
        
    end

    # Plots
    t_vals = 1:tmax
    plot(t_vals, alpha_hist, label="Alpha")
    plot!(t_vals, beta_hist, label="Beta")
    plot!(t_vals, gamma_hist, label="Gamma")
    plot!(t_vals, theta_hist, label="Theta")
    plot!(t_vals, prolate_hist, label="Prolate Gap", lw=2, ls=:dash)
    savefig("quiver_sim_output.png")
    
    return (alpha_hist, beta_hist, gamma_hist, theta_hist, kappa_hist, prolate_hist, rouquier_hist)
end
=#
function generate_simulation_plots(t_vals, alpha, beta, gamma, theta, gap, eta)
    # 1. Initialize a clean plot with clear margins
    p = plot(layout = (1, 1), size=(900, 600), margin=5Plots.mm)

    # 2. Plot Regional Waves (Left Axis)
    plot!(p, t_vals, alpha, label="Alpha (PFC)", color=:blue, lw=2.5)
    plot!(p, t_vals, beta,  label="Beta (BG)",   color=:red,  lw=2.5)
    plot!(p, t_vals, gamma, label="Gamma (Amygdala)", color=:green, lw=2.5)
    plot!(p, t_vals, theta, label="Theta", color=:orange, lw=2.5)

    # 3. Plot Spectral Metrics (Right Axis) 
    # This prevents 'clutter' by separating the scale of waves from HH2 stress
    plot!(p, t_vals, gap, 
          label="Prolate Gap (Spike)", 
          color=:black, ls=:dash, lw=3, twinx=true, ylabel="Spectral Metric")
          
    plot!(p, t_vals, eta, 
          label="Prolate η (HH² Stress)", 
          color=:purple, ls=:dot, lw=2)

    # 4. Critical Annotations (Structural Resolution)
    # Finding the moment the stress drops to zero (The Resolution)
    res_time = findfirst(x -> x < 0.1, eta)
    if res_time !== nothing
        vline!(p, [res_time], label="Hironaka Resolution", color=:gray, alpha=0.5, ls=:dash)
        annotate!(p, res_time + 1, 0.1, text("Blow-down Phase", :left, 8, :gray))
    end

    # 5. Global Styling
    title!(p, "AU Operator: Emergent Phase Transition\n(Stalk-Local Morita Resolution)")
    xaxis!(p, "Time (t)")
    yaxis!(p, "Wave Amplitude", ylims=(0, 1.2)) # Standardize wave height
    
    # Move legend to prevent overlapping the plot data
    plot!(p, legend=:outertopright, legendfontsize=9)

    return p
end


function run_and_plot(N, avg_deg, region_map, tmax)
    # Initialize
    nodes = init_nodes(N, region_map)
    quiver = MoritaQuiver(nodes, MoritaEdge[], [])
    init_edges!(quiver, avg_deg)
    
    # Storage for Plotting
    a_h, b_h, g_h, t_h = zeros(tmax), zeros(tmax), zeros(tmax), zeros(tmax)
    gap_h, eta_h = zeros(tmax), zeros(tmax)

    for t in 1:tmax
        # 1. Local AU Step (Strictly local deformation)
        for i in 1:N
            node = quiver.nodes[i]
    
            # Simulate the "Numbness" event (Molecule A spike)
            if t > 5 && t < 15
                node.load.A += 0.6  # PFC begins to feel 'numb'
            end
            
            # Simulate the "Activity" recovery (Molecule B spike)
            if t > 20
                node.load.B += 0.8  # High Hz activity emerges
            end
            update_local_topology!(quiver.nodes[i], t)
        end

        # 2. Local Wave Update (Coupling gated by local node.state)
        # 2. Local Wave Update (Iterate through all bands)
        # THIS IS THE CATEGORY (The Quiver Loop)
        for (i, n) in enumerate(quiver.nodes)
            for band_key in keys(n.bands)
                # COUPLING LOGIC: Multiply by quiver.nodes[e.src].state
                # This is the 'Valve' that causes the dip at Time 3
                incoming_signal = sum((e.weight * quiver.nodes[e.src].state * quiver.nodes[e.src].bands[band_key].z 
                                for eidx in quiver.adjacency[i] 
                                for e in [quiver.edges[eidx]]); init=0.0 + 0.0im)
                
                update_hopf!(n.bands[band_key], n, r0_global, coupling_strength * incoming_signal, dt)
            end
        end

        # 3. Harvest Observables for the Plot
        a_h[t] = mean(abs(n.bands[:alpha].z) for n in quiver.nodes)
        b_h[t] = mean(abs(n.bands[:beta].z) for n in quiver.nodes)
        g_h[t] = mean(abs(n.bands[:gamma].z) for n in quiver.nodes)
        t_h[t] = mean(abs(n.bands[:theta].z) for n in quiver.nodes)
        
        # Calculate the Gap (witnessing the local singularity)
        κ = compute_kappa_by_region(quiver, region_map)[:PFC]
        gap_h[t] = prolate_gap(κ, detect_kodaira_jumps(κ))
        eta_h[t] = mean([calculate_massey_obstruction(n) for n in quiver.nodes])
    end

    # Generate Output
    display(generate_simulation_plots(1:tmax, a_h, b_h, g_h, t_h, gap_h, eta_h))
    savefig("AU_Emergent_Transition.png")
end

# ------------------------
# === Run Example ===
# ------------------------

NODES = 200
region_map = Dict(
    :PFC => collect(1:60),
    :BG => collect(61:120),
    :Amygdala => collect(121:200)
)
run_and_plot(NODES, 5, region_map, tmax)

