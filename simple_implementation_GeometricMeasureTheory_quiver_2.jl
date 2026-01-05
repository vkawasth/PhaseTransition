using LinearAlgebra, Statistics, SparseArrays, Random
using Plots

# ======================================================
# === Core Structures
# ======================================================
mutable struct AUNodes
    state::Float64
    c0::Float64
    c1::Float64
    c2::Float64
end

mutable struct MolecularLoad
    A::Float64
    B::Float64
end

mutable struct BandHopf
    z::ComplexF64
    ω::Float64
    base_ω::Float64
end

mutable struct EdgeHopf
    ω::Float64
    γ::Float64
    phase::Float64
end

mutable struct MoritaAlgebra
    A::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    M::Matrix{ComplexF64}
end

mutable struct AUNode
    bands::Dict{Symbol,BandHopf}
    morita::MoritaAlgebra
    load::MolecularLoad
    c2::Float64
    state::Float64
end

mutable struct MoritaEdge
    src::Int
    dst::Int
    weight::Float64
    M::SparseMatrixCSC{Float64,Int}
    load::MolecularLoad
    # New Geometric Properties
    thickness::Float64   # Effective diameter for density flow
    curvature::Float64   # Resistance to frequency consensus (Twist)
    hopf::EdgeHopf       # Organic frequency/phase carrier
end

mutable struct MoritaQuiver
    nodes::Vector{AUNode}
    edges::Vector{MoritaEdge}
    adjacency::Vector{Vector{Int}}
end
Base.iterate(q::MoritaQuiver, state=1) = state > length(q.nodes) ? nothing : (q.nodes[state], state + 1)

function Base.isless(a::AUNode, b::AUNode)
    if isnan(a.c2) return false end
    if isnan(b.c2) return true end
    return a.c2 < b.c2
end
# max over c2
function max_c2(quiver::MoritaQuiver)
    maximum(node.c2 for node in quiver.nodes)
end
function min_c2(quiver::MoritaQuiver)
    minimum(node.c2 for node in quiver.nodes)
end

# 1. Define Edge Stiffness Logic
function calculate_stiffness(e::MoritaEdge, q::MoritaQuiver)
    src_n = q.nodes[e.src]
    dst_n = q.nodes[e.dst]
    
    # Stiffness is the delta in 'Disruption' across the edge
    # This captures the 3 categories (A-A, B-B, A-B)
    delta_A = abs(src_n.load.A - dst_n.load.A)
    delta_B = abs(src_n.load.B - dst_n.load.B)
    
    # Category III (Asymmetric) will produce the highest value here
    return (delta_A + delta_B) * e.weight * massey_norm(src_n.morita.A, dst_n.morita.B, e.M)
end


# 2. Overload isless for edges
Base.isless(e1::MoritaEdge, e2::MoritaEdge) = e1.weight < e2.weight # Or use a cached stiffness field

struct RegionPrior
    name::Symbol
    # Sensitivity of each band's ω to Molecule A
    # Positive = Speed up, Negative = Slow down
    ω_sensitivity::Dict{Symbol, Float64} 
    # Impact on power (amplitude) saturation
    p_sensitivity::Dict{Symbol, Float64}
end

# Example: PFC Prior - Opiate (A) slows Theta/Gamma
pfc_prior = RegionPrior(:PFC, 
    Dict(:theta => -5.0, :gamma => -10.0, :alpha => 0.0, :beta => 0.0, :delta => 0.0),
    Dict(:theta => -0.2, :gamma => -0.3, :alpha => 0.0, :beta => 0.0, :delta => 0.0))

# Example: BG Prior - Opiate (A) raises Gamma for reward processing
bg_prior = RegionPrior(:BasalGanglia, 
    Dict(:gamma => 15.0, :theta => 0.0, :alpha => 0.0, :beta => 0.0, :delta => 0.0),
    Dict(:gamma => 0.4, :theta => 0.0, :alpha => 0.0, :beta => 0.0, :delta => 0.0))

function update_node_spectral_geometry!(node::AUNode, prior::RegionPrior, dt::Float64)
    mol_A = node.load.A
    
    for (name, band) in node.bands
        # 1. Independent Frequency Drive
        # No global omega; ω is shifted by Molecule A + Regional Prior
        target_ω = band.base_ω + (mol_A * prior.ω_sensitivity[name])
        
        # 2. Independent Power Drive (Saturation)
        # Molecule A can raise or lower the limit cycle ceiling (mu)
        mu = 1.0 + (mol_A * prior.p_sensitivity[name]) - abs2(band.z)
        
        # 3. Complex Hopf Integration
        band.z += (Complex(mu, target_ω) * band.z) * dt
    end
    
    # node.state now reflects the total spectral energy for the Prolate observer
    node.state = sum(abs2(b.z) for (k, b) in node.bands)
end


function update_node_bands_harmonic!(node::AUNode, edge_omega::Float64, dt::Float64)
    # Define the Harmonic ratios for the 5 waves
    # Based on the single EdgeHopf consensus (omega)
    ratios = Dict(
        :delta => 0.1,  # 1-4 Hz
        :theta => 0.2,  # 4-8 Hz
        :alpha => 0.4,  # 8-13 Hz
        :beta  => 0.8,  # 13-30 Hz
        :gamma => 1.5   # 30-100 Hz
    )

    for (name, band) in node.bands
        # 1. MOLECULE A DRIVE (The Slow Down)
        # Molecule A (Opiate) acts as a 'Geometric Brake' on the scaling
        slow_factor = 1.0 / (1.0 + node.load.A) 
        
        # All 5 waves derive their speed from one omega
        local_target_w = edge_omega * ratios[name] * slow_factor
        
        # 2. Limit Cycle Evolution
        r2 = abs2(band.z)
        mu = 1.0 - r2
        dz = (Complex(mu, local_target_w) * band.z) * dt
        band.z += dz
    end
end



# 3. Fast Clamping via the "Edge Wall"
function resolve_sheaf_singularities!(q::MoritaQuiver, dt)
    # Step A: Morphism Negotiation (The 'Tension' calculation)
    # This identifies where Molecule B is dislodging Molecule A
    for e in q.edges
        organic_edge_interaction!(q.nodes[e.src], q.nodes[e.dst], e, dt)
    end

    # Step B: Stalk Resonances (The 'Spectral Response')
    # This is where Alpha/Gamma power shifts based on that tension
    for i in 1:length(q.nodes)
        update_node_bands!(q.nodes[i], q, i, dt)
    end
end

function get_stiffness_locus(q::MoritaQuiver)
    # This calls your Base.isless logic
    critical_node = maximum(q.nodes)
    
    # We return a normalized stiffness measure
    # This forces the "Plateau" by capping the stiffness at 1.0
    return tanh(critical_node.c2)
end

# ======================================================
# === HH² / Massey Obstruction
# ======================================================

function massey_norm(A,B,M)
    norm((A*B)*M - A*(B*M) + (M*A)*B)
end

# ======================================================
# === Initialization
# ======================================================

function init_nodes(N)
    nodes = AUNode[]
    for _ in 1:N
        bands = Dict(
            :alpha => BandHopf(0.01randn()+0.01im*randn(), 10.0, 10.0),
            :beta  => BandHopf(0.01randn()+0.01im*randn(), 20.0, 20.0),
            :gamma => BandHopf(0.01randn()+0.01im*randn(), 40.0, 40.0),
            :theta => BandHopf(0.01randn()+0.01im*randn(), 5.0, 5.0),
            :delta => BandHopf(0.01randn()+0.01im*randn(), 2.0, 2.0)
        )
        I3 = Matrix{ComplexF64}(I,3,3)
        push!(nodes, AUNode(
            bands,
            MoritaAlgebra(I3, I3, I3),
            MolecularLoad(0.0,0.0),
            0.0,
            1.0
        ))
    end
    return nodes
end

function init_edges!(q::MoritaQuiver, avg_deg)
    N = length(q.nodes)
    for i in 1:N
        for _ in 1:avg_deg
            j = rand(1:N)
            
            # New geometric/spectral initial states
            init_thickness = 1.0
            init_curvature = 0.0
            init_hopf = EdgeHopf(10.0, 0.01, rand() * 2π) # 10Hz, low damp, random phase
            
            push!(q.edges,
                MoritaEdge(
                    i, j,
                    rand(),              # weight
                    sprand(3,3,0.4),     # M (Morita bimodule)
                    MolecularLoad(0.0,0.0), 
                    init_thickness,      # thickness
                    init_curvature,      # curvature
                    init_hopf            # EdgeHopf
                )
            )
        end
    end
    rebuild_adjacency!(q)
end

function rebuild_adjacency!(q::MoritaQuiver)
    q.adjacency = [Int[] for _ in q.nodes]
    for (k,e) in enumerate(q.edges)
        push!(q.adjacency[e.dst], k)
    end
end

function stalk_prolate_operator(node::AUNode)
    # Build a local feature vector from bands
    v_band = [
        abs(node.bands[:delta].z),
        abs(node.bands[:theta].z),
        abs(node.bands[:alpha].z),
        abs(node.bands[:beta].z),
        abs(node.bands[:gamma].z)
    ]

    # Build a competing “physical” vector
    v_phys = [
        node.load.A,
        node.load.B,
        node.c2,
        node.state,
        sum(abs2(b.z) for b in values(node.bands))
    ]

    # Normalize
    vb = v_band / (norm(v_band) + 1e-9)
    vp = v_phys / (norm(v_phys) + 1e-9)

    # Rank-1 projectors
    P_band = vb * vb'
    P_phys = vp * vp'

    # Prolate operator = overlap of bases
    return P_band * P_phys * P_band
end

function stalk_prolate_gap(node::AUNode)
    P = stalk_prolate_operator(node)
    λ = eigvals(Symmetric(P))
    return maximum(λ) - minimum(λ)
end


# ======================================================
# === Molecular Dynamics
# ======================================================

struct MoleculeSchedule
    t_on::Int
    t_off::Int
    amp::Float64
end

level(t,s) = (s.t_on ≤ t ≤ s.t_off) ? s.amp : 0.0

# ======================================================
# === Edge → Node Propagation
# ======================================================

function propagate_edge_loads!(q::MoritaQuiver)
    for e in q.edges
        q.nodes[e.dst].load.A += 0.2e.load.A
        q.nodes[e.dst].load.B += 0.2e.load.B
    end
end

# ======================================================
# === Edge Stress & Pruning
# ======================================================

edge_stress(e::MoritaEdge) = nnz(e.M) * abs(e.load.A - e.load.B)

function prune_edges!(q::MoritaQuiver; thresh=0.4)
    q.edges = [e for e in q.edges if e.src==e.dst || edge_stress(e)<thresh]
    rebuild_adjacency!(q)
end

# ======================================================
# === Hopf Dynamics (HH²-Coupled)
# ======================================================

function update_hopf!(band::BandHopf, node::AUNode, coupling, dt)
    # Clamp r to prevent the "Frozen Oscillator" death spiral
    # r represents the gain; it shouldn't drop below a biological/physical floor.
    r_raw = -0.15 - 0.5node.load.A + 0.7node.load.B - 0.3node.c2
    r = max(r_raw, -2.0) 
    
    # Frequency scaling stays relevant to the phase change
    ω = band.ω * exp(-0.15 * node.c2)
    
    z = band.z
    # Normal form with the limited gain 'r'
    band.z += dt*((r + im*ω)*z - abs2(z)*z + coupling)
end

function transport_entropy_clamped(q::MoritaQuiver)
    # Weigh p by the edge stiffness categories
    # This ensures that 'Asymmetric' (A-B) regions contribute more to entropy
    p = [node.load.A + node.load.B + sum(abs(b.z) for b in values(node.bands)) for node in q.nodes]
    
    total_mass = sum(p)
    if total_mass < 1e-5 return 0.0 end
    
    p ./= total_mass
    # Remove the 'min' clamp momentarily to see if it moves
    return -sum(pi * log(pi + 1e-12) for pi in p)
end

function organic_edge_interaction!(A::AUNode, B::AUNode, e::MoritaEdge, dt::Float64)
    # Extract the local oscillator from the morphism
    h = e.hopf
    
    # 1. Target Frequency Negotiation
    # Stalk A (Opiate) pulls to 8Hz; Stalk B (Naloxone) pulls to 30Hz
    target_A = A.load.A > 0.1 ? 8.0 : 30.0
    target_B = B.load.B > 0.1 ? 30.0 : 8.0
    
    # 2. Geometric Tension (The 'Twist' where gluing breaks)
    e.curvature = abs(target_A - target_B)
    
    # 3. Frequency Update
    # Edge ω evolves toward the mean target, dampened by curvature
    avg_target = (target_A + target_B) / 2.0
    h.ω += (avg_target - h.ω) * dt - (0.05 * e.curvature * h.ω * dt)

    # 4. Phase Recalculation (Critical for Jacobi Tridiagonal)
    # The phase must evolve based on the organic frequency to avoid flat plots
    h.phase += h.ω * dt + (e.curvature * sin(h.phase)) * dt
    h.phase = mod2pi(h.phase)

    # 5. Intersection Memory (c2) Update
    # Accumulate failure of the t-structure at the stalks
    A.c2 += e.curvature * abs(sin(h.phase)) * dt
    B.c2 += e.curvature * abs(sin(h.phase)) * dt
end

#=
function update_hopf!(band::BandHopf, node::AUNode, coupling, dt)
    r = -0.15 - 0.5node.load.A + 0.7node.load.B - 0.3node.c2
    ω = band.ω * exp(-0.2node.c2)
    z = band.z
    band.z += dt*((r + im*ω)*z - abs2(z)*z + coupling)
end
=#
# ======================================================
# === Local Arithmetic Geometry Step
# ======================================================

function update_node_geometry!(node::AUNode)
    A,B,M = node.morita.A, node.morita.B, node.morita.M
    node.c2 = massey_norm(A,B,M)
    # 1. Lower the Activation Floor (Start the struggle earlier)
    # Change the state gate to be more sensitive
    node.state = 1 / (1 + (node.c2 / 0.8)^4) # Was 0.3, move to 0.8

    # 2. Raise the Resolution Ceiling (Allow more energy accumulation)
    # Let the Prolate Gap grow significantly before the Blow-down
    if node.c2 > 8.0  # Was 3.8 or 4.5
        # The resolution happens later, allowing the 'Gap' to show a curve
        node.morita.A += 0.1 * randn(3,3) # Add noise to kick it out of the flatline
        node.c2 = 0.5 
    end

    if node.c2 > 4.0
        # Instead of Identity(3,3), use a "Noisy" Identity
        # This keeps the Prolate Gap from collapsing to absolute zero
        node.morita.A = Matrix{ComplexF64}(I, 3, 3) + 0.05*randn(3,3)
        node.morita.B = Matrix{ComplexF64}(I, 3, 3) + 0.05*randn(3,3)
        
        # Don't zero out the load; just dampen it (The "Soft Landing")
        node.load.A *= 0.3
        node.load.B *= 0.3
        node.c2 = 0.5 # A baseline 'tension' remains
    end

    # molecular decay
    node.load.A *= 0.92
    node.load.B *= 0.90
end

# ======================================================
# === Kodaira & Prolate
# ======================================================

function local_kodaira(i::Int, q::MoritaQuiver; ε=1e-12)
    node = q.nodes[i]

    # collect neighbors
    nbrs = Int[]
    for e in q.edges
        if e.src == i
            push!(nbrs, e.dst)
        elseif e.dst == i
            push!(nbrs, e.src)
        end
    end

    # not enough spans → no deformation space
    if length(nbrs) < 2
        return 0.0 
    end
    
    band_mismatch = isempty(node.bands) ? 0.0 : sum(
        abs(b.z) * abs(b.ω) for b in values(node.bands);
        init = 0.0
    )

    hh2_mass = 0.0

    for a in 1:length(nbrs)-1
        for b in a+1:length(nbrs)
            j, k = nbrs[a], nbrs[b]

            eij = findfirst(e -> (e.src == i && e.dst == j) || (e.src == j && e.dst == i), q.edges)
            eik = findfirst(e -> (e.src == i && e.dst == k) || (e.src == k && e.dst == i), q.edges)


            if eij === nothing || eik === nothing
                continue
            end

            hh2_mass += abs(q.edges[eij].weight * q.edges[eik].weight) * band_mismatch
        end
    end

    return hh2_mass > 0 ? log(hh2_mass + ε) : -Inf
end

"""
Approximate Rouquier dimension proxy.
Tracks categorical complexity per effective edge scale.
"""

function rouquier_proxy(q::MoritaQuiver; ε = 1e-12)
    # Effective number of active edges
    E = length(q.edges)
    E == 0 && return 0.0

    # Proxy HH₀: node centers (non-isolated nodes)
    HH0 = 0
    for i in eachindex(q.nodes)
        if !isempty(q.adjacency[i])
            HH0 += 1
        end
    end

    # Proxy HH₁: active deformations (span-supported edges)
    HH1 = 0
    for e in q.edges
        # Use bimodule density as deformation signal
        HH1 += nnz(e.M)
    end

    # Normalize deformation mass
    HH1 = log(HH1 + 1)

    # Rouquier-like scaling
    rdim = log(HH0 + HH1 + ε) / log(E + 1)

    return rdim
end

function transport_entropy(q::MoritaQuiver)
    N = length(q.nodes)
    p = zeros(Float64, N)
    total_mass = 0.0

    # Use the sum of molecule A and B as “mass” for entropy
    for (i,node) in enumerate(q.nodes)
        mass = node.load.A + node.load.B
        p[i] = mass
        total_mass += mass
    end

    if total_mass == 0.0
        return 0.0
    end

    # Normalize to get probability distribution
    p ./= total_mass

    # Shannon entropy
    entropy = -sum(pi > 0 ? pi*log(pi) : 0.0 for pi in p)
    return entropy
end

"""
Measures entropy-driven transport imbalance across the quiver.
Detects separation of equilibrium from chaos.
"""
function transport_imbalance(q::MoritaQuiver)
    N = length(q.nodes)
    incoming = zeros(Float64, N)
    outgoing = zeros(Float64, N)

    for e in q.edges
        # Probability mass carried by edge
        # weight * Frobenius norm of bimodule = info capacity
        mass = abs(e.weight) * sqrt(nnz(e.M) + 1)

        outgoing[e.src] += mass
        incoming[e.dst] += mass
    end

    # Total imbalance (L¹ divergence)
    imbalance = sum(abs.(incoming .- outgoing))

    return imbalance
end

function prolate_gap(q::MoritaQuiver)
    c2_vals = [n.c2 for n in q.nodes]
    # If the gap is too large, the 'Resolution Functor' failed. 
    # We force a Hard Clip on the observable itself.
    raw_gap = maximum(c2_vals) - minimum(c2_vals)
    
    # Slepian-style clamping: The gap cannot exceed the 
    # algebraic capacity of the Morita dimension (3.0)
    return tanh(raw_gap / 3.0) * 3.0
end

prolate_gap(v) = max_c2(v)-min_c2(v)

# ======================================================
# === Simulation
# ======================================================

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

function update_node_bands!(node::AUNode, q::MoritaQuiver, node_idx::Int, dt)
    for (name, band) in node.bands
        # 1. Collect Coupling from Edges (Information Flow: Edge -> Band)
        coupling = 0.0 + 0.0im
        for edge_idx in q.adjacency[node_idx]
            edge = q.edges[edge_idx]
            
            # The EdgeHopf phase modulates the transfer of energy
            # from neighbor nodes into this specific band.
            neighbor = q.nodes[edge.src == node_idx ? edge.dst : edge.src]
            
            # This is the 'Resonant Gate': 
            # Energy flows if the Edge frequency matches the Band frequency
            resonance = exp(-abs(edge.hopf.ω - band.ω))
            coupling += edge.weight * resonance * neighbor.bands[name].z * exp(im * edge.hopf.phase)
        end

        # 2. Update the BandHopf (Non-linear saturation)
        update_hopf!(band, node, coupling, dt)
    end
end


function propagate_edge_loads!(q::MoritaQuiver, t, molA, molB)
    κ = 0.2 
    for e in q.edges
        A_stalk = q.nodes[e.src]
        B_stalk = q.nodes[e.dst]

        # 1. Update Edge Thickness via Local Asymmetry
        # Thickness collapses if one side is Opiate and other is Naloxone
        asymmetry = abs(A_stalk.load.A - B_stalk.load.B)
        e.thickness = exp(-asymmetry) # Thins the edge during disruption

        # 2. Geometry-Driven Diffusion
        # Thickness acts as the 'Pipe Diameter' for the Gaussian-like flow
        diff_A = (A_stalk.load.A - B_stalk.load.A) * κ * e.thickness
        diff_B = (A_stalk.load.B - B_stalk.load.B) * κ * e.thickness
        
        # Stalk-to-Stalk transport
        B_stalk.load.A += diff_A; A_stalk.load.A -= diff_A
        B_stalk.load.B += diff_B; A_stalk.load.B -= diff_B

        # 3. Organic Edge Interaction (Asymmetric Logic)
        organic_edge_interaction!(A_stalk, B_stalk, e, 0.1) # dt = 0.1
    end
end

function generate_simulation_plots_p(t_vals, rouquir, entropy, koidara, theta, gap, eta)
    # 1. Initialize a clean plot with clear margins
    p = plot(layout = (1, 1), size=(900, 600), margin=5Plots.mm)

    # 2. Plot Regional Waves (Left Axis)
    plot!(p, t_vals, rouquir, label="Rouquir Proxy (PFC)", color=:blue, lw=2.5)
    plot!(p, t_vals, entropy,  label="entropy ",   color=:red,  lw=2.5)
    plot!(p, t_vals, koidara, label="Koidara ", color=:green, lw=2.5)
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

# ======================================================
# === Stabilized Local Arithmetic Geometry
# ======================================================

function update_node_geometry!(node::AUNode, q::MoritaQuiver, idx::Int)
    # ... previous nbrs/load code ...

    # 1. Active Spectral Clip
    # Instead of letting m_norm grow, we treat it as an angle on a circle (S1)
    # This keeps the geometry compact (Invertible Morita context)
    m_norm = massey_norm(node.morita.A, node.morita.B, node.morita.M)
    clamped_m = 2.0 * atan(m_norm) # Projects [0, ∞) onto [0, π)

    # 2. Multiplicative Satiation
    # Instead of adding loads (which allows runaway), multiply by a satiation factor
    satiation = 1.0 / (1.0 + (node.load.A + node.load.B)^2)
    node.c2 = (clamped_m + 0.5) * (1.0 - satiation)

    # 3. THE HARD RESET (Hironaka Blow-down)
    # If the gap still tries to widen, we force a 'Phase Lock'
    if node.c2 > 1.8 
        # Reset the Morita Algebra to a stable "Skeletal" Identity
        node.morita.A .= [1.0 0 0; 0 1.0 0; 0 0 1.0]
        node.morita.B .= [1.0 0 0; 0 1.0 0; 0 0 1.0]
        # Drain the node's capacity to hold more information
        node.load.A = 0.0
        node.load.B = 0.0
        node.c2 = 0.05
    end
    
    node.state = 1 / (1 + (node.c2/0.3)^6)
end

#=
function update_node_geometry!(node::AUNode, q::MoritaQuiver, idx::Int)
    nbrs = [e.dst for e in q.edges if e.src == idx]
    load_sum = sum(q.nodes[j].load.A + q.nodes[j].load.B for j in nbrs; init=0.0)
    
    # Include band activity (alpha+theta) in c2
    band_activity = sum(abs(b.z) for b in values(node.bands))
    
    node.c2 = massey_norm(node.morita.A, node.morita.B, node.morita.M) +
              0.5*(node.load.A + node.load.B) + 
              0.3*load_sum + 0.2*band_activity
    node.state = 1 / (1 + (node.c2/0.3)^6)
    
    node.load.A *= 0.92
    node.load.B *= 0.90
end
=#
dt = 0.02

function propagate_wavefront!(nodes::Vector{AUNode}; factor_c1=0.2, factor_c0=0.1)
    N = length(nodes)
    new_c1 = zeros(N)
    new_c0 = zeros(N)
    
    # Spread along neighbors (The Geometric Morphism)
    for i in 1:N
        # Access the current stalk's chemical load
        c1_val = nodes[i].load.B  # Mapping c1 to Naloxone (B)
        c0_val = nodes[i].load.A  # Mapping c0 to Opiate (A)
        
        for j in max(1, i-1):min(N, i+1)
            new_c1[j] += c1_val * factor_c1
            new_c0[j] += c0_val * factor_c0
        end
    end
    
    # Apply updates to the Arithmetic Universe stalks
    for i in 1:N
        nodes[i].load.B += new_c1[i]
        nodes[i].load.A += new_c0[i]
        # Intersection memory c2 tracks the 'front' of the wave
        nodes[i].c2 += (new_c1[i] * nodes[i].load.A) * 0.1 
    end
end

function jacobi_tridiag(nodes::Vector{AUNode}, edges::Dict{Tuple{Int,Int},EdgeHopf})
    N = length(nodes)
    main_diag = zeros(Float64, N)
    off_diag = zeros(Float64, N - 1)

    for i in 1:N
        # Diagonal: Reflects the dominant regional wave power
        # This links the blue/green lines to the black dashed line
        alpha_pwr = abs2(nodes[i].bands[:alpha].z)
        gamma_pwr = abs2(nodes[i].bands[:gamma].z)
        main_diag[i] = (gamma_pwr - alpha_pwr) + nodes[i].c2
    end

    for ((i, j), h) in edges
        if abs(i - j) == 1
            idx = min(i, j)
            # Off-Diagonal: Gluing is broken by the phase-slip (h.phase)
            # and local tension (c2)
            tension = 0.5 * (nodes[i].c2 + nodes[j].c2)
            gluing = h.ω * cos(h.phase) * exp(-tension)

            off_diag[idx] = -gluing
            main_diag[i] += abs(gluing)
            main_diag[j] += abs(gluing)
        end
    end
    return Tridiagonal(off_diag, main_diag, off_diag)
end

function update_hopf_integrated!(band::BandHopf, node::AUNode, coupling::ComplexF64, dt::Float64)
    r2 = abs2(band.z)
    
    # Saturation term: keeps Gamma/Alpha below 1.0 (Fixes your 3rd plot runaway)
    mu = 1.0 - r2 
    
    # Frequency Flip: This creates the two basis functions in the Jacobi matrix
    # If Naloxone is present, we push for 30Hz; otherwise 8Hz
    target_ω = (node.load.B > 0.2) ? 30.0 : (node.load.A > 0.1 ? 8.0 : 15.0)
    
    dz = (Complex(mu, target_ω) * band.z + coupling) * dt
    band.z += dz
    
    # Update stalk state for the Jacobi diagonal
    node.state = sum(abs2(b.z) for b in values(node.bands))
end

function run_quiver_sim(N, avg_deg, tmax)
    nodes = init_nodes(N)
    q = MoritaQuiver(nodes, MoritaEdge[], [])
    init_edges!(q, avg_deg)
    # Storage for Plotting
    a_h, b_h, g_h, t_h = zeros(tmax), zeros(tmax), zeros(tmax), zeros(tmax)
    jacobi_gap_h, eta_h = zeros(tmax), zeros(tmax)

    α = zeros(tmax); prolate_stalk_gap_h = zeros(tmax)

    #gap = zeros(tmax)

    rdim = zeros(tmax)
    κ_var = zeros(tmax)
    entropy = zeros(tmax)

    molA = MoleculeSchedule(3,10,0.6)
    molB = MoleculeSchedule(12,25,0.8)

    for t in 1:tmax
        # 1. PRIMARY CHEMISTRY & WAVEFRONT (Geometric Spread)
        # Molecules spread along the 1D ladder first to set the chemical landscape
        propagate_wavefront!(q.nodes; factor_c1=0.2, factor_c0=0.1)
        
        # 2. MORPHISM INTERACTION (Organic Phase/Frequency Recalculation)
        # We iterate through the existing 'edges' array in MoritaQuiver
        for e in q.edges
            A_stalk = q.nodes[e.src]
            B_stalk = q.nodes[e.dst]
            organic_edge_interaction!(q.nodes[e.src], q.nodes[e.dst], e, dt)
        end
    
        # 3. INTERNAL RESONANCE (BandHopf)
        # Stalks update their amplitudes based on neighbors and EdgeHopf resonance
        # 3. Update BandHopf with clamping to prevent runaway
        for n in q.nodes
            for band in values(n.bands)
                update_hopf_integrated!(band, n, 0.0im, dt)
            end
        end

        # 4. STALK-LOCAL PROLATE (Basis Instability)
        for (i, n) in enumerate(q.nodes)
            prolate_stalk_gap_h[t] += stalk_prolate_gap(n)
        end
        prolate_stalk_gap_h[t] /= length(q.nodes)
    
        # 4. SPECTRAL OBSERVATION (Prolate Observer)
        # Build the Jacobi 1D ladder from the surviving node/edge states
        jacobi_gap_h[t] = 0.0
        if length(q.nodes) >= 10
            # We manually build the Dict here if the tridiag function requires it, 
            # or pass the edges directly.
            # Build a temporary map of the morphisms for the 1D ladder
            temp_edge_map = Dict{Tuple{Int,Int}, EdgeHopf}()
            for e in q.edges
                temp_edge_map[(e.src, e.dst)] = e.hopf
            end
            
            # Now correctly matches the Vector{AUNode} signature
            J = jacobi_tridiag(q.nodes, temp_edge_map)
            
            try
                λ, U = eigen(J)
                sort!(λ)
                jacobi_gap_h[t] += length(λ) >= 2 ? λ[end] - λ[end-1] : 0.0
                eta_h[t] = sum(abs.(U[:, end])) / length(q.nodes)
            catch
                jacobi_gap_h[t] = 0.0
            end
        end
        jacobi_gap = round(jacobi_gap_h[t]; digits=4)
        prolate_gap = round(prolate_stalk_gap_h[t], digits=4)
        @info "t=$t" 
            prolate_stalk_gap = prolate_gap
            jacobi_gap        = jacobi_gap
            edges             = length(q.edges)
    
        # 5. HARVEST OBSERVABLES
        # 6. HARVEST METRICS (Averages for surviving nodes)
        a_h[t] = mean(abs(n.bands[:alpha].z) for n in q.nodes)
        b_h[t] = mean(abs(n.bands[:beta].z) for n in q.nodes)
        g_h[t] = mean(abs(n.bands[:gamma].z) for n in q.nodes)
        t_h[t] = mean(abs(n.bands[:theta].z) for n in q.nodes)
        
        entropy[t] = transport_entropy_clamped(q)
        rdim[t]    = rouquier_proxy(q)
        
        # Local Kodaira Variance observes the t-structure collapse
        κ_vals = [local_kodaira(i, q) for i in 1:length(q.nodes)]
        κ_var[t] = var(filter(!isnan, κ_vals))
    end
    
    # Final Visualizations
    generate_simulation_plots(1:tmax, a_h, b_h, g_h, t_h, prolate_stalk_gap_h, eta_h)
    savefig("AU_Emergent_Transition_3.png")
    generate_simulation_plots_p(1:tmax, rdim, entropy, κ_var, a_h, jacobi_gap_h, t_h)
    savefig("AU_Emergent_Transition_2.png")
    return a_h, prolate_stalk_gap_h
end
tmax = 20
NODES = 200
region_map = Dict(
    :PFC => collect(1:60),
    :BG => collect(61:120),
    :Amygdala => collect(121:200)
)
run_quiver_sim(NODES, 5, tmax)