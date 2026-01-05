using LinearAlgebra, Statistics, Plots

# ======================================================
# === 1. The Algebra: kQ / I
# ======================================================

# Morphisms are elements of the path algebra
mutable struct PathMorphism
    e::ComplexF64    # Degree 1: The arrow
    ee::ComplexF64   # Degree 2: Composition (The Gluing)
    stiffness::Float64 # Resistance to Relation Ideal I
end

mutable struct Stalk
    z::ComplexF64    # Section of the sheaf
    ω_base::Float64  # Poisson base frequency
    mass::Float64    # Representation probability
end

struct RegionalPrior
    name::Symbol
    ω_sens::Float64  # PFC: Slows Gamma (-), Speeds Theta (+)
    p_sens::Float64  # PFC: Disinhibition (+)
end

mutable struct MoritaQuiver
    nodes::Vector{Stalk}
    priors::Vector{RegionalPrior}
    # Path algebra morphisms: src -> dst
    # These are the actual "gluing" operators
    morphisms::Dict{Tuple{Int, Int}, PathMorphism} 
    loads_A::Vector{Float64} # Disruptor concentration
end

# ======================================================
# === 2. Prolate Operator (Spectral Observer)
# ======================================================

function compute_prolate_gap(q::MoritaQuiver)
    N = length(q.nodes)
    J = zeros(Float64, N, N)
    
    for i in 1:N
        # Diagonal: Poisson-derived energy from Molecule A
        # (Assuming Molecule B = 0 for this disruption phase)
        J[i,i] = -q.loads_A[i]
    end
    
    # Off-diagonal: Determined by degree 2 path morphisms
    # This is where the "gluing" is physically inspected
    for ((src, dst), morph) in q.morphisms
        # Gluing exists only if the degree 2 path hasn't vanished into Ideal I
        J[src, dst] = -real(morph.ee)
        J[dst, src] = -real(morph.ee)
    end
    
    λ = eigvals(Hermitian(J))
    return λ[end] - λ[end-1]
end



# ======================================================
# === 3. t-structure Morphisms & Disruption
# ======================================================

function evolve_quiver_algebra!(q::MoritaQuiver, dt::Float64)
    N = length(q.nodes)
    
    # 1. Morphism Composition & Relation Generation
    for ((i, j), morph) in q.morphisms
        avg_A = (q.loads_A[i] + q.loads_A[j]) / 2.0
        
        # Relation I: Molecule A forces degree 2 paths to vanish
        # This is the "Breaking of Gluing"
        relation_strength = exp(-avg_A * morph.stiffness)
        morph.ee = (morph.e * morph.e) * relation_strength
        
        # Degree 1 Arrow Rotation (Spectral flow)
        morph.e *= exp(im * 10.0 * dt)
    end

    # 2. Stalk Dynamics (PFC Prior)
    for i in 1:N
        n = q.nodes[i]
        A = q.loads_A[i]
        p = q.priors[i]
        
        # Linear interaction on the stalk
        # No 'caps'—just Hopf bifurcation stiffness
        r2 = abs2(n.z)
        mu = (1.0 + (A * p.p_sens) - r2) 
        ω_eff = n.ω_base + (A * p.ω_sens)
        
        n.z += (Complex(mu, ω_eff) * n.z) * dt
        n.mass = abs2(n.z)
    end
end

# ======================================================
# === 4. Simulation: Watching the Wall Crossing
# ======================================================

function run_prolate_path_algebra(N=300, tmax=150)
    # Initialize PFC Priors (Gamma Slowing / Theta Disinhibition)
    pfc = RegionalPrior(:PFC, -20.0, 0.6)
    nodes = [Stalk(0.1+0.0im, 40.0, 1.0) for _ in 1:N]
    priors = [pfc for _ in 1:N]
    loads_A = zeros(N)

    # Path Morphisms (The Gluing structure)
    morphisms = Dict{Tuple{Int, Int}, PathMorphism}()
    for i in 1:N-1
        # Arrow e is initialized on unit circle
        morphisms[(i, i+1)] = PathMorphism(exp(im*rand()*2π), 1.0, 3.0)
    end
    
    q = MoritaQuiver(nodes, priors, morphisms, loads_A)
    dt = 0.02
    
    # Metric Logs
    gap_h = zeros(tmax)
    g_h, t_h = zeros(tmax), zeros(tmax)

    for t in 1:tmax
        # Molecule A propagates from the PFC edge
        q.loads_A[N] = 3.0
        for i in N:-1:2
            q.loads_A[i-1] += (q.loads_A[i] - q.loads_A[i-1]) * 0.1
        end

        # Evolve Algebra and Observe Gap
        evolve_quiver_algebra!(q, dt)
        gap_h[t] = compute_prolate_gap(q)

        # Track Phase Transition Metrics
        g_h[t] = mean(abs(n.z) for n in q.nodes) # Realized Gamma Power
    end

    # Plotting: Prolate Gap vs. Resulting Transition
    p = plot(gap_h, label="Prolate Gap (Relation Spike)", color=:black, lw=2, ylabel="Spectral Tension")
    plot!(p, g_h, label="Gamma Power (Slowing)", color=:green, twinx=true, ylabel="Amplitude", ylim=(0, 2.0))
    title!("Reverse Hironaka: Path Algebra degree 2 breakdown")
    savefig("toy_quiver2.png")
    return p
end

run_prolate_path_algebra()