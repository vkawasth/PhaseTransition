using LinearAlgebra, Statistics, Plots

# ======================================================
# === 1. Structural Stalks: Unitary Invariants
# ======================================================

mutable struct HopfStalk
    z::ComplexF64
    ω_local::Float64   # Includes 5% variance for wave interference
    power::Float64     # Target manifold (0.9 high, 0.3 low)
end

mutable struct MoritaNode
    bands::Dict{Symbol, HopfStalk}
    load_A::Float64    # Disruptor Locus (Molecule A)
end

# ======================================================
# === 2. Initialization: Power Manifold Separation
# ======================================================

function init_pfc_quiver(N=300)
    nodes = MoritaNode[]
    low_p = 0.3
    high_p = 0.9 # 3x the low power manifold

    freqs = Dict(:gamma=>40.0, :beta=>20.0, :alpha=>10.0, :theta=>6.0, :delta=>2.0)

    for i in 1:N
        bands = Dict{Symbol, HopfStalk}()
        for (name, base_ω) in freqs
            # 5% jitter ensures oscillators are not identical (prevents dead sync)
            jitter = 1.0 + (rand() * 0.1 - 0.05) 
            p = (name == :gamma || name == :theta) ? high_p : low_p
            
            # Unitary initialization on the specific manifold
            z_init = p * exp(im * rand() * 2π)
            bands[name] = HopfStalk(z_init, base_ω * jitter, p)
        end
        push!(nodes, MoritaNode(bands, 0.0))
    end
    return nodes
end

# ======================================================
# === 3. Dynamics: Unitary Poisson Rotation
# ======================================================

function update_quiver!(nodes::Vector{MoritaNode}, dt::Float64)
    for n in nodes
        A = n.load_A
        for (name, stalk) in n.bands
            # Molecule A impacts the local frequency (PFC Prior)
            # Slows Gamma/Delta, speeds Theta (disinhibition)
            shift = 0.0
            if name == :gamma; shift = -20.0 * A; end
            if name == :delta; shift = -5.0 * A; end
            if name == :theta; shift = 4.0 * A; end
            
            # Rotational update: Magnitude |z| is exactly conserved
            stalk.z *= exp(im * (stalk.ω_local + shift) * dt)
        end
    end
end

# ======================================================
# === 4. Prolate Observer (Jacobi Gap)
# ======================================================

function get_prolate_gap(nodes::Vector{MoritaNode})
    N = length(nodes)
    # Spectral contrast gradient driven by Molecule A
    d = [-n.load_A for n in nodes]
    dl = fill(-0.65, N-1) # Path Algebra degree 1 morphisms
    
    J = Tridiagonal(dl, d, dl)
    λ = eigvals(Hermitian(Matrix(J)))
    return λ[end] - λ[end-1]
end

# ======================================================
# === 5. Simulation & Neuro-Signal Plotting
# ======================================================

function run_final_audit(N=300, tmax=300)
    nodes = init_pfc_quiver(N)
    dt = 0.01
    
    # Storage for 5 bands + Prolate Gap
    g_p, b_p, a_p, t_p, d_p, gap_h = [zeros(tmax) for _ in 1:6]

    for t in 1:tmax
        # Disruptor arrives at t=50
        if t > 50
            nodes[N].load_A = 2.5
            for i in N:-1:2
                nodes[i-1].load_A += (nodes[i].load_A - nodes[i-1].load_A) * 0.08
            end
        end

        update_quiver!(nodes, dt)
        
        # KEY FIX: Plotting mean of magnitudes ensures manifold separation
        g_p[t] = mean(abs(n.bands[:gamma].z) for n in nodes)
        b_p[t] = mean(abs(n.bands[:beta].z) for n in nodes)
        a_p[t] = mean(abs(n.bands[:alpha].z) for n in nodes)
        t_p[t] = mean(abs(n.bands[:theta].z) for n in nodes)
        d_p[t] = mean(abs(n.bands[:delta].z) for n in nodes)
        
        gap_h[t] = get_prolate_gap(nodes)
    end

    # Plotting: Mimicking separated power bands (e.g., PMC PMC6673990)
    p = plot(gap_h, label="Prolate Gap (Spike)", color=:black, lw=3, ylabel="Spectral Gap")
    
    plot!(p, g_p, label="Gamma (High)", color=:green, lw=2, twinx=true, ylabel="% Signal Change", ylim=(0, 1.2))
    plot!(p, t_p, label="Theta (3x Low)", color=:red, lw=2)
    plot!(p, a_p, label="Alpha", color=:blue, alpha=0.6)
    plot!(p, b_p, label="Beta", color=:cyan, alpha=0.6)
    plot!(p, d_p, label="Delta", color=:purple, alpha=0.6)
    
    title!("Final PFC Audit: Stable Manifolds & Prolate Precursor")
    savefig("toy_quiver3.png")
    return p
end

run_final_audit()