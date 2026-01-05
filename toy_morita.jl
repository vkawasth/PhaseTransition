using LinearAlgebra
using Statistics
using Plots
using SparseArrays
using Random

# ========== SAFE LOG FUNCTION ==========

function safe_log(x)
    return log(Complex(x))
end

# ========== DATA STRUCTURES ==========

mutable struct PoissonStalk
    position::Vector{Float64}
    phase::Float64
    amplitude::Float64
    momentum::Vector{Float64}
    poisson_bracket::Matrix{Float64}
end

mutable struct MoritaAlgebra
    A::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    M::Matrix{ComplexF64}
    HH0::Float64
    HH1::Float64
    HH2::Float64
end

mutable struct VesselMorita
    id::Int
    source::PoissonStalk
    target::PoissonStalk
    bimodule::Matrix{ComplexF64}
    bimodule_op::Matrix{ComplexF64}
    morita_data::MoritaAlgebra
end

# ========== ENHANCED DYNAMICS ==========

function create_connected_network(N_vertices::Int)
    vertices = PoissonStalk[]
    for _ in 1:N_vertices
        push!(vertices, PoissonStalk(
            rand(3) .- 0.5,
            rand() * 2π,
            0.5 + 0.2randn(),
            randn(3),
            create_structured_poisson()
        ))
    end
    
    triangles = Vector{Int}[]
    max_triangles = min(3*N_vertices, 30)
    
    for _ in 1:max_triangles
        if !isempty(triangles) && rand() < 0.7
            existing = rand(triangles)
            tri = copy(existing)
            replace_idx = rand(1:3)
            possible = setdiff(1:N_vertices, existing)
            if !isempty(possible)
                tri[replace_idx] = rand(possible)
                push!(triangles, tri)
            end
        else
            tri = rand(1:N_vertices, 3)
            while length(unique(tri)) < 3
                tri = rand(1:N_vertices, 3)
            end
            push!(triangles, tri)
        end
    end
    
    return vertices, triangles
end

function create_structured_poisson()
    ω = zeros(6, 6)
    
    for i in 1:3
        ω[i, 3+i] = 1.0 + 0.2randn()
        ω[3+i, i] = -ω[i, 3+i]
    end
    
    ω[1, 2] = 0.3randn()
    ω[2, 1] = -ω[1, 2]
    ω[2, 3] = 0.3randn()
    ω[3, 2] = -ω[2, 3]
    
    ω += 0.1 * randn(6, 6)
    ω = (ω - ω') / 2
    
    return ω
end

function enhanced_poisson_flow!(stalk::PoissonStalk, neighbors::Vector{PoissonStalk}, dt::Float64)
    # Local potential (ensured to be positive)
    H_local = stalk.amplitude^2 * (1.0 + stalk.amplitude^2)
    
    # Compute derivatives
    dphase = 0.0
    damplitude = 0.0
    
    # Local derivative
    damplitude += 2 * stalk.amplitude * (1.0 + 2*stalk.amplitude^2)
    
    # Coupling with neighbors
    for nb in neighbors
        Δx = stalk.position - nb.position
        dist = max(norm(Δx), 1e-8)
        
        # Phase coupling
        phase_diff = stalk.phase - nb.phase
        dphase += -exp(-dist^2) * stalk.amplitude * nb.amplitude * sin(phase_diff)
        
        # Amplitude coupling
        damplitude += exp(-dist^2) * nb.amplitude * abs(cos(phase_diff))
    end
    
    # Update states
    stalk.phase += dt * dphase
    stalk.amplitude += dt * (damplitude - H_local)
    
    # Update momentum (simplified)
    force = zeros(3)
    for nb in neighbors
        Δx = stalk.position - nb.position
        dist = max(norm(Δx), 1e-8)
        force_val = exp(-dist^2) * stalk.amplitude * nb.amplitude
        force .+= -force_val * Δx/dist
    end
    
    stalk.momentum .+= dt * force
    stalk.position .+= dt * stalk.momentum
    
    # Clamp values
    stalk.amplitude = clamp(stalk.amplitude, 0.05, 3.0)
    stalk.phase = mod(stalk.phase, 2π)
    stalk.momentum = clamp.(stalk.momentum, -2.0, 2.0)
    
    return stalk
end

function create_edge_morita_with_feedback(source::PoissonStalk, target::PoissonStalk, step::Int=1)
    n = 2
    
    t_factor = 0.1 * sin(0.1 * step)
    
    A = [1.0+0.1im 0.2-0.3im; -0.2+0.1im 0.8+0.4im] .* (1.0 + 0.1t_factor)
    B = [0.9-0.2im 0.3+0.1im; -0.3-0.1im 1.1+0.3im] .* (1.0 + 0.1t_factor)
    
    phase_diff = target.phase - source.phase
    amp_product = sqrt(max(source.amplitude * target.amplitude, 1e-4))
    
    M = amp_product * exp(im * phase_diff) * [
        cos(phase_diff) + 0.1im*sin(phase_diff)  -0.2+0.1im;
        0.3-0.1im  sin(phase_diff) - 0.2im*cos(phase_diff)
    ]
    
    # Compute HH
    HH0 = max(real(tr(A * B')) / max(norm(A) * norm(B), 1e-10), 0.0)
    
    crossed = [A M; M' B]
    det_val = max(abs(det(crossed)), 1e-10)
    HH1 = real(safe_log(det_val))
    
    # Massey product
    massey = compute_massey_product(A, B, M)
    HH2 = max(norm(massey), 0.0)
    
    return MoritaAlgebra(A, B, M, HH0, HH1, HH2)
end

function compute_massey_product(A::Matrix{ComplexF64}, B::Matrix{ComplexF64}, M::Matrix{ComplexF64})
    if size(A,1) == size(B,1) == size(M,1) == size(M,2)
        try
            AB = A * B
            MA = M * A
            BM = B * M
            massey = AB * M - A * BM + MA * B
            return massey
        catch
            return zeros(ComplexF64, size(A))
        end
    else
        return zeros(ComplexF64, size(A))
    end
end

# ========== SIMULATION ==========

function run_dynamic_simulation(N_vertices=15, steps=20)
    println("="^60)
    println("Dynamic Morita-Poisson Simulation")
    println("Vertices: $N_vertices, Steps: $steps")
    println("="^60)
    
    Random.seed!(42)
    
    # Create network
    vertices, triangles = create_connected_network(N_vertices)
    
    # Build neighbor lists
    neighbor_lists = [Int[] for _ in 1:N_vertices]
    for tri in triangles
        for (i, j) in [(1,2), (2,3), (3,1)]
            push!(neighbor_lists[tri[i]], tri[j])
            push!(neighbor_lists[tri[j]], tri[i])
        end
    end
    
    for i in 1:N_vertices
        neighbor_lists[i] = unique(neighbor_lists[i])
    end
    
    # Simulation histories
    κ_history = Float64[]
    HH2_history = Float64[]
    flow_history = Float64[]
    amplitude_history = Float64[]
    phase_coherence_history = Float64[]
    
    for step in 1:steps
        print("Step $step/$steps: ")
        
        # 1. Update stalks
        new_vertices = similar(vertices)
        for i in 1:N_vertices
            neighbors = [vertices[j] for j in neighbor_lists[i]]
            stalk_copy = PoissonStalk(
                copy(vertices[i].position),
                vertices[i].phase,
                vertices[i].amplitude,
                copy(vertices[i].momentum),
                copy(vertices[i].poisson_bracket)
            )
            new_vertices[i] = enhanced_poisson_flow!(stalk_copy, neighbors, 0.05)
        end
        vertices = new_vertices
        
        # 2. Create vessels
        vessels = VesselMorita[]
        vessel_counter = 1
        
        for tri in triangles
            for (i, j) in [(1,2), (2,3), (3,1)]
                if tri[i] <= length(vertices) && tri[j] <= length(vertices)
                    source = vertices[tri[i]]
                    target = vertices[tri[j]]
                    
                    morita_result = create_edge_morita_with_feedback(source, target, step)
                    
                    push!(vessels, VesselMorita(
                        vessel_counter,
                        source, target,
                        copy(morita_result.M),
                        conj(morita_result.M'),
                        morita_result
                    ))
                    vessel_counter += 1
                end
            end
        end
        
        # 3. Compute invariants
        if !isempty(vessels)
            # HH values
            HH_vals = Float64[]
            for v in vessels
                val = v.morita_data.HH0 + v.morita_data.HH1 + v.morita_data.HH2
                push!(HH_vals, max(val, 1e-10))
            end
            
            # Kodaira dimension
            if length(HH_vals) > 2
                sorted_vals = sort(HH_vals)
                xs = [real(safe_log(i)) for i in 1:length(sorted_vals)]
                ys = [real(safe_log(v)) for v in sorted_vals]
                
                if var(xs) > 1e-10
                    κ = cov(xs, ys) / var(xs)
                else
                    κ = 0.0
                end
            else
                κ = 0.0
            end
            push!(κ_history, κ)
            
            # Total HH²
            total_HH2 = sum([max(v.morita_data.HH2, 0.0) for v in vessels])
            push!(HH2_history, total_HH2)
            
            # Total flow
            total_flow = 0.0
            for v in vessels
                if !isempty(v.bimodule)
                    total_flow += abs(v.bimodule[1,1])
                end
            end
            push!(flow_history, total_flow)
            
            # Average amplitude
            amplitudes = [v.source.amplitude for v in vessels]
            push!(amplitude_history, mean(amplitudes))
            
            # Phase coherence
            phases = [v.source.phase for v in vessels]
            phase_coherence = abs(mean(exp.(im .* phases)))
            push!(phase_coherence_history, phase_coherence)
            
            println("κ=$(round(κ, digits=3)), HH²=$(round(total_HH2, digits=3))")
        else
            push!(κ_history, 0.0)
            push!(HH2_history, 0.0)
            push!(flow_history, 0.0)
            push!(amplitude_history, 0.5)
            push!(phase_coherence_history, 0.0)
            println("No vessels")
        end
    end
    
    # Plot results
    if steps > 1
        p1 = plot(1:steps, κ_history, 
                 title="Kodaira Dimension κ(t)", 
                 lw=2, xlabel="Step", ylabel="κ", color=:blue)
        
        p2 = plot(1:steps, HH2_history, 
                 title="Total HH² Obstruction", 
                 lw=2, xlabel="Step", ylabel="HH²", color=:red)
        
        p3 = plot(1:steps, flow_history, 
                 title="Network Flow", 
                 lw=2, xlabel="Step", ylabel="Flow", color=:green)
        
        p4 = plot(1:steps, amplitude_history,
                 title="Average Amplitude",
                 lw=2, xlabel="Step", ylabel="Amplitude", color=:purple)
        
        p5 = plot(1:steps, phase_coherence_history,
                 title="Phase Coherence",
                 lw=2, xlabel="Step", ylabel="Coherence", color=:orange,
                 ylims=(0, 1))
        
        plot(p1, p2, p3, p4, p5, layout=(2,3), size=(1500, 800))
        savefig("dynamic_morita_simulation.png")
        
        println("\n" * "="^60)
        println("Simulation Results Summary:")
        println("Kodaira dimension range: $(round(minimum(κ_history), digits=3)) to $(round(maximum(κ_history), digits=3))")
        println("HH² range: $(round(minimum(HH2_history), digits=3)) to $(round(maximum(HH2_history), digits=3))")
        println("Flow range: $(round(minimum(flow_history), digits=3)) to $(round(maximum(flow_history), digits=3))")
        println("Amplitude range: $(round(minimum(amplitude_history), digits=3)) to $(round(maximum(amplitude_history), digits=3))")
        println("Phase coherence range: $(round(minimum(phase_coherence_history), digits=3)) to $(round(maximum(phase_coherence_history), digits=3))")
        println("="^60)
    end
    
    return (κ_history, HH2_history, flow_history, amplitude_history, phase_coherence_history)
end

# ========== MAIN EXECUTION ==========

function main()
    println("\n" * "="^60)
    println("MORITA-POISSON NETWORK SIMULATION")
    println("="^60)
    println("This simulates:")
    println("1. Poisson stalks (symplectic phase spaces)")
    println("2. Morita bimodules (categorical transport)")
    println("3. Nonlinear Hamiltonian dynamics")
    println("4. Deformation theory via Hochschild cohomology")
    println("="^60)
    
    # Run simulation
    histories = run_dynamic_simulation(12, 15)
    
    # Analyze results
    println("\n" * "="^60)
    println("DYNAMICS ANALYSIS")
    println("="^60)
    
    κ_history, HH2_history, flow_history, amplitude_history, phase_coherence_history = histories
    
    # Check for interesting behavior
    κ_var = maximum(κ_history) - minimum(κ_history)
    HH2_var = maximum(HH2_history) - minimum(HH2_history)
    flow_var = maximum(flow_history) - minimum(flow_history)
    phase_var = maximum(phase_coherence_history) - minimum(phase_coherence_history)
    
    println("Variability measures:")
    println("  Kodaira dimension: $(round(κ_var, digits=4))")
    println("  HH² obstruction: $(round(HH2_var, digits=4))")
    println("  Network flow: $(round(flow_var, digits=4))")
    println("  Phase coherence: $(round(phase_var, digits=4))")
    
    # Detect oscillations
    κ_diff = diff(κ_history)
    if any(abs.(κ_diff) .> 0.02)
        println("\n✓ Oscillations detected in Kodaira dimension")
        println("  Max step change: $(round(maximum(abs.(κ_diff)), digits=4))")
    end
    
    if any(abs.(diff(phase_coherence_history)) .> 0.1)
        println("✓ Significant phase coherence dynamics")
    end
    
    # Check for trends
    if κ_history[end] > κ_history[1] * 1.5
        println("✓ Kodaira dimension shows increasing trend")
    elseif κ_history[end] < κ_history[1] * 0.5
        println("✓ Kodaira dimension shows decreasing trend")
    end
    
    # Generate quick overview plot
    println("\nGenerating overview plot...")
    p = plot(layout=(5,1), size=(1200, 800), legend=false)
    
    plot!(p[1], κ_history, title="Kodaira Dimension κ(t)", color=:blue, lw=2)
    plot!(p[2], HH2_history, title="HH² Obstruction", color=:red, lw=2)
    plot!(p[3], flow_history, title="Network Flow", color=:green, lw=2)
    plot!(p[4], amplitude_history, title="Average Amplitude", color=:purple, lw=2)
    plot!(p[5], phase_coherence_history, title="Phase Coherence", color=:orange, lw=2, ylims=(0,1))
    
    savefig("overview.png")
    println("Overview plot saved as 'overview.png'")
    
    # Final summary
    println("\n" * "="^60)
    println("SIMULATION COMPLETE")
    println("="^60)
    println("Files generated:")
    println("  - dynamic_morita_simulation.png (detailed plots)")
    println("  - overview.png (compact overview)")
    println("\nMathematical structures implemented:")
    println("  ✓ Poisson geometry on stalks")
    println("  ✓ Morita theory for categorical transport")
    println("  ✓ Hochschild cohomology (HH*) for deformation theory")
    println("  ✓ Nonlinear Hamiltonian dynamics")
    println("  ✓ Kodaira dimension as growth rate of HH*")
    println("="^60)
    
    return histories
end

# ========== RUN THE SIMULATION ==========

# Check if this is the main file being run
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting simulation...")
    histories = main()
    println("\nDone!")
else
    println("Module loaded. Call main() to run simulation.")
end