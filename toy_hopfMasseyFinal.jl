using LinearAlgebra
using Statistics
using Plots

# === A∞-Algebra for Massey Products ===
mutable struct AInfinityAlgebra
    μ₂::Matrix{ComplexF64}
    μ₃::Array{ComplexF64,3}
    degree::Int
    cohomology_class::Vector{ComplexF64}
end

mutable struct HopfOscillator
    amplitude::Float64
    phase::Float64
    frequency::Float64
end

# === Prolate Operator ===
function prolate_operator(N::Int, bandwidth::Float64)
    P = zeros(ComplexF64, N, N)
    for i in 1:N
        for j in 1:N
            if i == j
                P[i,j] = bandwidth
            else
                x = π * bandwidth * (i - j)
                P[i,j] = x == 0 ? bandwidth : sin(x) / x
            end
        end
    end
    return Hermitian(P)
end

# === Massey Triple Product ===
function massey_triple_product(algebra::AInfinityAlgebra, x::ComplexF64, y::ComplexF64, z::ComplexF64)
    n = size(algebra.μ₂, 1)
    
    # Convert to vectors
    x_vec = [real(x), imag(x), 0.0][1:n]
    y_vec = [real(y), imag(y), 0.0][1:n]
    z_vec = [real(z), imag(z), 0.0][1:n]
    
    # μ₃ term
    μ₃_term = 0.0
    for i in 1:n, j in 1:n, k in 1:n
        μ₃_term += abs(algebra.μ₃[i,j,k] * x_vec[j] * y_vec[k] * z_vec[1])
    end
    
    # Simple obstruction measure
    obstruction = μ₃_term / (n^3 + 1e-10)
    return obstruction
end

# === Main Simulation ===
function simulate_molecule_transitions(N_sites=40, N_steps=80)
    # Initialize
    positions = LinRange(0, 1, N_sites)
    
    # A∞-algebras
    algebras = [AInfinityAlgebra(
        randn(ComplexF64, 2, 2) * 0.1,
        randn(ComplexF64, 2, 2, 2) * 0.01,
        mod(i, 2),
        normalize(randn(ComplexF64, 2))
    ) for i in 1:N_sites]
    
    # Oscillators
    oscillators = [HopfOscillator(1.0, 2π*rand(), 1.0) for _ in 1:N_sites]
    
    # Densities
    density_A = [x < 0.5 ? 1.0 - 0.5*x : 0.1 for x in positions]
    density_B = 1.0 .- density_A
    
    # Prolate operator
    P = prolate_operator(N_sites, 0.3)
    
    # Histories
    frequencies = zeros(N_steps, N_sites)
    massey_history = zeros(N_steps, N_sites-2)
    prolate_gaps = zeros(N_steps)
    density_history = zeros(N_steps, N_sites)
    
    println("Starting simulation...")
    
    for t in 1:N_steps
        # Diffuse molecules
        if t > 1
            diffusion = 0.02
            for i in 2:N_sites-1
                flux = diffusion * (density_B[i-1] - 2*density_B[i] + density_B[i+1])
                density_B[i] += flux
            end
            density_B[1] = 0.0
            density_B[end] = 1.0
            density_A = 1.0 .- density_B
        end
        
        density_history[t, :] = density_B
        
        # Update frequencies based on density
        for i in 1:N_sites
            ratio = density_B[i] / (density_A[i] + 1e-10)
            oscillators[i].frequency = 1.0 + 1.5 * ratio
        end
        
        # Evolve oscillators
        for (i, osc) in enumerate(oscillators)
            osc.amplitude += 0.01 * (osc.amplitude - osc.amplitude^3)
            osc.phase += 0.01 * osc.frequency
            frequencies[t, i] = osc.frequency
        end
        
        # Compute Massey products
        for i in 1:N_sites-2
            x = oscillators[i].amplitude * exp(im * oscillators[i].phase)
            y = oscillators[i+1].amplitude * exp(im * oscillators[i+1].phase)
            z = oscillators[i+2].amplitude * exp(im * oscillators[i+2].phase)
            
            massey_history[t, i] = massey_triple_product(algebras[i], x, y, z)
        end
        
        # Prolate gap
        eigvals = eigen(P).values
        prolate_gaps[t] = real(eigvals[2] - eigvals[1])
        
        # Progress
        if t % 20 == 0
            println("t=$t: Massey=$(round(mean(massey_history[t,:]), digits=4)), Gap=$(round(prolate_gaps[t], digits=4))")
        end
    end
    
    return frequencies, massey_history, prolate_gaps, density_history, positions
end

# === SIMPLE Plotting ===
function plot_results_simple(results)
    frequencies, massey_history, prolate_gaps, density_history, positions = results
    
    # Create individual plots
    p1 = heatmap(density_history',
                 title="Molecule B Density",
                 xlabel="Time", ylabel="Position",
                 color=:viridis)
    
    p2 = heatmap(massey_history',
                 title="Massey Obstruction",
                 xlabel="Time", ylabel="Position",
                 color=:inferno)
    
    p3 = plot(prolate_gaps, lw=2,
              title="Prolate Spectral Gap",
              xlabel="Time", ylabel="Δλ",
              color=:blue, legend=false)
    
    # Correlation plot (without vspan to avoid errors)
    density_flat = vec(density_history[:, 1:end-2])  # Match massey dimensions
    massey_flat = vec(massey_history)
    
    p4 = scatter(density_flat, massey_flat,
                 alpha=0.1, marker=:., markersize=2,
                 title="Massey vs Density",
                 xlabel="Density ϕ_B",
                 ylabel="Massey Obstruction",
                 legend=false)
    
    # Combine plots
    plot(p1, p2, p3, p4, layout=(4,1), size=(800, 1000))
    savefig("massey_results_simple.png")
    println("Plot saved as massey_results_simple.png")
    
    # Print statistics
    println("\n=== Statistics ===")
    println("Max Massey obstruction: ", maximum(massey_flat))
    println("Average Massey: ", mean(massey_flat))
    println("Prolate gap range: ", extrema(prolate_gaps))
    
    # Check for phase transitions (simple threshold)
    massey_threshold = 0.5 * maximum(massey_flat)
    transitions = findall(x -> x > massey_threshold, massey_flat)
    if !isempty(transitions)
        println("Potential phase transitions detected: ", length(transitions))
    end
end

# === Run ===
println("="^50)
println("Molecule A→B with Massey Products")
println("="^50)

results = simulate_molecule_transitions(30, 60)
plot_results_simple(results)

println("\n✓ Simulation complete!")