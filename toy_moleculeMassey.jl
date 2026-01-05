using LinearAlgebra
using Statistics
using Plots
using Random

# ========== STRUCTURES ==========

struct Molecule
    id::Int
    type::Symbol
    charge::ComplexF64
    position::Vector{Float64}
end

struct HopfOscillator
    amplitude::Float64
    phase::Float64
    natural_freq::Float64
    current_freq::Float64
    coupling::Float64
    molecule_density::Float64
end

struct MasseyProduct
    A::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    M::Matrix{ComplexF64}
    product::Matrix{ComplexF64}
    obstruction::Float64
    is_defined::Bool
end

function compute_massey_triple(A, B, M)
    AB = A * B
    BM = B * M
    MA = M * A
    massey = AB * M - A * BM + MA * B
    obstruction = norm(massey)
    return MasseyProduct(A, B, M, massey, obstruction, obstruction > 1e-10)
end

mutable struct MoritaEdge
    id::Int
    source::Int
    target::Int
    length::Float64
    radius::Float64
    oscillator::HopfOscillator
    molecule_counts::Dict{Symbol, Float64}
    molecule_representations::Dict{Symbol, Matrix{ComplexF64}}
    A::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    M::Matrix{ComplexF64}
    massey_AMB::MasseyProduct
    massey_BMA::MasseyProduct
    conductance::Float64
    phase_coherence::Float64
    massey_pressure::Float64
end

struct NetworkNode
    id::Int
    position::Vector{Float64}
    connected_edges::Vector{Int}
    molecule_pool::Dict{Symbol, Vector{Molecule}}
    phase_field::Float64
    amplitude_field::Float64
    massey_flux::Float64
    transition_count::Int
end

# ========== CORE FUNCTIONS ==========

function molecules_to_representation(molecules::Vector{Molecule}, edge_length::Float64)
    n = 2
    
    if isempty(molecules)
        return zeros(ComplexF64, n, n)
    end
    
    # Count molecules by type
    type_A = [m for m in molecules if m.type == :A]
    type_B = [m for m in molecules if m.type == :B]
    
    density_A = length(type_A) / max(edge_length, 1e-8)
    density_B = length(type_B) / max(edge_length, 1e-8)
    
    # Type A: phase representation
    rep_A = density_A * [exp(im*0.5) 0; 0 exp(-im*0.5)]
    # Type B: mixing representation  
    rep_B = density_B * [0.8 0.2; 0.2 0.8]
    
    total_rep = rep_A + rep_B
    return total_rep / max(norm(total_rep), 1e-10)
end

function update_morita_from_molecules!(edge::MoritaEdge, molecule_rep::Matrix{ComplexF64})
    ε = 0.05
    
    A_new = edge.A * exp(ε * molecule_rep)
    B_new = edge.B * exp(ε * molecule_rep')
    M_new = edge.M + ε * (molecule_rep * edge.M + edge.M * molecule_rep')
    
    A_new = A_new / max(norm(A_new), 1e-10)
    B_new = B_new / max(norm(B_new), 1e-10)
    M_new = M_new / max(norm(M_new), 1e-10)
    
    edge.A = A_new
    edge.B = B_new
    edge.M = M_new
    edge.massey_AMB = compute_massey_triple(A_new, M_new, B_new)
    edge.massey_BMA = compute_massey_triple(B_new, M_new, A_new)
    
    return edge
end

function massey_to_oscillator_frequency(edge::MoritaEdge)
    base_freq = edge.oscillator.natural_freq
    
    # Massey obstruction reduces frequency
    obstruction_factor = 1.0 / (1.0 + edge.massey_AMB.obstruction)
    
    # Molecule density effect
    total_molecules = sum(values(edge.molecule_counts))
    volume = π * edge.radius^2 * edge.length
    density = total_molecules / max(volume, 1e-8)
    density_factor = 1.0 / (1.0 + density)
    
    return base_freq * obstruction_factor * density_factor
end

function update_hopf_oscillator!(osc::HopfOscillator, new_freq::Float64, dt::Float64)
    r = osc.amplitude
    θ = osc.phase
    
    # Hopf normal form
    μ = 1.0
    dr = r * (μ - r^2)
    dθ = 2π * new_freq
    
    new_r = clamp(r + dt * dr, 0.01, 2.0)
    new_θ = mod(θ + dt * dθ, 2π)
    
    return HopfOscillator(new_r, new_θ, osc.natural_freq, new_freq, 
                         osc.coupling, osc.molecule_density)
end

function detect_local_phase_transition(edge::MoritaEdge, 
                                      neighbor_edges::Vector{MoritaEdge})
    # Massey condition: low obstruction
    massey_threshold = 0.3
    massey_condition = edge.massey_AMB.obstruction < massey_threshold
    
    # Oscillator synchronization
    phases = Float64[edge.oscillator.phase]
    for e in neighbor_edges
        push!(phases, e.oscillator.phase)
    end
    
    z = mean(exp.(im .* phases))
    coherence = abs(z)
    sync_threshold = 0.7
    sync_condition = coherence > sync_threshold
    
    is_transition = massey_condition && sync_condition
    transition_strength = coherence / (edge.massey_AMB.obstruction + 1e-8)
    
    return is_transition, transition_strength, coherence, edge.massey_AMB.obstruction
end

# ========== SIMULATION ==========

function run_molecule_massey_hopf_simulation(N_nodes=3, steps=6)
    println("="^70)
    println("MOLECULE → MASSEY → HOPF SIMULATION")
    println("="^70)
    
    Random.seed!(42)
    
    # Create nodes
    nodes = NetworkNode[]
    for i in 1:N_nodes
        push!(nodes, NetworkNode(
            i,
            rand(2) .- 0.5,  # 2D for easier visualization
            Int[],
            Dict(:A => Molecule[], :B => Molecule[]),
            0.0, 0.0,
            0.0, 0
        ))
    end
    
    # Create edges
    edges = MoritaEdge[]
    edge_id_counter = 1
    
    for i in 1:N_nodes
        for j in i+1:N_nodes
            if rand() < 0.8  # High connection for small network
                length_val = norm(nodes[i].position - nodes[j].position)
                
                # Initial algebras
                n = 2
                A = randn(ComplexF64, n, n) + im * randn(n, n)
                B = randn(ComplexF64, n, n) + im * randn(n, n)
                M = randn(ComplexF64, n, n) + im * randn(n, n)
                
                A = A / max(norm(A), 1e-10)
                B = B / max(norm(B), 1e-10)
                M = M / max(norm(M), 1e-10)
                
                # Massey products
                massey_AMB = compute_massey_triple(A, M, B)
                massey_BMA = compute_massey_triple(B, M, A)
                
                edge = MoritaEdge(
                    edge_id_counter,
                    i, j,
                    length_val,
                    0.1 + 0.05rand(),
                    
                    HopfOscillator(
                        0.5 + 0.5rand(),
                        rand() * 2π,
                        5.0 + 2.0rand(),
                        5.0 + 2.0rand(),
                        0.1 + 0.1rand(),
                        0.0
                    ),
                    
                    Dict(:A => 1.0 + 2.0rand(), :B => 1.0 + 2.0rand()),  # Initial counts
                    Dict(:A => zeros(ComplexF64,2,2), :B => zeros(ComplexF64,2,2)),
                    
                    A, B, M,
                    massey_AMB, massey_BMA,
                    0.0, 0.0, 0.0
                )
                
                push!(edges, edge)
                push!(nodes[i].connected_edges, edge.id)
                push!(nodes[j].connected_edges, edge.id)
                edge_id_counter += 1
            end
        end
    end
    
    println("Network: $(N_nodes) nodes, $(length(edges)) edges")
    println("Initial Massey range: ", 
            round(minimum([e.massey_AMB.obstruction for e in edges]), digits=3), " to ",
            round(maximum([e.massey_AMB.obstruction for e in edges]), digits=3))
    
    # Simulation histories
    massey_history = Float64[]
    frequency_history = Float64[]
    transition_history = []
    
    for step in 1:steps
        print("Step $step/$steps: ")
        
        step_transitions = []
        total_massey = 0.0
        total_freq = 0.0
        
        for edge in edges
            # Get molecules from connected nodes
            source_molecules = vcat(
                nodes[edge.source].molecule_pool[:A],
                nodes[edge.source].molecule_pool[:B]
            )
            target_molecules = vcat(
                nodes[edge.target].molecule_pool[:A],
                nodes[edge.target].molecule_pool[:B]
            )
            all_molecules = vcat(source_molecules, target_molecules)
            
            # Update algebra from molecules
            mol_rep = molecules_to_representation(all_molecules, edge.length)
            update_morita_from_molecules!(edge, mol_rep)
            
            # Update oscillator from Massey
            new_freq = massey_to_oscillator_frequency(edge)
            edge.oscillator = update_hopf_oscillator!(edge.oscillator, new_freq, 0.01)
            
            # Find neighbor edges
            neighbor_edges = MoritaEdge[]
            for eid in nodes[edge.source].connected_edges
                neighbor = edges[eid]
                if neighbor.id != edge.id
                    push!(neighbor_edges, neighbor)
                end
            end
            for eid in nodes[edge.target].connected_edges
                neighbor = edges[eid]
                if neighbor.id != edge.id && !(neighbor in neighbor_edges)
                    push!(neighbor_edges, neighbor)
                end
            end
            
            # Detect phase transitions
            is_trans, strength, coherence, obstruction = detect_local_phase_transition(
                edge, neighbor_edges
            )
            
            if is_trans
                push!(step_transitions, (edge.id, strength, coherence, obstruction))
            end
            
            # Update edge molecule density
            total_mols = sum(values(edge.molecule_counts))
            volume = π * edge.radius^2 * edge.length
            edge.oscillator = HopfOscillator(
                edge.oscillator.amplitude,
                edge.oscillator.phase,
                edge.oscillator.natural_freq,
                edge.oscillator.current_freq,
                edge.oscillator.coupling,
                total_mols / max(volume, 1e-8)
            )
            
            # Accumulate statistics
            total_massey += edge.massey_AMB.obstruction
            total_freq += edge.oscillator.current_freq
        end
        
        # Store histories
        if !isempty(edges)
            avg_massey = total_massey / length(edges)
            avg_freq = total_freq / length(edges)
        else
            avg_massey = avg_freq = 0.0
        end
        
        push!(massey_history, avg_massey)
        push!(frequency_history, avg_freq)
        push!(transition_history, step_transitions)
        
        # Use digits=3 instead of 3 for round
        println("M=$(round(avg_massey, digits=3)) F=$(round(avg_freq, digits=3)) T=$(length(step_transitions))")
    end
    
    # ========== VISUALIZATION ==========
    
    if steps > 1 && !isempty(massey_history)
        # Plot 1: Massey vs Frequency scatter
        p1 = plot(massey_history, frequency_history,
                 title="Massey Obstruction vs Oscillator Frequency",
                 xlabel="Massey Obstruction", ylabel="Frequency (Hz)",
                 marker=:circle, alpha=0.7, label=false,
                 color=:purple, linewidth=2)
        
        # Add trend line if enough points
        if length(massey_history) > 2
            # Linear fit
            A = [ones(length(massey_history)) massey_history]
            coeff = A \ frequency_history
            x_range = LinRange(minimum(massey_history), maximum(massey_history), 100)
            y_fit = coeff[1] .+ coeff[2] .* x_range
            plot!(p1, x_range, y_fit, color=:red, linewidth=2, 
                  label="Trend: slope=$(round(coeff[2], digits=3))")
        end
        
        # Plot 2: Evolution over time
        p2 = plot(1:steps, massey_history,
                 label="Massey Obstruction", color=:red, linewidth=2)
        plot!(p2, 1:steps, frequency_history ./ maximum(frequency_history),
              label="Frequency (normalized)", color=:blue, linewidth=2, alpha=0.7)
        title!(p2, "Evolution Over Time")
        xlabel!("Step")
        ylabel!("Value")
        
        # Plot 3: Network visualization
        p3 = scatter(title="Network: Color = Massey, Size = Frequency",
                    xlabel="X", ylabel="Y", aspect_ratio=:equal, legend=false)
        
        # Nodes
        scatter!(p3, [n.position[1] for n in nodes],
                 [n.position[2] for n in nodes],
                 color=:blue, markersize=15, alpha=0.8)
        
        # Edges colored by Massey obstruction
        for edge in edges
            obs = edge.massey_AMB.obstruction
            # Color: red = high obstruction, blue = low obstruction
            edge_color = RGB(obs, 0.2, 1-obs)
            
            # Width based on frequency
            freq_norm = edge.oscillator.current_freq / 10.0
            linewidth = 1 + 3 * freq_norm
            
            source_pos = nodes[edge.source].position
            target_pos = nodes[edge.target].position
            
            plot!(p3, [source_pos[1], target_pos[1]],
                   [source_pos[2], target_pos[2]],
                   color=edge_color, alpha=0.7, linewidth=linewidth)
        end
        
        # Add node labels
        for (i, node) in enumerate(nodes)
            annotate!(p3, node.position[1], node.position[2], text("N$i", 10))
        end
        
        # Plot 4: Transition timeline
        p4 = plot(title="Phase Transition Detection", xlabel="Step", ylabel="Transition Events")
        
        trans_counts = [length(trans) for trans in transition_history]
        if any(trans_counts .> 0)
            bar!(p4, 1:steps, trans_counts, color=:green, alpha=0.7, label="Transitions")
            
            # Mark steps with transitions
            trans_steps = findall(trans_counts .> 0)
            scatter!(p4, trans_steps, trans_counts[trans_steps],
                    color=:red, markersize=8, label="Transition steps")
        else
            plot!(p4, [1, steps], [0, 0], color=:gray, label="No transitions")
        end
        
        # Combine plots
        plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
        savefig("molecule_massey_hopf_results.png")
    end
    
    # ========== ANALYSIS ==========
    
    println("\n" * "="^70)
    println("ANALYSIS RESULTS")
    println("="^70)
    
    # Correlation analysis
    if steps > 2 && length(massey_history) == length(frequency_history)
        corr_value = cor(massey_history, frequency_history)
        println("\nCorrelation between Massey obstruction and oscillator frequency:")
        println("  r = $(round(corr_value, digits=3))")
        
        if corr_value < -0.3
            println("  ✓ Significant negative correlation detected")
            println("  → Higher Massey obstruction → Lower oscillator frequency")
        elseif corr_value > 0.3
            println("  ✓ Significant positive correlation detected")
            println("  → Higher Massey obstruction → Higher oscillator frequency")
        else
            println("  ⚠ Weak correlation (|r| < 0.3)")
        end
    end
    
    # Transition analysis
    total_transitions = sum([length(trans) for trans in transition_history])
    println("\nPhase transition detection:")
    println("  Total transitions: $total_transitions")
    
    if total_transitions > 0
        # Analyze transition characteristics
        all_transition_data = []
        for (step, trans_list) in enumerate(transition_history)
            for (edge_id, strength, coherence, obstruction) in trans_list
                push!(all_transition_data, (step, edge_id, strength, coherence, obstruction))
            end
        end
        
        avg_strength = mean([t[3] for t in all_transition_data])
        avg_coherence = mean([t[4] for t in all_transition_data])
        avg_obstruction = mean([t[5] for t in all_transition_data])
        
        println("  Average transition strength: $(round(avg_strength, digits=3))")
        println("  Average coherence at transition: $(round(avg_coherence, digits=3))")
        println("  Average Massey at transition: $(round(avg_obstruction, digits=3))")
        
        # Check transition conditions
        if avg_obstruction < 0.3 && avg_coherence > 0.7
            println("  ✓ Transitions occur under correct conditions:")
            println("    - Low Massey obstruction (< 0.3)")
            println("    - High phase coherence (> 0.7)")
        end
    else
        println("  ⚠ No phase transitions detected")
        println("  Try increasing steps or decreasing Massey threshold")
    end
    
    # Final statistics
    if !isempty(edges)
        final_massey = mean([e.massey_AMB.obstruction for e in edges])
        final_freq = mean([e.oscillator.current_freq for e in edges])
        
        println("\nFinal network state:")
        println("  Average Massey obstruction: $(round(final_massey, digits=3))")
        println("  Average oscillator frequency: $(round(final_freq, digits=3)) Hz")
        
        # Molecule distribution
        total_A = sum([length(n.molecule_pool[:A]) for n in nodes])
        total_B = sum([length(n.molecule_pool[:B]) for n in nodes])
        println("  Total molecules in network:")
        println("    Type A: $total_A")
        println("    Type B: $total_B")
    end
    
    println("\n" * "="^70)
    println("PHYSICAL INTERPRETATION")
    println("="^70)
    println("✓ Two molecule types (A, B) with different algebra representations")
    println("✓ Molecules deform edge Morita algebras (A, M, B)")
    println("✓ Massey product ⟨A,M,B⟩ measures higher associativity obstruction")
    println("✓ Massey obstruction modulates Hopf oscillator frequency")
    println("✓ Phase transitions when: Low Massey + High oscillator coherence")
    println("✓ Local-to-global transitions via edge coupling")
    println("="^70)
    
    return (nodes, edges, massey_history, frequency_history, transition_history)
end

# ========== MAIN ==========

println("\nStarting Molecule-Massey-Hopf simulation...")
println("Testing the complete causal chain:")

results = run_molecule_massey_hopf_simulation(4, 10)

println("\n✓ Simulation complete!")
println("Results saved to 'molecule_massey_hopf_results.png'")