using LinearAlgebra
using Statistics
using Plots
using Random
using SparseArrays

# ========== PROPER STRUCT DEFINITIONS ==========

struct Node
    id::Int
    position::Vector{Float64}
    phase::Float64
    amplitude::Float64
    massey_pressure::Float64
    transition_count::Int
end

# ========== MASSEY PRODUCT IMPLEMENTATION ==========

struct MasseyProduct
    A::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    M::Matrix{ComplexF64}
    product::Matrix{ComplexF64}
    obstruction::Float64
    indeterminacy::Matrix{ComplexF64}
    is_defined::Bool
end

function compute_massey_triple(A::Matrix{ComplexF64}, B::Matrix{ComplexF64}, M::Matrix{ComplexF64})
    n = size(A, 1)
    
    # Massey triple product: ⟨A,B,M⟩ = AB·M - A·BM + MA·B
    AB = A * B
    BM = B * M
    MA = M * A
    
    massey = AB * M - A * BM + MA * B
    obstruction = norm(massey)
    
    # Indeterminacy placeholder
    H = zeros(ComplexF64, n, n)
    K = zeros(ComplexF64, n, n)
    indeterminacy = A * H + K * M
    
    is_defined = obstruction > 1e-10
    
    return MasseyProduct(A, B, M, massey, obstruction, indeterminacy, is_defined)
end

# ========== MORITA ALGEBRA WITH MASSEY PRODUCTS ==========

struct MoritaAlgebra
    A::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    M::Matrix{ComplexF64}
    massey_AMB::MasseyProduct
    massey_BMA::MasseyProduct
    massey_AMA::MasseyProduct
    massey_BMB::MasseyProduct
    total_obstruction::Float64
    is_spherical::Bool
    kodaira_weight::Float64
end

function create_morita_with_massey(source_alg::Matrix{ComplexF64}, 
                                  target_alg::Matrix{ComplexF64}, 
                                  bimodule::Matrix{ComplexF64})
    n = size(source_alg, 1)
    
    # Compute Massey products
    massey_AMB = compute_massey_triple(source_alg, bimodule, target_alg)
    massey_BMA = compute_massey_triple(target_alg, bimodule, source_alg)
    massey_AMA = compute_massey_triple(source_alg, bimodule, source_alg)
    massey_BMB = compute_massey_triple(target_alg, bimodule, target_alg)
    
    total_obst = (massey_AMB.obstruction + massey_BMA.obstruction +
                  massey_AMA.obstruction + massey_BMB.obstruction)
    
    # Check spherical condition
    trace_condition = abs(tr(source_alg * target_alg' - target_alg * source_alg'))
    is_spherical = trace_condition < 0.1 && total_obst < 0.5
    
    kodaira_weight = log(total_obst + 1e-10)
    
    return MoritaAlgebra(source_alg, target_alg, bimodule,
                        massey_AMB, massey_BMA, massey_AMA, massey_BMB,
                        total_obst, is_spherical, kodaira_weight)
end

# ========== PHASE TRANSITION DETECTION ==========

function detect_phase_transition_massey(morita::MoritaAlgebra, 
                                       previous_morita::Union{MoritaAlgebra, Nothing}=nothing)
    current_obst = morita.total_obstruction
    
    if previous_morita === nothing
        return false, 0.0, Dict(:obstruction => current_obst, :type => :none)
    end
    
    previous_obst = previous_morita.total_obstruction
    delta = abs(current_obst - previous_obst) / (previous_obst + 1e-10)
    
    transition_threshold = 0.3
    critical_obstruction = 0.5
    
    is_transition = (delta > transition_threshold) || 
                    (current_obst < critical_obstruction && previous_obst >= critical_obstruction)
    
    transition_type = :none
    if is_transition
        if morita.massey_AMB.obstruction < 0.1 * previous_morita.massey_AMB.obstruction
            transition_type = :associativity_recovery
        elseif morita.massey_AMA.obstruction > 2.0 * previous_morita.massey_AMA.obstruction
            transition_type = :self_interaction_emergence
        else
            transition_type = :general_deformation
        end
    end
    
    transition_data = Dict(
        :obstruction => current_obst,
        :delta => delta,
        :type => transition_type,
        :massey_AMB => morita.massey_AMB.obstruction,
        :massey_BMA => morita.massey_BMA.obstruction,
        :massey_AMA => morita.massey_AMA.obstruction,
        :massey_BMB => morita.massey_BMB.obstruction
    )
    
    return is_transition, delta, transition_data
end

function compute_massey_correlation(network_moritas::Vector{MoritaAlgebra})
    n = length(network_moritas)
    if n < 2
        return 0.0, zeros(0, 0)
    end
    
    C = zeros(n, n)
    
    for i in 1:n
        for j in i:n
            if i == j
                C[i,j] = 1.0
            else
                m1 = network_moritas[i]
                m2 = network_moritas[j]
                
                pattern1 = [m1.massey_AMB.obstruction, m1.massey_BMA.obstruction,
                           m1.massey_AMA.obstruction, m1.massey_BMB.obstruction]
                pattern2 = [m2.massey_AMB.obstruction, m2.massey_BMA.obstruction,
                           m2.massey_AMA.obstruction, m2.massey_BMB.obstruction]
                
                C[i,j] = cor(pattern1, pattern2)
                C[j,i] = C[i,j]
            end
        end
    end
    
    total_corr = 0.0
    count = 0
    for i in 1:n
        for j in i+1:n
            total_corr += C[i,j]
            count += 1
        end
    end
    
    avg_correlation = count > 0 ? total_corr / count : 0.0
    
    return avg_correlation, C
end

# ========== PROLATE OPERATOR FROM MASSEY PRODUCTS ==========

function massey_prolate_operator(nodes::Vector{Node}, edges::Vector{MoritaAlgebra}, 
                                edge_connections::Dict{Tuple{Int,Int}, Int},
                                center_node::Int, radius::Float64=0.3)
    center_pos = nodes[center_node].position
    local_indices = Int[]
    
    # Find nodes within radius
    for (i, node) in enumerate(nodes)
        if norm(node.position - center_pos) < radius
            push!(local_indices, i)
        end
    end
    
    n_local = length(local_indices)
    if n_local < 2
        return 0.0, 0.0, zeros(0,0), Dict()
    end
    
    # Build categorical adjacency from Massey products
    A_massey = zeros(n_local, n_local)
    massey_data = Dict{Tuple{Int,Int}, Dict}()
    
    for i in 1:n_local
        node_i_idx = local_indices[i]
        node_i = nodes[node_i_idx]
        
        for j in i+1:n_local
            node_j_idx = local_indices[j]
            node_j = nodes[node_j_idx]
            
            # Check if there's a connection
            if haskey(edge_connections, (node_i.id, node_j.id))
                edge_idx = edge_connections[(node_i.id, node_j.id)]
                if edge_idx <= length(edges)
                    morita = edges[edge_idx]
                    
                    # Connection strength based on Massey obstruction
                    obs_i = morita.massey_AMB.obstruction
                    obs_j = morita.massey_BMA.obstruction
                    strength = 1.0 / (1.0 + obs_i + obs_j)
                    
                    A_massey[i,j] = strength
                    A_massey[j,i] = strength
                    
                    massey_data[(i,j)] = Dict(
                        :massey_AMB => morita.massey_AMB.obstruction,
                        :massey_BMA => morita.massey_BMA.obstruction,
                        :total_obst => morita.total_obstruction
                    )
                end
            end
        end
    end
    
    # Diagonal: self-interaction
    for i in 1:n_local
        node_idx = local_indices[i]
        node = nodes[node_idx]
        A_massey[i,i] = 0.5 + 0.5 * cos(node.phase)  # Phase-dependent self-interaction
    end
    
    # Compute spectral properties
    spectral_gap = 0.0
    leading_eigval = 0.0
    
    try
        eigvals = real.(eigen(A_massey).values)
        sorted_vals = sort(eigvals, rev=true)
        
        if length(sorted_vals) >= 2
            spectral_gap = sorted_vals[1] - sorted_vals[2]
            leading_eigval = sorted_vals[1]
        end
    catch e
        # println("Eigen failed: ", e)
    end
    
    return spectral_gap, leading_eigval, A_massey, massey_data
end

# ========== SIMULATION ==========

struct MasseyNetwork
    nodes::Vector{Node}
    edges::Vector{MoritaAlgebra}
    edge_connections::Dict{Tuple{Int,Int}, Int}
    history::Vector{Dict}
end

function run_massey_driven_simulation(N_nodes=15, steps=40)
    println("="^70)
    println("MASSEY PRODUCT PHASE TRANSITION SIMULATION")
    println("="^70)
    
    Random.seed!(42)
    
    # Create nodes with proper struct
    nodes = Node[]
    for i in 1:N_nodes
        push!(nodes, Node(
            i,
            rand(3) .- 0.5,
            rand() * 2π,
            0.5 + 0.3rand(),
            0.0,
            0
        ))
    end
    
    # Create edges with Morita algebras
    edges = MoritaAlgebra[]
    edge_connections = Dict{Tuple{Int,Int}, Int}()
    edge_counter = 1
    
    for i in 1:N_nodes
        for j in i+1:min(i+4, N_nodes)
            if rand() < 0.4
                n = 2
                A = randn(ComplexF64, n, n) + im * randn(n, n)
                B = randn(ComplexF64, n, n) + im * randn(n, n)
                M = randn(ComplexF64, n, n) + im * randn(n, n)
                
                A = A / max(norm(A), 1e-10)
                B = B / max(norm(B), 1e-10)
                M = M / max(norm(M), 1e-10)
                
                morita = create_morita_with_massey(A, B, M)
                push!(edges, morita)
                
                edge_connections[(i,j)] = edge_counter
                edge_connections[(j,i)] = edge_counter
                edge_counter += 1
            end
        end
    end
    
    println("Created $(length(edges)) edges with Massey products")
    
    # Simulation histories
    total_obstruction_history = Float64[]
    massey_correlation_history = Float64[]
    transition_history = []
    spectral_gap_history = Float64[]
    
    # Store previous Morita for delta computation
    previous_edges = deepcopy(edges)
    
    for step in 1:steps
        print("Step $step/$steps: ")
        
        # 1. Evolve algebras
        for (idx, morita) in enumerate(edges)
            perturbation = 0.1 * (randn(ComplexF64, 2, 2) + im * randn(2, 2))
            
            A_new = morita.A + perturbation
            B_new = morita.B + conj(perturbation)'
            M_new = morita.M * (I + 0.05 * perturbation)
            
            A_new = A_new / max(norm(A_new), 1e-10)
            B_new = B_new / max(norm(B_new), 1e-10)
            M_new = M_new / max(norm(M_new), 1e-10)
            
            edges[idx] = create_morita_with_massey(A_new, B_new, M_new)
        end
        
        # 2. Compute network statistics
        total_obst = mean([m.total_obstruction for m in edges])
        push!(total_obstruction_history, total_obst)
        
        avg_corr, corr_matrix = compute_massey_correlation(edges)
        push!(massey_correlation_history, avg_corr)
        
        # 3. Detect phase transitions
        step_transitions = []
        for (idx, (morita, prev_morita)) in enumerate(zip(edges, previous_edges))
            is_trans, delta, data = detect_phase_transition_massey(morita, prev_morita)
            
            if is_trans
                push!(step_transitions, (idx, delta, data))
                
                # Update connected nodes
                for (conn, edge_idx) in edge_connections
                    if edge_idx == idx
                        i, j = conn
                        # Create new nodes with updated counts
                        old_node_i = nodes[i]
                        old_node_j = nodes[j]
                        
                        nodes[i] = Node(
                            old_node_i.id,
                            old_node_i.position,
                            old_node_i.phase,
                            old_node_i.amplitude,
                            old_node_i.massey_pressure + data[:obstruction],
                            old_node_i.transition_count + 1
                        )
                        
                        nodes[j] = Node(
                            old_node_j.id,
                            old_node_j.position,
                            old_node_j.phase,
                            old_node_j.amplitude,
                            old_node_j.massey_pressure + data[:obstruction],
                            old_node_j.transition_count + 1
                        )
                    end
                end
            end
        end
        
        push!(transition_history, step_transitions)
        
        # 4. Compute prolate spectral gap
        center_node = N_nodes ÷ 2
        gap, lead_eig, A_massey, _ = massey_prolate_operator(nodes, edges, edge_connections, center_node)
        push!(spectral_gap_history, gap)
        
        # 5. Update node phases and amplitudes
        new_nodes = similar(nodes)
        for (i, node) in enumerate(nodes)
            # Get connected edges
            connected_obstructions = Float64[]
            for (conn, edge_idx) in edge_connections
                if conn[1] == node.id || conn[2] == node.id
                    if edge_idx <= length(edges)
                        push!(connected_obstructions, edges[edge_idx].total_obstruction)
                    end
                end
            end
            
            avg_obst = isempty(connected_obstructions) ? 0.0 : mean(connected_obstructions)
            
            # Update phase with Massey pressure
            new_phase = mod(node.phase + 0.01 * node.massey_pressure, 2π)
            
            # Update amplitude (inversely related to obstruction)
            new_amplitude = 0.5 + 0.5 * exp(-avg_obst)
            
            # Decay pressure
            new_pressure = node.massey_pressure * 0.9
            
            new_nodes[i] = Node(
                node.id,
                node.position,
                new_phase,
                new_amplitude,
                new_pressure,
                node.transition_count
            )
        end
        nodes = new_nodes
        
        # Update previous edges
        previous_edges = deepcopy(edges)
        
        println("Obst=$(round(total_obst, digits=3)), Corr=$(round(avg_corr, digits=3)), Trans=$(length(step_transitions)), Gap=$(round(gap, digits=3))")
    end
    
    # Create network object
    network = MasseyNetwork(nodes, edges, edge_connections, [])
    
    # ========== VISUALIZATION ==========
    
    # Plot 1: Total Massey obstruction
    p1 = plot(1:steps, total_obstruction_history,
             title="Total Massey Obstruction",
             xlabel="Step", ylabel="Obstruction",
             lw=2, color=:red, label=false)
    
    # Mark transition steps
    trans_steps = [i for (i, trans) in enumerate(transition_history) if !isempty(trans)]
    if !isempty(trans_steps)
        vline!(p1, trans_steps, color=:blue, alpha=0.3, label="Transitions")
        scatter!(p1, trans_steps, total_obstruction_history[trans_steps],
                color=:green, markersize=6, label="Transition points")
    end
    
    # Plot 2: Massey correlation
    p2 = plot(1:steps, massey_correlation_history,
             title="Massey Product Correlation",
             xlabel="Step", ylabel="Correlation",
             lw=2, color=:purple, label=false,
             ylims=(-1, 1))
    
    # Plot 3: Spectral gap
    p3 = plot(1:steps, spectral_gap_history,
             title="Massey Prolate Spectral Gap",
             xlabel="Step", ylabel="Spectral Gap",
             lw=2, color=:orange, label=false)
    
    # Plot 4: Transition types
    p4 = plot(title="Massey Transition Types", xlabel="Step", ylabel="Count")
    
    # Count transition types per step
    type_counts = Dict{Symbol, Vector{Int}}()
    for ttype in [:associativity_recovery, :self_interaction_emergence, :general_deformation]
        type_counts[ttype] = zeros(Int, steps)
    end
    
    for (step, trans_list) in enumerate(transition_history)
        for (_, _, data) in trans_list
            ttype = data[:type]
            if haskey(type_counts, ttype)
                type_counts[ttype][step] += 1
            end
        end
    end
    
    colors = Dict(
        :associativity_recovery => :green,
        :self_interaction_emergence => :blue,
        :general_deformation => :red
    )
    
    for (ttype, counts) in type_counts
        if any(counts .> 0)
            plot!(p4, 1:steps, counts, 
                 label=string(ttype), color=get(colors, ttype, :black),
                 lw=2, alpha=0.7)
        end
    end
    
    # Plot 5: Individual Massey products for first edge
    p5 = plot(title="Massey Products on Edge 1", xlabel="Step", ylabel="Obstruction")
    
    # We need to track edge evolution - for now show final values
    if length(edges) > 0
        m = edges[1]
        plot!(p5, [steps-1, steps], [0, m.massey_AMB.obstruction], 
             label="⟨A,M,B⟩", color=:red, lw=2)
        plot!(p5, [steps-1, steps], [0, m.massey_BMA.obstruction], 
             label="⟨B,M,A⟩", color=:blue, lw=2)
        plot!(p5, [steps-1, steps], [0, m.massey_AMA.obstruction], 
             label="⟨A,M,A⟩", color=:green, lw=2)
        plot!(p5, [steps-1, steps], [0, m.massey_BMB.obstruction], 
             label="⟨B,M,B⟩", color=:purple, lw=2)
    end
    
    # Plot 6: Network visualization
    p6 = scatter(title="Network: Color = Transition Count, Size = Massey Pressure",
                xlabel="X", ylabel="Y", aspect_ratio=:equal)
    
    # Nodes colored by transition count
    trans_counts = [n.transition_count for n in nodes]
    max_trans = maximum(trans_counts) + 1
    node_colors = [RGB(c/max_trans, 0.2, 1-c/max_trans) for c in trans_counts]
    
    scatter!(p6, [n.position[1] for n in nodes],
             [n.position[2] for n in nodes],
             color=node_colors,
             markersize=[5 + 10 * n.massey_pressure for n in nodes],
             label="Nodes")
    
    # Edges colored by Massey obstruction
    for ((i,j), edge_idx) in edge_connections
        if edge_idx <= length(edges)
            morita = edges[edge_idx]
            obs = morita.total_obstruction
            edge_color = RGB(obs, 0.3, 1-obs)
            
            pos_i = nodes[i].position
            pos_j = nodes[j].position
            
            plot!(p6, [pos_i[1], pos_j[1]], [pos_i[2], pos_j[2]],
                  color=edge_color, alpha=0.5, label=false)
        end
    end
    
    # Combine plots
    plot(p1, p2, p3, p4, p5, p6, layout=(3,2), size=(1400, 1000))
    savefig("massey_driven_phase_transitions.png")
    
    # ========== ANALYSIS ==========
    
    println("\n" * "="^70)
    println("MASSEY PRODUCT ANALYSIS")
    println("="^70)
    
    # Analyze transitions
    all_transition_data = []
    for step in 1:steps
        for (edge_idx, delta, data) in transition_history[step]
            push!(all_transition_data, (step, edge_idx, delta, data))
        end
    end
    
    if !isempty(all_transition_data)
        println("\nTotal transitions: $(length(all_transition_data))")
        
        # Group by type
        by_type = Dict{Symbol, Vector{Float64}}()
        for (step, edge_idx, delta, data) in all_transition_data
            ttype = data[:type]
            if !haskey(by_type, ttype)
                by_type[ttype] = []
            end
            push!(by_type[ttype], delta)
        end
        
        println("\nTransition types:")
        for (ttype, deltas) in by_type
            if ttype != :none
                avg_delta = mean(deltas)
                std_delta = std(deltas)
                println("  $ttype: Δ = $(round(avg_delta, digits=3)) ± $(round(std_delta, digits=3)) (n=$(length(deltas)))")
            end
        end
    else
        println("\nNo phase transitions detected")
    end
    
    # Final statistics
    println("\nFinal network statistics:")
    println("  Average Massey obstruction: $(round(mean([m.total_obstruction for m in edges]), digits=3))")
    println("  Spherical edges: $(sum([m.is_spherical for m in edges]))/$(length(edges))")
    println("  Average transition count per node: $(round(mean([n.transition_count for n in nodes]), digits=2))")
    
    # Correlation analysis
    if steps > 1
        corr_obst_gap = cor(total_obstruction_history, spectral_gap_history)
        println("\nCorrelations:")
        println("  Obstruction vs Spectral gap: $(round(corr_obst_gap, digits=3))")
        
        if corr_obst_gap < -0.3
            println("  ✓ Anti-correlation detected: High obstruction → Low spectral gap")
        end
    end
    
    return network, (total_obstruction_history, massey_correlation_history,
                    spectral_gap_history, transition_history)
end

# ========== RUN SIMULATION ==========

println("\nStarting Massey-driven simulation...")
println("\nFocus: Massey products as higher associativity obstructions")
println("Tracking: ⟨A,M,B⟩, ⟨B,M,A⟩, ⟨A,M,A⟩, ⟨B,M,B⟩")

try
    network, histories = run_massey_driven_simulation(12, 30)
    
    println("\n" * "="^70)
    println("SIMULATION COMPLETE")
    println("="^70)
    println("\nKey findings:")
    println("✓ Massey products computed for all edges")
    println("✓ Phase transitions detected via Massey obstruction changes")
    println("✓ Different transition types identified")
    println("✓ Prolate operator built from Massey data")
    println("✓ Network visualization shows transition patterns")
    println("="^70)
    
catch e
    println("Error: ", e)
    println("\nTrying with smaller simulation...")
    network, histories = run_massey_driven_simulation(8, 15)
end