# NeuroSheaf_Simulation_N1000.jl
# Complete simulation of neuro-sheaf architecture for brain phase transitions

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Distributions
using DataFrames

# Set random seed for reproducibility
Random.seed!(42)

# ============================================================================
# 1. DATA GENERATION FOR N=1000 GRAPH
# ============================================================================

function generate_brain_graph(N::Int=1000)
    println("Generating brain graph with $N nodes...")
    
    # Generate random 3D positions (simulating brain regions)
    positions = zeros(N, 3)  # Pre-allocate N×3 matrix
    for i in 1:N
        positions[i, :] = rand(3) .* [3000.0, 4000.0, 100.0]  # x, y, z in micrometers
    end
    
    # Alternative vectorized approach:
    # positions = rand(N, 3) .* [3000.0 4000.0 100.0]  # Note: row vector
    
    # Assign regions based on position (simplified)
    regions = Vector{Vector{Symbol}}(undef, N)
    for i in 1:N
        x, y, z = positions[i, 1], positions[i, 2], positions[i, 3]
        if z > 50 && x < 1500  # Deep structure
            regions[i] = [:BG, :TH]  # Basal ganglia + Thalamus
        elseif y > 2000 && x > 2000  # Posterior
            regions[i] = [:CUL4, :bgr]  # Cerebellum + background
        elseif x < 1000  # Anterior
            regions[i] = [:PFC, :bgr]  # Prefrontal cortex
        else
            regions[i] = [:bgr]  # Background
        end
    end
    
    # Generate edges with realistic properties
    n_edges_target = Int(round(1.6 * N))  # ~1.6 edges per node
    
    edges = []
    for _ in 1:n_edges_target
        i = rand(1:N)
        j = rand(1:N)
        while j == i
            j = rand(1:N)
        end
        
        # Calculate distance
        dx = positions[i, 1] - positions[j, 1]
        dy = positions[i, 2] - positions[j, 2]
        dz = positions[i, 3] - positions[j, 3]
        dist = sqrt(dx^2 + dy^2 + dz^2)
        
        # Generate realistic edge properties
        length_val = dist * (0.8 + 0.4 * rand())
        curvature = 1.0 + 0.5 * rand()
        cross_section = 10.0 + 40.0 * rand()
        roundness = 0.3 + 0.5 * rand()
        
        # Edge degree (simplified)
        node1_degree = rand(1:5)
        node2_degree = rand(1:5)
        
        push!(edges, (i, j, length_val, dist, curvature, 
                     cross_section, roundness, node1_degree, node2_degree))
    end
    
    # Create adjacency matrix
    I = Int[]
    J = Int[]
    V = Float64[]
    
    for (i, j, len, dist, curv, cs, rnd, d1, d2) in edges
        push!(I, i)
        push!(J, j)
        # Weight based on edge properties
        weight = (1.0 / len) * (cs / 50.0) * exp(-0.1 * curv)
        push!(V, weight)
    end
    
    adjacency = sparse(I, J, V, N, N)
    
    println("Generated: $N nodes, $(length(edges)) edges")
    
    return positions, regions, edges, adjacency
end

# ============================================================================
# 2. CORE DATA STRUCTURES
# ============================================================================

# Edge interaction types
@enum EdgeInteractionType AA=1 AB=2 BB=3 AN=4 BN=5

struct MoleculeState
    fentanyl::Float64
    noloxone::Float64
    dopamine::Float64
    
    MoleculeState() = new(0.0, 0.0, 0.5)  # Baseline dopamine
end

struct HopfOscillator
    frequency::Float64
    amplitude::Float64
    phase::Float64
    
    function HopfOscillator(base_freq::Float64=8.0)
        new(base_freq, 1.0, 2π * rand())
    end
end

struct EdgeInteraction
    type::EdgeInteractionType
    oscillator::HopfOscillator
    ladder_effect::Float64
    tower_effect::Float64
    
    function EdgeInteraction(mol_i::MoleculeState, mol_j::MoleculeState)
        # Determine interaction type based on molecule predominance
        f_ratio_i = mol_i.fentanyl / (mol_i.fentanyl + mol_i.noloxone + 1e-10)
        f_ratio_j = mol_j.fentanyl / (mol_j.fentanyl + mol_j.noloxone + 1e-10)
        n_ratio_i = mol_i.noloxone / (mol_i.fentanyl + mol_i.noloxone + 1e-10)
        n_ratio_j = mol_j.noloxone / (mol_j.fentanyl + mol_j.noloxone + 1e-10)
        
        # Classification rules
        if f_ratio_i > 0.7 && f_ratio_j > 0.7
            type = AA
            base_freq = 4.0  # Slower with fentanyl
        elseif n_ratio_i > 0.7 && n_ratio_j > 0.7
            type = BB
            base_freq = 12.0  # Faster with noloxone
        elseif (f_ratio_i > 0.5 && n_ratio_j > 0.5) || (n_ratio_i > 0.5 && f_ratio_j > 0.5)
            type = AB
            base_freq = 8.0 + 3.0 * randn()  # Unstable interface
        elseif f_ratio_i > 0.7 && (f_ratio_j + n_ratio_j) < 0.3
            type = AN
            base_freq = 5.0
        elseif n_ratio_i > 0.7 && (f_ratio_j + n_ratio_j) < 0.3
            type = BN
            base_freq = 10.0
        else
            type = AA  # Default
            base_freq = 8.0
        end
        
        new(type, HopfOscillator(base_freq), 0.0, 0.0)
    end
end

struct ProlateOperator
    band::Symbol
    min_freq::Float64
    max_freq::Float64
    jacobi_matrix::Matrix{Float64}
    HT_transform::Matrix{Float64}  # Hardy-Titchmarsh
    
    function ProlateOperator(band::Symbol, n::Int=50)
        # Frequency bands
        bands = Dict(
            :delta => (0.5, 4.0),
            :theta => (4.0, 8.0),
            :alpha => (8.0, 13.0),
            :beta => (13.0, 30.0),
            :gamma => (30.0, 100.0)
        )
        
        f_min, f_max = bands[band]
        
        # Create tridiagonal Jacobi matrix
        J = zeros(n, n)
        diag_vals = range(f_min, f_max, length=n)
        for i in 1:n
            J[i, i] = diag_vals[i]
            if i < n
                J[i, i+1] = 0.2
                J[i+1, i] = 0.2
            end
        end
        
        # Hardy-Titchmarsh transform for stability
        HT = exp.(-0.05 * abs.(J))
        
        new(band, f_min, f_max, J, HT)
    end
end

struct LadderOperator
    drug::Symbol
    half_life::Float64
    propagation_speed::Float64
    region_permeability::Dict{Symbol, Float64}
    
    function LadderOperator(drug::Symbol)
        if drug == :fentanyl
            hl = 3600.0
            speed = 0.3
        else  # noloxone
            hl = 1800.0
            speed = 0.5  # Faster antidote
        end
        
        # Region permeabilities
        permeability = Dict(
            :CUL4 => 0.9,
            :TH => 0.7,
            :PFC => 0.6,
            :BG => 0.8,
            :bgr => 0.5
        )
        
        new(drug, hl, speed, permeability)
    end
end

# ============================================================================
# 3. NEURO-SHEAF SYSTEM
# ============================================================================

mutable struct NeuroSheafSystem
    # Basic properties
    N::Int
    positions::Matrix{Float64}
    regions::Vector{Vector{Symbol}}
    edges::Vector{Tuple}  # (i, j, length, dist, curvature, cross_section, ...)
    adjacency::SparseMatrixCSC{Float64, Int}
    
    # Dynamic state
    molecules::Vector{MoleculeState}
    edge_interactions::Dict{Tuple{Int, Int}, EdgeInteraction}
    wave_states::Dict{Symbol, Vector{Float64}}
    
    # Operators
    ladders::Dict{Symbol, LadderOperator}
    prolate_ops::Dict{Symbol, ProlateOperator}
    
    # Region constraints
    region_freq_bounds::Dict{Symbol, Tuple{Symbol, Float64, Float64}}
    
    # Simulation state
    time::Float64
    phase_history::Vector{Dict{Symbol, Float64}}
    
    function NeuroSheafSystem(N::Int=1000)
        # Generate graph
        positions, regions, edges, adjacency = generate_brain_graph(N)
        
        # Initialize molecules
        molecules = [MoleculeState() for _ in 1:N]
        
        # Initialize edge interactions
        edge_interactions = Dict{Tuple{Int, Int}, EdgeInteraction}()
        for (i, j, _, _, _, _, _, _, _) in edges
            edge_interactions[(i, j)] = EdgeInteraction(molecules[i], molecules[j])
        end
        
        # Initialize wave states
        wave_states = Dict(
            :theta => zeros(N),
            :alpha => zeros(N),
            :beta => zeros(N),
            :gamma => zeros(N)
        )
        
        # Initialize operators
        ladders = Dict(
            :fentanyl => LadderOperator(:fentanyl),
            :noloxone => LadderOperator(:noloxone)
        )
        
        prolate_ops = Dict(
            :theta => ProlateOperator(:theta),
            :alpha => ProlateOperator(:alpha),
            :beta => ProlateOperator(:beta),
            :gamma => ProlateOperator(:gamma)
        )
        
        # Region frequency constraints
        region_freq_bounds = Dict(
            :PFC => (:theta, 4.0, 8.0),
            :CUL4 => (:alpha, 8.0, 12.0),
            :TH => (:alpha, 8.0, 13.0),
            :BG => (:beta, 13.0, 30.0),
            :bgr => (:theta, 4.0, 100.0)  # Broad for background
        )
        
        new(N, positions, regions, edges, adjacency, 
            molecules, edge_interactions, wave_states,
            ladders, prolate_ops, region_freq_bounds,
            0.0, [])
    end
end

# ============================================================================
# 4. CORE ALGORITHMS
# ============================================================================

function apply_ladder!(system::NeuroSheafSystem, drug::Symbol, dt::Float64)
    ladder = system.ladders[drug]
    λ = log(2) / ladder.half_life  # Decay constant
    
    # Temporary storage for updates
    delta_molecules = zeros(system.N)
    
    for (i, j, len, _, curv, cs, _, _, _) in system.edges
        # Get current concentrations
        if drug == :fentanyl
            c_i = system.molecules[i].fentanyl
            c_j = system.molecules[j].fentanyl
        else
            c_i = system.molecules[i].noloxone
            c_j = system.molecules[j].noloxone
        end
        
        # Calculate region permeability
        regions_i = system.regions[i]
        regions_j = system.regions[j]
        perm_i = minimum([ladder.region_permeability[r] for r in regions_i])
        perm_j = minimum([ladder.region_permeability[r] for r in regions_j])
        
        # Propagation with edge geometry effects
        speed = ladder.propagation_speed * exp(-0.1 * curv) * (cs / 50.0)
        permeability = min(perm_i, perm_j)
        
        # Flux across edge
        flux = speed * permeability * dt * (c_i - c_j) / len
        
        # Apply updates
        delta_molecules[i] -= flux
        delta_molecules[j] += flux
    end
    
    # Apply updates with decay
    for i in 1:system.N
        if drug == :fentanyl
            current = system.molecules[i].fentanyl
            decayed = current * exp(-λ * dt)
            system.molecules[i].fentanyl = max(0.0, decayed + delta_molecules[i])
        else
            current = system.molecules[i].noloxone
            decayed = current * exp(-λ * dt)
            system.molecules[i].noloxone = max(0.0, decayed + delta_molecules[i])
        end
    end
    
    # Update edge interactions
    update_edge_interactions!(system)
end

function update_edge_interactions!(system::NeuroSheafSystem)
    for (key, _) in system.edge_interactions
        i, j = key
        new_interaction = EdgeInteraction(system.molecules[i], system.molecules[j])
        system.edge_interactions[key] = new_interaction
    end
end

function update_waves!(system::NeuroSheafSystem, dt::Float64)
    # Update each frequency band
    for (band, prolate_op) in system.prolate_ops
        current_state = system.wave_states[band]
        n = length(current_state)
        
        # Apply prolate operator (using reduced dimension for efficiency)
        if n <= 50
            transformed = prolate_op.jacobi_matrix * current_state
        else
            # For larger N, use sparse approximation
            transformed = similar(current_state)
            for i in 1:n
                # Local averaging effect
                neighbors = findnz(system.adjacency[i, :])[1]
                if !isempty(neighbors)
                    transformed[i] = 0.7 * current_state[i] + 
                                    0.3 * mean(current_state[neighbors])
                else
                    transformed[i] = current_state[i]
                end
            end
        end
        
        # Add contributions from edge interactions
        for ((i, j), edge_int) in system.edge_interactions
            # Hopf oscillator contribution
            phase = edge_int.oscillator.phase
            freq = edge_int.oscillator.frequency
            amp = edge_int.oscillator.amplitude
            
            # Update oscillator phase
            edge_int.oscillator.phase = (phase + 2π * freq * dt) % (2π)
            
            # Contribution based on interaction type
            type_factor = Dict(
                AA => 0.5,  # Fentanyl-fentanyl: suppressed
                AB => 1.5,  # Interface: amplified
                BB => 1.0,  # Noloxone-noloxone: normal
                AN => 0.8,  # Fentanyl front
                BN => 0.9   # Noloxone front
            )[edge_int.type]
            
            contribution = amp * sin(phase) * type_factor * dt
            
            # Distribute to nodes
            transformed[i] += contribution
            transformed[j] += contribution
            
            # Record ladder effect
            edge_int.ladder_effect = 0.1 * contribution
        end
        
        # Apply Hardy-Titchmarsh transform for stability
        if n <= 50
            transformed = prolate_op.HT_transform * transformed
        end
        
        # Enforce region frequency constraints
        for i in 1:system.N
            regions = system.regions[i]
            for region in regions
                if haskey(system.region_freq_bounds, region)
                    r_band, f_min, f_max = system.region_freq_bounds[region]
                    if band == r_band
                        # Enforce bounds
                        if transformed[i] < f_min
                            transformed[i] = f_min
                        elseif transformed[i] > f_max
                            transformed[i] = f_max
                        end
                    end
                end
            end
        end
        
        # Update wave state
        system.wave_states[band] = transformed
    end
end

function apply_tower_diffusion!(system::NeuroSheafSystem, level::Int, dt::Float64)
    # Simplified tower diffusion: coarser propagation
    # Level 1: fine, Level 2: medium, Level 3: coarse
    
    scale_factor = 2.0^level
    diffusion_coeff = 0.01 / scale_factor
    
    for drug in [:fentanyl, :noloxone]
        # Create temporary array
        temp_conc = zeros(system.N)
        
        if drug == :fentanyl
            for i in 1:system.N
                temp_conc[i] = system.molecules[i].fentanyl
            end
        else
            for i in 1:system.N
                temp_conc[i] = system.molecules[i].noloxone
            end
        end
        
        # Apply diffusion
        new_conc = copy(temp_conc)
        for i in 1:system.N
            neighbors = findnz(system.adjacency[i, :])[1]
            if !isempty(neighbors)
                avg_neighbor = mean(temp_conc[neighbors])
                diff = diffusion_coeff * dt * (avg_neighbor - temp_conc[i])
                new_conc[i] += diff
            end
        end
        
        # Update molecules
        for i in 1:system.N
            if drug == :fentanyl
                system.molecules[i].fentanyl = max(0.0, new_conc[i])
            else
                system.molecules[i].noloxone = max(0.0, new_conc[i])
            end
        end
    end
    
    update_edge_interactions!(system)
end

function administer_drug!(system::NeuroSheafSystem, drug::Symbol, 
                         region::Symbol, dose::Float64)
    count = 0
    for i in 1:system.N
        if region in system.regions[i]
            if drug == :fentanyl
                system.molecules[i].fentanyl += dose
                count += 1
            else
                system.molecules[i].noloxone += dose
                count += 1
            end
        end
    end
    println("Administered $drug to $count nodes in $region region")
    update_edge_interactions!(system)
end

function compute_phase_metrics(system::NeuroSheafSystem)
    metrics = Dict{Symbol, Float64}()
    
    # 1. Drug concentration ratios
    total_fent = sum([m.fentanyl for m in system.molecules])
    total_nolo = sum([m.noloxone for m in system.molecules])
    metrics[:fent_nolo_ratio] = total_fent / (total_nolo + 1e-10)
    
    # 2. Wave coherence
    for band in [:theta, :alpha, :beta]
        wave = system.wave_states[band]
        metrics[Symbol(band, :_power)] = mean(abs.(wave))
        metrics[Symbol(band, :_coherence)] = std(wave) / (mean(abs.(wave)) + 1e-10)
    end
    
    # 3. Region-specific metrics
    for region in [:PFC, :CUL4, :TH, :BG]
        nodes = findall(i -> region in system.regions[i], 1:system.N)
        if !isempty(nodes)
            theta_power = mean([system.wave_states[:theta][i] for i in nodes])
            beta_power = mean([system.wave_states[:beta][i] for i in nodes])
            metrics[Symbol(region, :_theta_beta_ratio)] = theta_power / (beta_power + 1e-10)
        end
    end
    
    # 4. Edge interaction distribution
    type_counts = Dict(AA=>0, AB=>0, BB=>0, AN=>0, BN=>0)
    for edge_int in values(system.edge_interactions)
        type_counts[edge_int.type] += 1
    end
    total_edges = length(system.edge_interactions)
    for (type, count) in type_counts
        type_str = string(type)
        metrics[Symbol("edge_", type_str)] = count / total_edges
    end
    
    return metrics
end

function detect_phase_transition(system::NeuroSheafSystem)
    metrics = compute_phase_metrics(system)
    
    # Phase 1: Opiate dominant (fentanyl > noloxone, low theta)
    if metrics[:fent_nolo_ratio] > 2.0 && metrics[:PFC_theta_beta_ratio] < 0.8
        return 1, "Opiate Phase"
    
    # Phase 2: Critical competition (drugs balanced, unstable waves)
    elseif 0.5 < metrics[:fent_nolo_ratio] < 2.0 && metrics[:theta_coherence] > 1.0
        return 2, "Critical Phase"
    
    # Phase 3: Transition (noloxone winning, theta emerging)
    elseif metrics[:fent_nolo_ratio] < 0.5 && metrics[:PFC_theta_beta_ratio] > 1.2
        return 3, "Transition Phase"
    
    # Phase 4: Norcain recovery (noloxone dominant, stable theta)
    elseif metrics[:fent_nolo_ratio] < 0.2 && metrics[:PFC_theta_beta_ratio] > 1.5
        return 4, "Norcain Recovery"
    
    else
        return 0, "Mixed/Transitional"
    end
end

# ============================================================================
# 5. VISUALIZATION AND MONITORING
# ============================================================================

function print_status(system::NeuroSheafSystem, step::Int)
    metrics = compute_phase_metrics(system)
    phase_num, phase_name = detect_phase_transition(system)
    
    println("\n" * "="^60)
    println("Time: $(system.time)s | Step: $step | Phase: $phase_name")
    println("="^60)
    
    println("Drug Metrics:")
    println("  Fentanyl/Noloxone ratio: $(round(metrics[:fent_nolo_ratio], digits=3))")
    println("  Total Fentanyl: $(round(sum([m.fentanyl for m in system.molecules]), digits=2))")
    println("  Total Noloxone: $(round(sum([m.noloxone for m in system.molecules]), digits=2))")
    
    println("\nWave Metrics:")
    println("  Theta power: $(round(metrics[:theta_power], digits=3))")
    println("  Alpha power: $(round(metrics[:alpha_power], digits=3))")
    println("  Beta power: $(round(metrics[:beta_power], digits=3))")
    println("  PFC Theta/Beta: $(round(metrics[:PFC_theta_beta_ratio], digits=3))")
    
    println("\nEdge Interactions:")
    println("  AA (fent-fent): $(round(100*metrics[:edge_AA], digits=1))%")
    println("  AB (fent-nolo): $(round(100*metrics[:edge_AB], digits=1))%")
    println("  BB (nolo-nolo): $(round(100*metrics[:edge_BB], digits=1))%")
    println("  AN (fent-none): $(round(100*metrics[:edge_AN], digits=1))%")
    println("  BN (nolo-none): $(round(100*metrics[:edge_BN], digits=1))%")
end

function record_history!(system::NeuroSheafSystem)
    metrics = compute_phase_metrics(system)
    push!(system.phase_history, metrics)
end

# ============================================================================
# 6. MAIN SIMULATION LOOP
# ============================================================================

function run_simulation(N::Int=1000, total_time::Float64=1200.0, dt::Float64=1.0)
    println("="^60)
    println("NEURO-SHEAF SIMULATION")
    println("N = $N nodes")
    println("Time = $total_time seconds")
    println("="^60)
    
    # Initialize system
    system = NeuroSheafSystem(N)
    
    # Initial drug administration
    println("\nInitial conditions:")
    administer_drug!(system, :fentanyl, :PFC, 2.0)  # High dose in PFC
    administer_drug!(system, :fentanyl, :BG, 1.0)   # Moderate dose in BG
    
    # Simulation parameters
    n_steps = Int(total_time / dt)
    print_interval = Int(60 / dt)  # Print every minute
    
    # Main simulation loop
    for step in 1:n_steps
        system.time = step * dt
        
        # Apply ladder operators
        apply_ladder!(system, :fentanyl, dt)
        apply_ladder!(system, :noloxone, dt)
        
        # Administer noloxone antidote at 5 minutes
        if system.time == 300.0
            println("\n=== ADMINISTERING NOLOXONE ANTIDOTE ===")
            administer_drug!(system, :noloxone, :PFC, 4.0)  # High dose antidote
            administer_drug!(system, :noloxone, :BG, 2.0)
        end
        
        # Apply tower diffusion every 30 seconds
        if step % Int(30/dt) == 0
            apply_tower_diffusion!(system, 2, dt)  # Medium-level diffusion
        end
        
        # Update wave dynamics
        update_waves!(system, dt)
        
        # Record history
        if step % Int(5/dt) == 0  # Every 5 seconds
            record_history!(system)
        end
        
        # Print status
        if step % print_interval == 0
            print_status(system, step)
        end
        
        # Emergency stop if phase reversal detected
        metrics = compute_phase_metrics(system)
        if metrics[:fent_nolo_ratio] > 3.0 && system.time > 600.0
            println("\n!!! PHASE REVERSAL DETECTED !!!")
            println("Fentanyl re-emerging after noloxone administration")
            break
        end
    end
    
    # Final analysis
    println("\n" * "="^60)
    println("SIMULATION COMPLETE")
    println("="^60)
    
    analyze_results(system)
    
    return system
end

function analyze_results(system::NeuroSheafSystem)
    println("\nFINAL ANALYSIS:")
    println("-"^40)
    
    # Get final phase
    phase_num, phase_name = detect_phase_transition(system)
    println("Final Phase: $phase_name")
    
    # Calculate recovery metrics
    pfc_nodes = findall(i -> :PFC in system.regions[i], 1:system.N)
    if !isempty(pfc_nodes)
        theta_power = mean([system.wave_states[:theta][i] for i in pfc_nodes])
        beta_power = mean([system.wave_states[:beta][i] for i in pfc_nodes])
        ratio = theta_power / (beta_power + 1e-10)
        
        println("\nPrefrontal Cortex (Executive Function):")
        println("  Theta/Beta ratio: $(round(ratio, digits=3))")
        if ratio > 1.5
            println("  ✓ Executive function recovered")
        elseif ratio > 1.0
            println("  ⚠ Executive function partially recovered")
        else
            println("  ✗ Executive function still impaired")
        end
    end
    
    # Drug clearance
    total_fent = sum([m.fentanyl for m in system.molecules])
    total_nolo = sum([m.noloxone for m in system.molecules])
    
    println("\nDrug Clearance:")
    println("  Remaining fentanyl: $(round(total_fent, digits=3))")
    println("  Remaining noloxone: $(round(total_nolo, digits=3))")
    
    # Edge interaction final distribution
    type_counts = Dict(AA=>0, AB=>0, BB=>0, AN=>0, BN=>0)
    for edge_int in values(system.edge_interactions)
        type_counts[edge_int.type] += 1
    end
    
    println("\nFinal Edge Interaction Distribution:")
    for type in [AA, AB, BB, AN, BN]
        count = type_counts[type]
        percentage = 100 * count / length(system.edge_interactions)
        type_str = string(type)
        println("  $type_str: $(round(percentage, digits=1))%")
    end
    
    # Phase timeline
    if !isempty(system.phase_history)
        println("\nPhase Timeline:")
        times = range(0, stop=system.time, length=length(system.phase_history))
        
        phase_changes = []
        prev_phase = 0
        for (i, metrics) in enumerate(system.phase_history)
            # Simplified phase detection from history
            fent_ratio = metrics[:fent_nolo_ratio]
            theta_beta = get(metrics, :PFC_theta_beta_ratio, 1.0)
            
            if fent_ratio > 2.0 && theta_beta < 0.8
                current_phase = 1
            elseif 0.5 < fent_ratio < 2.0
                current_phase = 2
            elseif fent_ratio < 0.5 && theta_beta > 1.2
                current_phase = 3
            elseif fent_ratio < 0.2 && theta_beta > 1.5
                current_phase = 4
            else
                current_phase = 0
            end
            
            if current_phase != prev_phase && current_phase > 0
                push!(phase_changes, (times[i], current_phase))
                prev_phase = current_phase
            end
        end
        
        for (time, phase) in phase_changes
            phase_names = ["", "Opiate", "Critical", "Transition", "Norcain"]
            println("  $(round(time, digits=0))s: $(phase_names[phase])")
        end
    end
end

# ============================================================================
# 7. RUN THE SIMULATION
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Run simulation with 1000 nodes
    system = run_simulation(1000, 1200.0, 0.5)  # 20 minutes simulation
    
    # Summary
    println("\n" * "="^60)
    println("SUMMARY")
    println("="^60)
    println("This simulation demonstrates:")
    println("1. Fentanyl propagation via ladder operators")
    println("2. Noloxone antidote administration at 5 minutes")
    println("3. Tower diffusion for coarse-scale transport")
    println("4. 5-class edge interactions (AA, AB, BB, AN, BN)")
    println("5. Wave dynamics constrained by brain regions")
    println("6. Phase transitions driven by molecular competition")
    println("7. Executive function recovery via theta/beta ratio")
    println("="^60)
end
