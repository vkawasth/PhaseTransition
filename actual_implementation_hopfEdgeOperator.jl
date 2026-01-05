# EdgeHopf_Implementation.jl
# Complete implementation of 5 interaction types driving Hopf oscillators at edges

using LinearAlgebra
using SparseArrays
using StatsBase
using Random

# ============================================================================
# 1. MOLECULE STATES AT NODES
# ============================================================================

struct MoleculeState
    fentanyl::Float64    # Concentration [0, ∞]
    noloxone::Float64    # Concentration [0, ∞]
    dopamine::Float64    # Baseline neuromodulator [0, 1]
    
    function MoleculeState(; fent=0.0, nolo=0.0, dopa=0.5)
        new(max(0.0, fent), max(0.0, nolo), clamp(dopa, 0.0, 1.0))
    end
end

# ============================================================================
# 2. 5 EDGE INTERACTION TYPES (AA, AB, BB, AN, BN)
# ============================================================================

@enum EdgeInteractionType begin
    AA  # Fentanyl-Fentanyl
    AB  # Fentanyl-Noloxone (COMPETITIVE INTERFACE)
    BB  # Noloxone-Noloxone
    AN  # Fentanyl-None (FRONT)
    BN  # Noloxone-None (FRONT)
end

function determine_interaction_type(node1::MoleculeState, node2::MoleculeState, 
                                   threshold::Float64=0.3)
    # Get normalized proportions at each node
    total1 = node1.fentanyl + node1.noloxone + 1e-10
    total2 = node2.fentanyl + node2.noloxone + 1e-10
    
    fent_frac1 = node1.fentanyl / total1
    nolo_frac1 = node1.noloxone / total1
    fent_frac2 = node2.fentanyl / total2
    nolo_frac2 = node2.noloxone / total2
    
    # Determine predominant molecule at each node
    pred1 = fent_frac1 > nolo_frac1 ? :fentanyl : :noloxone
    pred2 = fent_frac2 > nolo_frac2 ? :fentanyl : :noloxone
    
    # Check if concentration is significant (> threshold)
    sig1 = max(fent_frac1, nolo_frac1) > threshold
    sig2 = max(fent_frac2, nolo_frac2) > threshold
    
    if !sig1 && !sig2
        # Neither node has significant drug → no special interaction
        return AA  # Default
    elseif !sig1 || !sig2
        # One node has drug, other doesn't → front
        if sig1
            # Node1 has drug
            if pred1 == :fentanyl
                return AN  # Fentanyl front
            else
                return BN  # Noloxone front
            end
        else
            # Node2 has drug
            if pred2 == :fentanyl
                return AN
            else
                return BN
            end
        end
    else
        # Both nodes have significant drug
        if pred1 == :fentanyl && pred2 == :fentanyl
            return AA
        elseif pred1 == :noloxone && pred2 == :noloxone
            return BB
        else
            return AB  # Competitive interface
        end
    end
end

# ============================================================================
# 3. HOPF OSCILLATOR AT EDGE (FREQUENCY GENERATOR)
# ============================================================================

mutable struct HopfOscillator
    # Core oscillator parameters
    natural_frequency::Float64   # Hz (4-100 for brain waves)
    amplitude::Float64           # [0, 1]
    phase::Float64               # [0, 2π]
    
    # Dopamine modulation
    dopamine_sensitivity::Float64  # How much dopamine affects frequency
    
    # Edge geometry effects
    curvature_factor::Float64     # From edge curvature
    cross_section_factor::Float64 # From edge cross-section
    
    # Current dynamic state
    instantaneous_frequency::Float64
    power::Float64
    
    function HopfOscillator(base_freq::Float64=8.0, curvature::Float64=1.0, 
                           cross_section::Float64=30.0)
        # Dopamine sensitivity: higher for reward pathways
        dopa_sens = 0.3 + 0.4 * rand()
        
        # Geometry effects
        curve_factor = exp(-0.1 * curvature)
        cross_factor = tanh(cross_section / 50.0)
        
        new(base_freq, 1.0, 2π * rand(), dopa_sens, 
            curve_factor, cross_factor, base_freq, 1.0)
    end
end

# ============================================================================
# 4. EDGE WITH HOPF OSCILLATOR + INTERACTION TYPE
# ============================================================================

mutable struct NeuroEdge
    id::Tuple{Int, Int}          # (node1, node2)
    interaction_type::EdgeInteractionType
    oscillator::HopfOscillator
    
    # Physical properties
    length::Float64
    curvature::Float64
    cross_section::Float64
    roundness::Float64
    
    # Wave generation
    current_wave::Float64        # sin(phase) * amplitude
    frequency_power::Dict{Symbol, Float64}  # Power in each band
    coherence::Float64           # Phase coherence with neighbors
    
    # Molecular history
    type_history::Vector{EdgeInteractionType}
    frequency_history::Vector{Float64}
    
    function NeuroEdge(node1::Int, node2::Int, length::Float64, 
                      curvature::Float64=1.0, cross_section::Float64=30.0,
                      roundness::Float64=0.5)
        # Start with default interaction (AA)
        interaction = AA
        
        # Create Hopf oscillator based on edge geometry
        # Base frequency depends on curvature: straighter = higher freq
        base_freq = 8.0 + 20.0 * exp(-0.2 * curvature)
        osc = HopfOscillator(base_freq, curvature, cross_section)
        
        # Initialize wave generation
        current_wave = 0.0
        freq_power = Dict(:delta => 0.0, :theta => 0.0, :alpha => 0.0,
                         :beta => 0.0, :gamma => 0.0)
        
        new((node1, node2), interaction, osc, length, curvature, 
            cross_section, roundness, current_wave, freq_power, 0.0,
            [], [])
    end
end

# ============================================================================
# 5. INTERACTION TYPE → FREQUENCY MAPPING
# ============================================================================

function update_oscillator_from_interaction!(edge::NeuroEdge, 
                                            interaction::EdgeInteractionType,
                                            avg_dopamine::Float64=0.5)
    # Each interaction type induces specific frequency modulation
    
    base_freq = edge.oscillator.natural_frequency
    
    # Interaction-specific frequency shifts
    if interaction == AA
        # Fentanyl-Fentanyl: SLOWS oscillation (opiate effect)
        target_freq = base_freq * 0.6  # 40% slower
        amplitude_factor = 0.7         # Reduced amplitude
        coherence_boost = 0.3          # Synchronizing effect
        
    elseif interaction == AB
        # Fentanyl-Noloxone interface: UNSTABLE oscillation
        target_freq = base_freq * (1.0 + 0.5 * randn())  # Random fluctuations
        amplitude_factor = 1.5         # Amplified
        coherence_boost = -0.4         # Destabilizing
        
    elseif interaction == BB
        # Noloxone-Noloxone: SPEEDS oscillation (recovery)
        target_freq = base_freq * 1.4  # 40% faster
        amplitude_factor = 1.2
        coherence_boost = 0.6          # Highly coherent
        
    elseif interaction == AN
        # Fentanyl front: MODERATE slowing
        target_freq = base_freq * 0.8
        amplitude_factor = 0.9
        coherence_boost = 0.1
        
    else  # BN
        # Noloxone front: MODERATE speeding
        target_freq = base_freq * 1.2
        amplitude_factor = 1.1
        coherence_boost = 0.2
    end
    
    # Dopamine modulation
    dopa_mod = 1.0 + edge.oscillator.dopamine_sensitivity * (avg_dopamine - 0.5)
    target_freq *= dopa_mod
    
    # Geometry effects
    geo_mod = edge.oscillator.curvature_factor * edge.oscillator.cross_section_factor
    target_freq *= (0.8 + 0.4 * geo_mod)
    
    # Update oscillator (smooth transition)
    current_freq = edge.oscillator.instantaneous_frequency
    new_freq = 0.7 * current_freq + 0.3 * target_freq
    
    edge.oscillator.instantaneous_frequency = new_freq
    edge.oscillator.amplitude *= amplitude_factor
    edge.coherence = max(0.0, min(1.0, edge.coherence + coherence_boost))
    
    # Record history
    push!(edge.type_history, interaction)
    push!(edge.frequency_history, new_freq)
    
    return edge
end

# ============================================================================
# 6. WAVE GENERATION FROM HOPF OSCILLATORS
# ============================================================================

function generate_wave!(edge::NeuroEdge, dt::Float64)
    # Update phase: θ(t+dt) = θ(t) + 2π f dt
    freq = edge.oscillator.instantaneous_frequency
    edge.oscillator.phase = mod(edge.oscillator.phase + 2π * freq * dt, 2π)
    
    # Generate wave: A sin(θ)
    edge.current_wave = edge.oscillator.amplitude * sin(edge.oscillator.phase)
    
    # Update power in frequency bands
    update_frequency_power!(edge, freq)
    
    # Natural amplitude decay (damping)
    edge.oscillator.amplitude *= exp(-0.01 * dt)
    
    return edge.current_wave
end

function update_frequency_power!(edge::NeuroEdge, freq::Float64)
    # Assign power to appropriate frequency band
    bands = [
        (:delta, 0.5, 4.0),
        (:theta, 4.0, 8.0),
        (:alpha, 8.0, 13.0),
        (:beta, 13.0, 30.0),
        (:gamma, 30.0, 100.0)
    ]
    
    # Find which band this frequency belongs to
    for (band, f_min, f_max) in bands
        if f_min <= freq <= f_max
            # Add power to this band (weighted by amplitude)
            power = edge.oscillator.amplitude^2 * (1.0 / (1.0 + abs(freq - (f_min+f_max)/2)))
            edge.frequency_power[band] = 0.9 * edge.frequency_power[band] + 0.1 * power
        end
    end
end

# ============================================================================
# 7. NODE AGGREGATION OF EDGE WAVES
# ============================================================================

struct BrainNode
    id::Int
    position::Vector{Float64}  # (x, y, z)
    region::Symbol            # :PFC, :CUL4, :TH, :BG, :bgr
    
    # Molecular state
    molecules::MoleculeState
    
    # Connected edges
    edge_ids::Vector{Tuple{Int, Int}}
    
    # Aggregated wave state
    wave_sum::Float64
    wave_power::Dict{Symbol, Float64}
    dominant_frequency::Float64
    
    # Phase information
    phase_coherence::Float64  # With neighboring nodes
    
    function BrainNode(id::Int, position::Vector{Float64}, region::Symbol)
        # Initialize with baseline molecules
        molecules = MoleculeState(dopa=0.5)
        
        # Initialize wave states
        wave_power = Dict(:delta => 0.0, :theta => 0.0, :alpha => 0.0,
                         :beta => 0.0, :gamma => 0.0)
        
        new(id, position, region, molecules, [], 0.0, wave_power, 8.0, 0.0)
    end
end

function aggregate_waves_at_node!(node::BrainNode, edges::Dict{Tuple{Int, Int}, NeuroEdge})
    # Sum waves from all connected edges
    total_wave = 0.0
    freq_powers = Dict(:delta => 0.0, :theta => 0.0, :alpha => 0.0,
                      :beta => 0.0, :gamma => 0.0)
    
    phase_sum_real = 0.0
    phase_sum_imag = 0.0
    n_edges = 0
    
    for edge_id in node.edge_ids
        if haskey(edges, edge_id)
            edge = edges[edge_id]
            
            # Add wave contribution (weighted by edge properties)
            weight = 1.0 / (edge.length + 1e-10) * edge.cross_section / 50.0
            total_wave += edge.current_wave * weight
            
            # Aggregate frequency power
            for band in keys(freq_powers)
                freq_powers[band] += edge.frequency_power[band] * weight
            end
            
            # For phase coherence calculation
            phase_sum_real += cos(edge.oscillator.phase)
            phase_sum_imag += sin(edge.oscillator.phase)
            n_edges += 1
        end
    end
    
    # Update node state
    node.wave_sum = total_wave
    
    # Normalize frequency powers
    if n_edges > 0
        for band in keys(freq_powers)
            node.wave_power[band] = freq_powers[band] / n_edges
        end
        
        # Compute phase coherence
        mean_real = phase_sum_real / n_edges
        mean_imag = phase_sum_imag / n_edges
        node.phase_coherence = sqrt(mean_real^2 + mean_imag^2)
        
        # Find dominant frequency band
        max_power = -Inf
        dominant_band = :theta
        for (band, power) in node.wave_power
            if power > max_power
                max_power = power
                dominant_band = band
            end
        end
        # Set dominant frequency to band center
        band_centers = Dict(:delta => 2.0, :theta => 6.0, :alpha => 10.5,
                           :beta => 21.5, :gamma => 65.0)
        node.dominant_frequency = band_centers[dominant_band]
    end
    
    return node
end

# ============================================================================
# 8. COMPLETE SIMULATION WITH 5 INTERACTION TYPES
# ============================================================================

mutable struct NeuroSheafSimulation
    # Graph structure
    nodes::Dict{Int, BrainNode}
    edges::Dict{Tuple{Int, Int}, NeuroEdge}
    
    # Time state
    time::Float64
    dt::Float64
    
    # Wave observables
    regional_waves::Dict{Symbol, Vector{Float64}}  # Time series per region
    edge_type_counts::Dict{EdgeInteractionType, Int}
    frequency_spectra::Dict{Symbol, Vector{Float64}}  # Power spectrum
    
    # Pharmacological state
    fentanyl_administered::Bool
    noloxone_administered::Bool
    
    function NeuroSheafSimulation(n_nodes::Int=100)
        # Create random brain graph
        nodes = Dict{Int, BrainNode}()
        edges = Dict{Tuple{Int, Int}, NeuroEdge}()
        
        # Create nodes with random positions and regions
        regions = [:PFC, :CUL4, :TH, :BG, :bgr]
        for i in 1:n_nodes
            pos = rand(3) .* [3000.0, 4000.0, 100.0]
            region = rand(regions)
            nodes[i] = BrainNode(i, pos, region)
        end
        
        # Create edges (simplified random connectivity)
        n_edges = Int(round(1.5 * n_nodes))
        for e in 1:n_edges
            i, j = rand(1:n_nodes, 2)
            while j == i
                j = rand(1:n_nodes)
            end
            
            # Edge properties
            length = 10.0 + 40.0 * rand()
            curvature = 1.0 + 2.0 * rand()
            cross_section = 10.0 + 40.0 * rand()
            roundness = 0.3 + 0.4 * rand()
            
            edge = NeuroEdge(i, j, length, curvature, cross_section, roundness)
            edges[(i, j)] = edge
            
            # Connect to nodes
            push!(nodes[i].edge_ids, (i, j))
            push!(nodes[j].edge_ids, (i, j))
        end
        
        # Initialize wave tracking
        regional_waves = Dict(region => Float64[] for region in regions)
        edge_type_counts = Dict(type => 0 for type in instances(EdgeInteractionType))
        
        # Initialize spectra
        freq_bands = [:delta, :theta, :alpha, :beta, :gamma]
        frequency_spectra = Dict(band => Float64[] for band in freq_bands)
        
        new(nodes, edges, 0.0, 0.1, regional_waves, edge_type_counts, 
            frequency_spectra, false, false)
    end
end

function simulate_step!(sim::NeuroSheafSimulation)
    dt = sim.dt
    sim.time += dt
    
    # 1. Update molecular states (drug propagation)
    update_molecular_states!(sim)
    
    # 2. Determine edge interaction types based on node molecules
    update_edge_interactions!(sim)
    
    # 3. Generate waves from Hopf oscillators
    generate_all_waves!(sim, dt)
    
    # 4. Aggregate waves at nodes
    aggregate_all_nodes!(sim)
    
    # 5. Record observables
    record_observables!(sim)
    
    return sim
end

function update_molecular_states!(sim::NeuroSheafSimulation)
    # Simplified drug propagation: random diffusion
    
    for node in values(sim.nodes)
        # Natural dopamine fluctuation
        node.molecules.dopamine += 0.1 * randn() * sim.dt
        node.molecules.dopamine = clamp(node.molecules.dopamine, 0.0, 1.0)
        
        # Drug administration events
        if !sim.fentanyl_administered && sim.time > 60.0  # At 1 minute
            if node.region == :PFC || node.region == :BG
                node.molecules.fentanyl += 2.0
            end
            if sim.time > 61.0
                sim.fentanyl_administered = true
            end
        end
        
        if !sim.noloxone_administered && sim.time > 300.0  # At 5 minutes
            if node.region == :PFC || node.region == :BG
                node.molecules.noloxone += 4.0  # Higher dose for antidote
            end
            if sim.time > 301.0
                sim.noloxone_administered = true
            end
        end
        
        # Drug clearance (exponential decay)
        half_life_fent = 3600.0  # 1 hour
        half_life_nolo = 1800.0  # 30 minutes
        
        node.molecules.fentanyl *= exp(-log(2) * sim.dt / half_life_fent)
        node.molecules.noloxone *= exp(-log(2) * sim.dt / half_life_nolo)
    end
end

function update_edge_interactions!(sim::NeuroSheafSimulation)
    # Update each edge's interaction type based on connected node molecules
    
    # Reset counts
    for type in instances(EdgeInteractionType)
        sim.edge_type_counts[type] = 0
    end
    
    for (edge_id, edge) in sim.edges
        i, j = edge_id
        
        if haskey(sim.nodes, i) && haskey(sim.nodes, j)
            node_i = sim.nodes[i]
            node_j = sim.nodes[j]
            
            # Determine interaction type
            interaction = determine_interaction_type(node_i.molecules, node_j.molecules)
            edge.interaction_type = interaction
            
            # Update oscillator based on interaction type
            avg_dopamine = (node_i.molecules.dopamine + node_j.molecules.dopamine) / 2
            update_oscillator_from_interaction!(edge, interaction, avg_dopamine)
            
            # Count this interaction type
            sim.edge_type_counts[interaction] += 1
        end
    end
end

function generate_all_waves!(sim::NeuroSheafSimulation, dt::Float64)
    # Generate waves from all edge Hopf oscillators
    
    for edge in values(sim.edges)
        generate_wave!(edge, dt)
    end
end

function aggregate_all_nodes!(sim::NeuroSheafSimulation)
    # Aggregate waves at all nodes
    
    for node in values(sim.nodes)
        aggregate_waves_at_node!(node, sim.edges)
    end
end

function record_observables!(sim::NeuroSheafSimulation)
    # Record wave observables for analysis
    
    # Regional average waves
    regional_sums = Dict{Symbol, Float64}()
    regional_counts = Dict{Symbol, Int}()
    
    for region in keys(sim.regional_waves)
        regional_sums[region] = 0.0
        regional_counts[region] = 0
    end
    
    for node in values(sim.nodes)
        region = node.region
        regional_sums[region] += node.wave_sum
        regional_counts[region] += 1
    end
    
    for region in keys(sim.regional_waves)
        if regional_counts[region] > 0
            avg_wave = regional_sums[region] / regional_counts[region]
            push!(sim.regional_waves[region], avg_wave)
        end
    end
    
    # Frequency spectra
    band_powers = Dict{Symbol, Float64}()
    for band in keys(sim.frequency_spectra)
        band_powers[band] = 0.0
    end
    
    for node in values(sim.nodes)
        for (band, power) in node.wave_power
            band_powers[band] += power
        end
    end
    
    n_nodes = length(sim.nodes)
    for (band, total_power) in band_powers
        avg_power = total_power / n_nodes
        push!(sim.frequency_spectra[band], avg_power)
    end
end

# ============================================================================
# 9. VISUALIZATION & ANALYSIS
# ============================================================================

function analyze_phase_transition(sim::NeuroSheafSimulation)
    # Detect phase transitions from edge interaction patterns
    
    # Get current distribution of interaction types
    total_edges = sum(values(sim.edge_type_counts))
    if total_edges == 0
        return :unknown, 0.0
    end
    
    # Calculate percentages
    percentages = Dict{EdgeInteractionType, Float64}()
    for (type, count) in sim.edge_type_counts
        percentages[type] = count / total_edges
    end
    
    # Phase detection logic
    if percentages[AA] > 0.6
        phase = :opiate
        stability = percentages[AA]
        
    elseif percentages[AB] > 0.3
        phase = :critical
        stability = percentages[AB]
        
    elseif percentages[BB] > 0.5
        phase = :recovery
        stability = percentages[BB]
        
    else
        phase = :transition
        stability = 1.0 - max(percentages[AA], percentages[BB])
    end
    
    # Executive function indicator: theta dominance in PFC
    pfc_nodes = [n for n in values(sim.nodes) if n.region == :PFC]
    if !isempty(pfc_nodes)
        theta_power = mean([n.wave_power[:theta] for n in pfc_nodes])
        beta_power = mean([n.wave_power[:beta] for n in pfc_nodes])
        
        if theta_power > 1.5 * beta_power
            phase = Symbol("$(phase)_executive")
        end
    end
    
    return phase, stability
end

function print_status(sim::NeuroSheafSimulation)
    phase, stability = analyze_phase_transition(sim)
    
    println("\n" * "="^60)
    println("Time: $(round(sim.time, digits=1))s | Phase: $phase (stability: $(round(stability, digits=3)))")
    println("="^60)
    
    # Edge interaction distribution
    total_edges = sum(values(sim.edge_type_counts))
    println("\nEdge Interactions (5 types):")
    for type in instances(EdgeInteractionType)
        count = sim.edge_type_counts[type]
        percentage = 100 * count / total_edges
        type_str = string(type)
        println("  $type_str: $count edges ($(round(percentage, digits=1))%)")
    end
    
    # Frequency spectra
    println("\nFrequency Bands (average power):")
    for band in [:delta, :theta, :alpha, :beta, :gamma]
        if !isempty(sim.frequency_spectra[band])
            current_power = sim.frequency_spectra[band][end]
            println("  $band: $(round(current_power, digits=4))")
        end
    end
    
    # Regional waves
    println("\nRegional Wave Activity:")
    for region in [:PFC, :CUL4, :TH, :BG]
        if !isempty(sim.regional_waves[region])
            current_wave = sim.regional_waves[region][end]
            println("  $region: $(round(current_wave, digits=3))")
        end
    end
    
    # Molecular summary
    total_fent = sum([n.molecules.fentanyl for n in values(sim.nodes)])
    total_nolo = sum([n.molecules.noloxone for n in values(sim.nodes)])
    println("\nMolecular Summary:")
    println("  Total fentanyl: $(round(total_fent, digits=3))")
    println("  Total noloxone: $(round(total_nolo, digits=3))")
    println("  Ratio (F/N): $(round(total_fent/(total_nolo+1e-10), digits=3))")
end

# ============================================================================
# 10. MAIN SIMULATION
# ============================================================================

function run_complete_simulation()
    println("NEURO-SHEAF SIMULATION WITH EDGE HOPF OSCILLATORS")
    println("5 Interaction Types: AA, AB, BB, AN, BN")
    println("="^60)
    
    # Create simulation
    sim = NeuroSheafSimulation(200)  # 200 nodes
    
    # Run simulation for 10 minutes
    total_time = 600.0  # seconds
    n_steps = Int(total_time / sim.dt)
    print_interval = Int(60.0 / sim.dt)  # Print every minute
    
    phase_history = []
    
    for step in 1:n_steps
        simulate_step!(sim)
        
        if step % print_interval == 0
            print_status(sim)
            
            phase, stability = analyze_phase_transition(sim)
            push!(phase_history, (sim.time, phase, stability))
        end
    end
    
    # Final analysis
    println("\n" * "="^60)
    println("SIMULATION COMPLETE")
    println("="^60)
    
    println("\nPhase History:")
    for (t, phase, stability) in phase_history
        println("  $(round(t, digits=0))s: $phase (stability: $(round(stability, digits=2)))")
    end
    
    # Final interaction distribution
    println("\nFinal Edge Interaction Distribution:")
    total_edges = sum(values(sim.edge_type_counts))
    for type in instances(EdgeInteractionType)
        count = sim.edge_type_counts[type]
        percentage = 100 * count / total_edges
        type_str = string(type)
        println("  $type_str: $(round(percentage, digits=1))%")
    end
    
    # Frequency analysis
    println("\nDominant Frequency Bands:")
    band_powers = Dict{Symbol, Float64}()
    for band in keys(sim.frequency_spectra)
        if !isempty(sim.frequency_spectra[band])
            band_powers[band] = mean(sim.frequency_spectra[band][end-10:end])
        end
    end
    
    sorted_bands = sort(collect(band_powers), by=x->x[2], rev=true)
    for (band, power) in sorted_bands[1:3]
        println("  $band: $(round(power, digits=4))")
    end
    
    return sim, phase_history
end

# Run the simulation
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting simulation with edge Hopf oscillators and 5 interaction types...")
    sim, history = run_complete_simulation()
    
    println("\n" * "="^60)
    println("KEY FEATURES IMPLEMENTED:")
    println("="^60)
    println("1. Molecules at nodes → Edge interaction types (AA, AB, BB, AN, BN)")
    println("2. Hopf oscillators at edges with interaction-specific frequencies")
    println("3. Wave generation from oscillator phases")
    println("4. Node aggregation of edge waves")
    println("5. Frequency band assignment (delta, theta, alpha, beta, gamma)")
    println("6. Phase detection from interaction patterns")
    println("7. Executive function monitoring via theta/beta ratio in PFC")
    println("="^60)
end

