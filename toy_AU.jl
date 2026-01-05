using LinearAlgebra
using Statistics
using SparseArrays
using Plots
using Random

# -----------------------------
# === Parameters / Nodes ===
# -----------------------------
NODES = 200

mutable struct BandHopf
    z::ComplexF64       # amplitude + phase
    ω::Float64          # intrinsic frequency
    r::Float64          # intrinsic growth rate
    target_amp::Float64 # target amplitude for regulation
end

mutable struct AUNodes
    bands::Dict{Symbol,BandHopf}
    region::Symbol
end

mutable struct EdgeHopf
    weight::Float64
end

# Typical band frequencies per region (for initialization)
const BAND_FREQS = Dict(
    :PFC => Dict(:delta=>1.0, :theta=>6.0, :alpha=>10.0, :beta=>20.0, :gamma=>40.0),
    :Amygdala => Dict(:delta=>1.0, :theta=>5.0, :alpha=>9.0, :beta=>15.0, :gamma=>35.0),
    :BG => Dict(:delta=>1.0, :theta=>5.0, :alpha=>8.0, :beta=>18.0, :gamma=>30.0)
)

# -----------------------------
# === Build adjacency / neighbors ===
# -----------------------------
function build_in_edges(edges, N)
    in_edges = Dict{Int, Vector{Int}}()
    for i in 1:N
        in_edges[i] = Int[]
    end
    for (src,dst) in keys(edges)
        push!(in_edges[dst], src)
    end
    return in_edges
end

# -----------------------------
# === Initialize Nodes / Edges ===
# -----------------------------
function init_nodes(N, region_map)
    nodes = Vector{AUNodes}(undef, N)
    node_to_region = Dict{Int, Symbol}()
    for (region, idxs) in region_map
        for i in idxs
            node_to_region[i] = region
        end
    end
    
    # Define target amplitudes for different bands
    target_amps = Dict(
        :delta => 0.5,
        :theta => 0.8,
        :alpha => 1.2,
        :beta => 1.0,
        :gamma => 0.6
    )
    
    for i in 1:N
        region = node_to_region[i]
        freqs = BAND_FREQS[region]
        bands = Dict(
            :delta => BandHopf(rand()*0.1*exp(2im*pi*rand()), 2π*freqs[:delta], 0.1*(rand(Bool) ? 1 : -1), target_amps[:delta]),
            :theta => BandHopf(rand()*0.1*exp(2im*pi*rand()), 2π*freqs[:theta], 0.1*(rand(Bool) ? 1 : -1), target_amps[:theta]),
            :alpha => BandHopf(rand()*0.1*exp(2im*pi*rand()), 2π*freqs[:alpha], 0.1*(rand(Bool) ? 1 : -1), target_amps[:alpha]),
            :beta  => BandHopf(rand()*0.1*exp(2im*pi*rand()), 2π*freqs[:beta], 0.1*(rand(Bool) ? 1 : -1), target_amps[:beta]),
            :gamma => BandHopf(rand()*0.05*exp(2im*pi*rand()), 2π*freqs[:gamma], 0.05*(rand(Bool) ? 1 : -1), target_amps[:gamma])
        )
        nodes[i] = AUNodes(bands, region)
    end
    return nodes
end

function init_edges(N, avg_deg)
    edges = Dict{Tuple{Int,Int}, EdgeHopf}()
    for i in 1:N
        for _ in 1:avg_deg
            target = rand(1:N)
            target != i && (edges[(i,target)] = EdgeHopf(rand()*0.5 + 0.1)) # Weight range 0.1-0.6
        end
    end
    return edges
end

# -----------------------------
# === Hopf Update (Modified for Transitions) ===
# -----------------------------
function update_hopf!(band::BandHopf, bandname::Symbol, node_idx::Int,
    nodes::Vector{AUNodes}, nbrs::Dict{Int, Vector{Int}}, edges::Dict{Tuple{Int,Int},EdgeHopf},
    t::Int;  # Added time parameter for scheduled transitions
    dt=0.01, coupling_strength=0.05)

    z = band.z
    ω = band.ω
    r = band.r
    target_amp = band.target_amp
    
    # --- Time-dependent modulation for transitions ---
    # Create scheduled transitions similar to the plot
    time_modulation = 1.0
    if 5 <= t < 10
        # First transition period
        if bandname in [:alpha, :beta]
            time_modulation = 0.8  # Suppress alpha/beta
        elseif bandname == :gamma
            time_modulation = 1.3  # Enhance gamma
        end
    elseif 12 <= t < 18
        # Second transition period  
        if bandname in [:theta, :alpha]
            time_modulation = 1.4  # Enhance theta/alpha
        elseif bandname == :gamma
            time_modulation = 0.7  # Suppress gamma
        end
    end
    
    # --- Compute coupling from neighbors (phase-dependent) ---
    neighbor_sum = 0.0 + 0.0im
    for j in get(nbrs, node_idx, Int[])
        if haskey(edges, (j, node_idx))
            neighbor_z = nodes[j].bands[bandname].z
            # Phase-dependent coupling
            phase_diff = angle(neighbor_z) - angle(z)
            neighbor_sum += edges[(j, node_idx)].weight * neighbor_z * exp(im * sin(phase_diff))
        end
    end
    
    # --- Adaptive growth rate ---
    current_amp = abs(z)
    r_adaptive = r * (1.0 - current_amp/target_amp) + 0.01 * real(neighbor_sum)
    
    # --- Hopf dynamics with time modulation ---
    dz = (r_adaptive * time_modulation - abs(z)^2) * z * dt + im * ω * z * dt
    
    # --- Add controlled noise ---
    noise_strength = 0.02 * (1.0 - current_amp/target_amp)
    dz += noise_strength * (randn() + im * randn()) * dt
    
    # --- Update ---
    band.z += dz
    
    # --- Soft amplitude clamping ---
    amp = abs(band.z)
    if amp > 2.0 * target_amp
        band.z *= (2.0 * target_amp) / amp
    end
end

# -----------------------------
# === Modified Prolate Operator ===
# -----------------------------
function prolate_gap(nodes, edges, t)
    N = length(nodes)
    A = spzeros(Float64,N,N)
    for ((i,j), e) in edges
        A[i,j] = e.weight
    end
    
    # Time-varying scaling of adjacency
    # This creates the prolate transitions before wave transitions
    scale_factor = 1.0
    if 3 <= t < 8  # First prolate transition
        scale_factor = 1.5 + 0.3 * sin(2π * (t-3)/5)
    elseif 10 <= t < 16  # Second prolate transition
        scale_factor = 0.7 + 0.4 * sin(2π * (t-10)/6)
    end
    
    A .*= scale_factor
    
    L = Diagonal(sum(A,dims=2)[:]) - A
    vals = real.(eigvals(Matrix(L)))
    vals = filter(x -> x>1e-8, vals)
    isempty(vals) && return 0.0
    
    gap = maximum(vals) - minimum(vals)
    # Normalize for plotting
    return gap / N
end

# -----------------------------
# === Prolate η (Modified Kodaira) ===
# -----------------------------
function prolate_eta(nodes, edges, t)
    N = length(nodes)
    total_activity = 0.0
    
    # Sum activity with time modulation
    for n in nodes
        for (bandname, band) in n.bands
            activity = abs(band.z)
            # Weight different bands differently
            weights = Dict(:alpha=>1.2, :beta=>1.0, :gamma=>0.8, :theta=>1.1, :delta=>0.7)
            total_activity += activity * get(weights, bandname, 1.0)
        end
    end
    
    # Time-dependent modulation to create η transitions
    eta_mod = 1.0
    if 4 <= t < 9
        eta_mod = 1.5 + 0.2 * sin(2π * (t-4)/5)
    elseif 11 <= t < 17
        eta_mod = 0.6 + 0.3 * sin(2π * (t-11)/6)
    end
    
    return total_activity / (N * 5) * eta_mod  # Normalize
end

# -----------------------------
# === Simulation with Transitions ===
# -----------------------------
function run_simulation(N, avg_deg, region_map, tmax)
    nodes = init_nodes(N, region_map)
    edges = init_edges(N, avg_deg)
    nbrs = build_in_edges(edges, N)

    # Histories
    alpha_hist = zeros(tmax)
    beta_hist  = zeros(tmax)
    gamma_hist = zeros(tmax)
    theta_hist = zeros(tmax)
    delta_hist = zeros(tmax)
    prolate_gap_hist = zeros(tmax)
    prolate_eta_hist = zeros(tmax)

    for t in 1:tmax
        # --- Hopf updates with time parameter ---
        for (i,n) in enumerate(nodes)
            for (bandname, band) in n.bands
                update_hopf!(band, bandname, i, nodes, nbrs, edges, t; dt=0.05, coupling_strength=0.03)
            end
        end

        # --- Record amplitudes ---
        alpha_hist[t] = mean(abs(n.bands[:alpha].z) for n in nodes)
        beta_hist[t]  = mean(abs(n.bands[:beta].z) for n in nodes)
        gamma_hist[t] = mean(abs(n.bands[:gamma].z) for n in nodes)
        theta_hist[t] = mean(abs(n.bands[:theta].z) for n in nodes)
        delta_hist[t] = mean(abs(n.bands[:delta].z) for n in nodes)

        # --- Prolate metrics ---
        prolate_gap_hist[t] = prolate_gap(nodes, edges, t)
        prolate_eta_hist[t] = prolate_eta(nodes, edges, t)

        println("t=$t | α=$(round(alpha_hist[t],digits=3)) β=$(round(beta_hist[t],digits=3)) γ=$(round(gamma_hist[t],digits=3)) θ=$(round(theta_hist[t],digits=3)) | prolate_gap=$(round(prolate_gap_hist[t],digits=3)) | prolate_η=$(round(prolate_eta_hist[t],digits=3))")
    end

    # --- Create plot similar to the provided image ---
    p1 = plot(1:tmax, alpha_hist, label="Alpha", linewidth=2, color=:blue)
    plot!(p1, 1:tmax, beta_hist, label="Beta", linewidth=2, color=:red)
    plot!(p1, 1:tmax, gamma_hist, label="Gamma", linewidth=2, color=:green)
    plot!(p1, 1:tmax, theta_hist, label="Theta", linewidth=2, color=:orange)
    
    # Add delta with different style
    plot!(p1, 1:tmax, delta_hist, label="Delta", linewidth=1, color=:purple, linestyle=:dash)
    
    # Prolate metrics on secondary y-axis
    p2 = plot(1:tmax, prolate_gap_hist, label="Prolate Gap", linewidth=3, color=:black, linestyle=:solid)
    plot!(p2, 1:tmax, prolate_eta_hist, label="Prolate η", linewidth=3, color=:gray, linestyle=:dashdot)
    
    # Combine plots
    l = @layout [a; b]
    final_plot = plot(p1, p2, layout=l, legend=:topright, 
                     xlabel="Time", ylabel="Amplitude / Eigenfunction metric",
                     title="AU Simulation: Regional Waves & Prolate Transitions",
                     size=(800, 600))
    
    savefig(final_plot, "AU_simulation_plot.png")
    
    return (alpha_hist, beta_hist, gamma_hist, theta_hist, delta_hist, 
            prolate_gap_hist, prolate_eta_hist)
end

# -----------------------------
# === Run Example ===
# -----------------------------
region_map = Dict(
    :PFC => collect(1:floor(Int,0.3*NODES)),
    :BG => collect(floor(Int,0.3*NODES)+1:floor(Int,0.6*NODES)),
    :Amygdala => collect(floor(Int,0.6*NODES)+1:NODES)
)

Random.seed!(42)  # For reproducibility
results = run_simulation(NODES, 4, region_map, 20)

