using LinearAlgebra, SparseArrays, Statistics

# -------------------------------
# 1. Parameters
# -------------------------------
const N = 1000
τ_d = 30.0      # dopamine half-life
τ_o = 120.0     # opiate half-life
τ_c2 = 200.0    # intersection persistence
ε_c2 = 1e-3     # tolerance for exactness
ρ_min = 1e-3    # pruning threshold
dt = 0.1

#alpha_power ≈ spectral_power_between(8,12)
#theta_power ≈ spectral_power_between(4,8)
#beta_power ≈ spectral_power_between(13,30)
#gamma_power ≈ spectral_power_above(30)

# -------------------------------
# 2. Graph and neighbors
# -------------------------------
neighbors(i) = filter(j -> abs(j - i) ≤ 2 && j != i, 1:N)

# -------------------------------
# 3. AU Node and Edge Structures
# -------------------------------
mutable struct AUNodes
    state::Float64
    c0::Float64
    c1::Float64
    c2::Float64
end

mutable struct EdgeHopf
    ω::Float64
    γ::Float64
    phase::Float64
end

# -------------------------------
# 4. Initialize nodes and edges
# -------------------------------
nodes = [AUNodes(0.0, 0.0, 0.0, 0.0) for _ in 1:N]
edges = Dict{Tuple{Int,Int},EdgeHopf}()

for i in 1:N
    for j in neighbors(i)
        edges[(i,j)] = EdgeHopf(rand(), 0.01, 0.0)
    end
end

# -------------------------------
# 5. Edge update (AU + Hopf)
# -------------------------------
function edge_interaction!(A::AUNodes, B::AUNodes, e::EdgeHopf, dt::Float64;
    alpha_c1=0.2, alpha_c0=0.1, beta_inter=0.05)
    #----------------------------------------
    # Hopf oscillator (edge activity)
    #----------------------------------------
    e.phase += e.ω * dt
    e.ω *= exp(-e.γ * dt)  # exponential damping of edge frequency

    #----------------------------------------
    # Transport along edge for c1 (dopamine)
    #----------------------------------------
    transport_c1 = alpha_c1 * e.ω * (A.c1 - B.c1)

    #----------------------------------------
    # Transport along edge for c0 (opiates)
    #----------------------------------------
    transport_c0 = alpha_c0 * e.ω * (A.c0 - B.c0)

    #----------------------------------------
    # Nonlinear intersection term (cross-molecule interaction)
    #----------------------------------------
    intersection = beta_inter * A.c2 * B.c1  # A.c2 interacts with B.c1

    #----------------------------------------
    # Update AU node states (accumulation)
    #----------------------------------------
    A.state += transport_c1 - transport_c0 - intersection - A.c0
    B.state -= transport_c1 - transport_c0 - intersection - B.c0

    # Optional: soft decay to avoid runaway accumulation
    A.state *= 0.99
    B.state *= 0.99
end

function step!(nodes::Vector{AUNodes}, edges::Dict{Tuple{Int,Int},EdgeHopf}, dt::Float64)
    for ((i,j), e) in edges
        edge_interaction!(nodes[i], nodes[j], e, dt)
    end
end

# -------------------------------
# 6. Molecule deployment (wavefront)
# -------------------------------
function deploy!(nodes::Vector{AUNodes}, center::Int, radius::Int, kind::Symbol)
    for i in max(1, center-radius):min(length(nodes), center+radius)
        if kind == :dopamine
            nodes[i].c1 += 1.0
        elseif kind == :opiate
            nodes[i].c0 += 0.5
        end
        nodes[i].c2 += 0.1   # intersection memory
    end
end

# -------------------------------
# 7. Half-life decay of algebraic influence
# -------------------------------
function decay!(nodes::Vector{AUNodes}, dt::Float64)
    for n in nodes
        n.c1 *= exp(-dt/τ_d)
        n.c0 *= exp(-dt/τ_o)
        n.c2 *= exp(-dt/τ_c2)
    end
end

# -------------------------------
# 8. Blow-up / Blow-down pruning
# -------------------------------
function blow_up(node::AUNodes)
    tangent_state = node.state / (abs(node.state)+1e-8)
    return AUNodes(tangent_state, node.c0, node.c1, node.c2)
end

function can_prune(node::AUNodes)
    exact_c2 = abs(node.c2) < ε_c2
    low_density = node.c0 < ρ_min
    low_transport = abs(node.c1) < ρ_min
    return exact_c2 && low_density && low_transport
end

function prune_and_reindex!(nodes::Vector{AUNodes}, edges::Dict{Tuple{Int,Int},EdgeHopf}; 
    threshold=1e-6, decay_factor=0.99)

    old_N = length(nodes)
    new_nodes = Vector{AUNodes}()
    old_to_new = Dict{Int,Int}()

    counter = 1
    for old_idx in 1:old_N
        # Soft decay
        nodes[old_idx].c0 *= decay_factor
        nodes[old_idx].c1 *= decay_factor
        nodes[old_idx].c2 *= decay_factor

        influence = abs(nodes[old_idx].c0) + abs(nodes[old_idx].c1) + abs(nodes[old_idx].c2)
        if influence > threshold
            push!(new_nodes, nodes[old_idx])
            old_to_new[old_idx] = counter
            counter += 1
        end
    end

    # Rebuild edges only using surviving nodes
    new_edges = Dict{Tuple{Int,Int}, EdgeHopf}()
    for ((i,j), e) in edges
        if haskey(old_to_new,i) && haskey(old_to_new,j)
            new_edges[(old_to_new[i], old_to_new[j])] = e
        end
    end

    # Replace old nodes and edges
    empty!(nodes)
    append!(nodes, new_nodes)

    empty!(edges)
    merge!(edges, new_edges)

    return old_to_new
end


function prune!(nodes::Vector{AUNodes}, edges::Dict{Tuple{Int,Int},EdgeHopf})
    to_remove = []
    Nn = length(nodes)
    for i in 1:Nn
        if can_prune(nodes[i])
            push!(to_remove, i)
        end
    end
    for i in reverse(sort(to_remove))
        deleteat!(nodes, i)
        keys_to_delete = [k for k in keys(edges) if i in k]
        for k in keys_to_delete
            delete!(edges, k)
        end
    end
end

# -------------------------------
# 9. Enhanced Prolate / Slepian Observer
# -------------------------------
# Build tridiagonal Jacobi matrix for graph (1D ladder approximation)
function jacobi_tridiag(nodes::Vector{AUNodes}, edges::Dict{Tuple{Int,Int},EdgeHopf})
    N = length(nodes)
    main_diag = zeros(Float64, N)
    off_diag  = zeros(Float64, N-1)

    # Fill main diagonal and off-diagonal
    for i in 1:N
        # Main diagonal: sum of edge weights connected to node i
        main_diag[i] = 0.0
    end

    for ((i,j), e) in edges
        if abs(i-j) == 1  # Only nearest neighbors for tridiagonal
            w = e.ω
            off_diag[min(i,j)] = w
            main_diag[i] -= w
            main_diag[j] -= w
        end
    end

    return Tridiagonal(off_diag, main_diag, off_diag)
end


function propagate_wavefront!(nodes::Vector{AUNodes}; factor_c1=0.2, factor_c0=0.1)
    N = length(nodes)
    new_c1 = zeros(N)
    new_c0 = zeros(N)
    
    # Spread along neighbors
    for i in 1:N
        for j in max(1,i-1):min(N,i+1)
            new_c1[j] += nodes[i].c1 * factor_c1
            new_c0[j] += nodes[i].c0 * factor_c0
        end
    end
    
    # Apply updates
    for i in 1:N
        nodes[i].c1 += new_c1[i]
        nodes[i].c0 += new_c0[i]
    end
end

function prolate_observer(nodes::Vector{AUNodes}, edges::Dict{Tuple{Int,Int},EdgeHopf}; min_nodes=10)
    N = length(nodes)
    if N < min_nodes
        return nothing
    end

    clamp_val = 1e6
    for n in nodes
        n.c0 = clamp(n.c0, -clamp_val, clamp_val)
        n.c1 = clamp(n.c1, -clamp_val, clamp_val)
        n.c2 = clamp(n.c2, -clamp_val, clamp_val)
        n.state = clamp(n.state, -clamp_val, clamp_val)
    end

    # Build tridiagonal Jacobi
    J = jacobi_tridiag(nodes, edges)

    # Debug print
    println("Prolate observer: sum(J)=", sum(abs.(J)), " max(J)=", maximum(abs.(J)))

    # Skip invalid matrices
    if any(isnan, J) || any(isinf, J) || all(J .== 0.0)
        println("Skipping prolate observer: Jacobi matrix invalid")
        return nothing
    end

    # Safe eigen computation
    eig_result = nothing
    try
        eig_result = eigen(J)
    catch err
        println("Skipping prolate observer: eigen failed - ", err)
        return nothing
    end

    # Extract results safely
    λ = eig_result.values
    U = eig_result.vectors
    gap = N >= 2 ? λ[end] - λ[end-1] : 0.0
    eta = sum(abs.(U[:,end])) / N

    return λ, U, gap, eta
end






# -------------------------------
# 10. Cadence vs noise classifier
# -------------------------------
function classify(gap_hist::Vector{Float64}, eta_hist::Vector{Float64})
    if length(gap_hist) < 10 || length(eta_hist) < 11
        return :unknown
    end
    smooth_gap = all(diff(gap_hist[end-5:end]) .< 0)
    smooth_eta = std(eta_hist[end-10:end]) < 0.2*mean(eta_hist)
    return (smooth_gap && smooth_eta) ? :true_cadence : :noise
end

# =========================
# Simulation parameters
# =========================
dt = 0.01
decay_factor = 0.99
prune_threshold = 1e-8   # low enough to avoid immediate pruning
total_timesteps = 1000
min_nodes_prolate = 10

# =========================
# Initialize node states safely
# =========================
for n in nodes
    n.c0 = rand() * 0.1    # small initial molecule concentrations
    n.c1 = rand() * 0.1
    n.c2 = rand() * 0.1
    n.state = rand() * 0.1
end

gap_hist = Float64[]
eta_hist = Float64[]

# =========================
# Main simulation loop
# =========================
for t in 1:total_timesteps
    N = length(nodes)

    # -------------------------
    # Step 1: propagate waves
    # -------------------------
    for ((i,j), e) in edges
        A, B = nodes[i], nodes[j]

        # Hopf oscillator
        e.phase += e.ω * dt
        e.ω *= exp(-e.γ * dt)
        e.ω = max(e.ω, 1e-12)

        # Transport / intersection
        transport = e.ω * (A.c1 - B.c1)
        intersection = A.c2 * B.c1

        # Update AU states including c0
        A.state += transport - intersection - A.c0
        B.state -= transport - intersection - B.c0

        # Clamp states to avoid NaN/Inf
        clamp_val = 1e6
        A.c0 = clamp(A.c0, -clamp_val, clamp_val)
        A.c1 = clamp(A.c1, -clamp_val, clamp_val)
        A.c2 = clamp(A.c2, -clamp_val, clamp_val)
        B.c0 = clamp(B.c0, -clamp_val, clamp_val)
        B.c1 = clamp(B.c1, -clamp_val, clamp_val)
        B.c2 = clamp(B.c2, -clamp_val, clamp_val)
    end

    # -------------------------
    # Step 2: gradual node decay
    # -------------------------
    for n in nodes
        n.c0 *= decay_factor
        n.c1 *= decay_factor
        n.c2 *= decay_factor
        n.state *= decay_factor
    end

    # -------------------------
    # Step 3: prune and reindex nodes/edges
    # -------------------------
    old_to_new = Dict{Int,Int}()
    new_nodes = Vector{AUNodes}()
    counter = 1
    for old_idx in 1:N
        influence = abs(nodes[old_idx].c0) + abs(nodes[old_idx].c1) + abs(nodes[old_idx].c2)
        # Ensure initial survival to prevent collapse
        if influence > prune_threshold || t < 10
            push!(new_nodes, nodes[old_idx])
            old_to_new[old_idx] = counter
            counter += 1
        end
    end

    # Rebuild edges using new indices
    new_edges = Dict{Tuple{Int,Int},EdgeHopf}()
    for ((i,j), e) in edges
        if haskey(old_to_new,i) && haskey(old_to_new,j)
            new_edges[(old_to_new[i], old_to_new[j])] = e
        end
    end

    # Replace nodes and edges
    empty!(nodes)
    append!(nodes, new_nodes)
    empty!(edges)
    merge!(edges, new_edges)

    println("t=$t, remaining nodes=$(length(nodes)), remaining edges=$(length(edges))")

    # -------------------------
    # Step 4: prolate observer
    # -------------------------
    if length(nodes) >= min_nodes_prolate
        result = prolate_observer(nodes, edges; min_nodes=min_nodes_prolate)
        if result !== nothing
            λ, U, gap, eta = result
            push!(gap_hist, gap)
            push!(eta_hist, eta)
        else
            println("Skipping prolate observer: eigen failed or too few nodes")
        end
    else
        println("Skipping prolate observer: too few nodes")
    end

    # -------------------------
    # Step 5: cadence detection
    # -------------------------
    if t > 500 && length(gap_hist) >= 2
        state = classify(gap_hist, eta_hist)
        if state == :true_cadence
            println("True cadence detected at t=$t")
        end
    end
end



