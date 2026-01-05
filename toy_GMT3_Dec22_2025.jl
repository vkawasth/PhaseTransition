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
    for i in 1:N
        region = node_to_region[i]
        freqs = BAND_FREQS[region]
        bands = Dict(
            :delta => BandHopf(rand()*0.1*exp(2im*pi*rand()), 2π*freqs[:delta], 0.2*(rand(Bool) ? 1 : -1)),
            :theta => BandHopf(rand()*0.1*exp(2im*pi*rand()), 2π*freqs[:theta], 0.2*(rand(Bool) ? 1 : -1)),
            :alpha => BandHopf(rand()*0.1*exp(2im*pi*rand()), 2π*freqs[:alpha], 0.2*(rand(Bool) ? 1 : -1)),
            :beta  => BandHopf(rand()*0.1*exp(2im*pi*rand()), 2π*freqs[:beta], 0.2*(rand(Bool) ? 1 : -1)),
            :gamma => BandHopf(rand()*0.05*exp(2im*pi*rand()), 2π*freqs[:gamma], 0.2*(rand(Bool) ? 1 : -1))
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
            target != i && (edges[(i,target)] = EdgeHopf(rand()))
        end
    end
    return edges
end

# -----------------------------
# === Hopf Update (Local Power-driven) ===
# -----------------------------
function update_hopf!(band::BandHopf, bandname::Symbol, node_idx::Int,
    nodes::Vector{AUNodes}, nbrs::Dict{Int, Vector{Int}}, edges::Dict{Tuple{Int,Int},EdgeHopf};
    dt=0.01, coupling_strength=0.05, act_supp=0.0)

    z = band.z
    ω = band.ω
    r = band.r

    # --- Compute local input power from neighbors ---
    neighbor_activity = sum(abs(nodes[j].bands[bandname].z) * edges[(j,node_idx)].weight
                            for j in get(nbrs, node_idx, Int[]); init=0.0)

    # --- Effective growth rate ---
    r_eff = r + neighbor_activity + act_supp  # act_supp can be negative

    # --- Hopf dynamics ---
    dz = (r_eff - abs(z)^2) * z * dt + im * ω * z * dt

    # --- Small random fluctuation ---
    dz += 0.01*(rand()-0.5)*z

    # --- Update ---
    band.z += dz

    # --- Clamp amplitude ---
    amp = abs(band.z)
    amp > 5.0 && (band.z *= 5.0 / amp)
end

# -----------------------------
# === Extremal Rays / Blowdown ===
# -----------------------------
function extremal_rays(nodes, edges; bands=[:alpha,:beta,:gamma,:theta], τ=0.5)
    rays = Set{Tuple{Int,Int}}()
    for ((i,j), e1) in edges
        for ((j2,k), e2) in edges
            j != j2 && continue
            for b in bands
                xi = abs(nodes[i].bands[b].z)
                xj = abs(nodes[j].bands[b].z)
                xk = abs(nodes[k].bands[b].z)
                φ = abs(xk - xj) * e1.weight * e2.weight
                φ > τ && (push!(rays,(i,j)); push!(rays,(j,k)))
            end
        end
    end
    return rays
end

function categorical_blowdown!(edges, rays; factor=0.1)
    for e in rays
        haskey(edges,e) && (edges[e].weight *= factor)
    end
end

# -----------------------------
# === Prolate Operator ===
# -----------------------------
function prolate_gap(nodes, edges)
    N = length(nodes)
    A = spzeros(Float64,N,N)
    for ((i,j), e) in edges
        A[i,j] = e.weight
    end
    L = Diagonal(sum(A,dims=2)[:]) - A
    vals = real.(eigvals(Matrix(L)))
    vals = filter(x -> x>1e-8, vals)
    isempty(vals) && return 0.0
    return maximum(vals) - minimum(vals)
end

# -----------------------------
# === Local Kodaira Dimension ===
# -----------------------------
function local_kodaira_dimension(i::Int, nodes::Vector{AUNodes}, edges::Dict{Tuple{Int,Int},EdgeHopf}; ε=1e-12)
    node = nodes[i]
    nbrs = [u==i ? v : u for (u,v) in keys(edges) if u==i || v==i]
    length(nbrs)<2 && return -Inf

    hh2_mass = 0.0
    band_contrib = sum(abs(b.z) for b in values(node.bands))

    for a in 1:length(nbrs)-1
        j = nbrs[a]
        eij = get(edges, (i,j), nothing)
        eij === nothing && continue
        for b in a+1:length(nbrs)
            k = nbrs[b]
            eik = get(edges, (i,k), nothing)
            eik === nothing && continue
            hh2_mass += abs(eij.weight * eik.weight) * band_contrib
        end
    end

    return log(max(hh2_mass + ε, 1e-12))
end

function compute_kodaira(nodes, edges)
    [local_kodaira_dimension(i, nodes, edges) for i in 1:length(nodes)]
end

# -----------------------------
# === Simulation ===
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
    prolate_hist = Float64[]
    kodaira_hist = Float64[]

    for t in 1:tmax
        # --- Hopf updates ---
        for (i,n) in enumerate(nodes)
            for (bandname, band) in n.bands
                update_hopf!(band, bandname, i, nodes, nbrs, edges; dt=0.05, coupling_strength=0.05, act_supp=0.0)
            end
        end

        # --- Extremal rays / blowdown ---
        rays = extremal_rays(nodes, edges)
        categorical_blowdown!(edges, rays; factor=0.5)

        # --- Record amplitudes ---
        alpha_hist[t] = mean(abs(n.bands[:alpha].z) for n in nodes)
        beta_hist[t]  = mean(abs(n.bands[:beta].z) for n in nodes)
        gamma_hist[t] = mean(abs(n.bands[:gamma].z) for n in nodes)
        theta_hist[t] = mean(abs(n.bands[:theta].z) for n in nodes)
        delta_hist[t] = mean(abs(n.bands[:delta].z) for n in nodes)

        # --- Prolate gap ---
        gap = prolate_gap(nodes, edges)
        push!(prolate_hist, gap)

        # --- Kodaira dimension ---
        k_hist = mean(compute_kodaira(nodes, edges))
        push!(kodaira_hist, k_hist)

        println("t=$t | α=$(round(alpha_hist[t],digits=3)) β=$(round(beta_hist[t],digits=3)) γ=$(round(gamma_hist[t],digits=3)) θ=$(round(theta_hist[t],digits=3)) | prolate=$(round(gap,digits=3)) | kodaira=$(round(k_hist,digits = 3))")
    end

    # --- Plot ---
    plot(1:tmax, alpha_hist,label="Alpha")
    plot!(1:tmax, beta_hist,label="Beta")
    plot!(1:tmax, gamma_hist,label="Gamma")
    plot!(1:tmax, theta_hist,label="Theta")
    plot!(1:tmax, delta_hist,label="Delta")
    plot!(1:tmax, prolate_hist,label="Prolate Gap", lw=2, ls=:dash)
    plot!(1:tmax, kodaira_hist,label="Prolate Gap", lw=2, ls=:dash)
    savefig("simulation_final.png")

    return (alpha_hist,beta_hist,gamma_hist,theta_hist,delta_hist,prolate_hist,kodaira_hist)
end

# -----------------------------
# === Run Example ===
# -----------------------------
region_map = Dict(
    :PFC => collect(1:floor(Int,0.3*NODES)),
    :BG => collect(floor(Int,0.3*NODES)+1:floor(Int,0.6*NODES)),
    :Amygdala => collect(floor(Int,0.6*NODES)+1:NODES)
)

run_simulation(NODES, 5, region_map, 20)
