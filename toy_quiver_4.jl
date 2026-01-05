using LinearAlgebra, Statistics, SparseArrays, Random

# ======================================================
# === Core Structures
# ======================================================

mutable struct MolecularLoad
    A::Float64
    B::Float64
end

mutable struct BandHopf
    z::ComplexF64
    ω::Float64
end

mutable struct MoritaAlgebra
    A::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    M::Matrix{ComplexF64}
end

mutable struct AUNode
    bands::Dict{Symbol,BandHopf}
    morita::MoritaAlgebra
    load::MolecularLoad
    c2::Float64
    state::Float64
end

mutable struct MoritaEdge
    src::Int
    dst::Int
    weight::Float64
    M::SparseMatrixCSC{Float64,Int}
    load::MolecularLoad
end

mutable struct MoritaQuiver
    nodes::Vector{AUNode}
    edges::Vector{MoritaEdge}
    adjacency::Vector{Vector{Int}}
end

# ======================================================
# === HH² / Massey Obstruction
# ======================================================

function massey_norm(A,B,M)
    norm((A*B)*M - A*(B*M) + (M*A)*B)
end

# ======================================================
# === Initialization
# ======================================================

function init_nodes(N)
    nodes = AUNode[]
    for _ in 1:N
        bands = Dict(
            :alpha => BandHopf(0.01randn()+0.01im*randn(), 10.0),
            :beta  => BandHopf(0.01randn()+0.01im*randn(), 20.0),
            :gamma => BandHopf(0.01randn()+0.01im*randn(), 40.0),
            :theta => BandHopf(0.01randn()+0.01im*randn(), 5.0)
        )
        I3 = Matrix{ComplexF64}(I,3,3)
        push!(nodes, AUNode(
            bands,
            MoritaAlgebra(I3, I3, I3),
            MolecularLoad(0.0,0.0),
            0.0,
            1.0
        ))
    end
    return nodes
end

function init_edges!(q::MoritaQuiver, avg_deg)
    N = length(q.nodes)
    for i in 1:N
        for _ in 1:avg_deg
            j = rand(1:N)          # self loop
            push!(q.edges,
                MoritaEdge(
                    i, j,
                    rand(),
                    sprand(3,3,0.4),
                    MolecularLoad(0.0,0.0)
                )
            )
        end
    end
    rebuild_adjacency!(q)
end

function rebuild_adjacency!(q::MoritaQuiver)
    q.adjacency = [Int[] for _ in q.nodes]
    for (k,e) in enumerate(q.edges)
        push!(q.adjacency[e.dst], k)
    end
end

function laplacian_eigenbasis(q::MoritaQuiver)
    L = graph_laplacian(q.adjacency)
    vals, vecs = eigen(Symmetric(L))
    return vals, vecs
end

function band_projector(evals, vecs; λmax)
    idx = findall(λ -> λ ≤ λmax, evals)
    return vecs[:,idx] * vecs[:,idx]'
end

function prolate_operator(q, region_nodes; λmax)
    evals, vecs = laplacian_eigenbasis(q)
    P_r = region_projector(q, region_nodes)
    P_b = band_projector(evals, vecs; λmax=λmax)
    return P_r * P_b * P_r
end

function prolate_gap(q, region_nodes; λmax)
    P = prolate_operator(q, region_nodes; λmax=λmax)
    eigvals = eigvals(Symmetric(P))
    return maximum(eigvals) - minimum(eigvals)
end

# ======================================================
# === Molecular Dynamics
# ======================================================

struct MoleculeSchedule
    t_on::Int
    t_off::Int
    amp::Float64
end

level(t,s) = (s.t_on ≤ t ≤ s.t_off) ? s.amp : 0.0

# ======================================================
# === Edge → Node Propagation
# ======================================================

function propagate_edge_loads!(q::MoritaQuiver)
    for e in q.edges
        q.nodes[e.dst].load.A += 0.2e.load.A
        q.nodes[e.dst].load.B += 0.2e.load.B
    end
end

# ======================================================
# === Edge Stress & Pruning
# ======================================================

edge_stress(e::MoritaEdge) = nnz(e.M) * abs(e.load.A - e.load.B)

function prune_edges!(q::MoritaQuiver; thresh=0.4)
    q.edges = [e for e in q.edges if e.src==e.dst || edge_stress(e)<thresh]
    rebuild_adjacency!(q)
end

# ======================================================
# === Hopf Dynamics (HH²-Coupled)
# ======================================================

function update_hopf!(band::BandHopf, node::AUNode, coupling, dt)
    r = -0.15 - 0.5node.load.A + 0.7node.load.B - 0.3node.c2
    ω = band.ω * exp(-0.2node.c2)
    z = band.z
    band.z += dt*((r + im*ω)*z - abs2(z)*z + coupling)
end

# ======================================================
# === Local Arithmetic Geometry Step
# ======================================================

function update_node_geometry!(node::AUNode)
    A,B,M = node.morita.A, node.morita.B, node.morita.M
    node.c2 = massey_norm(A,B,M)
    node.state = 1 / (1 + (node.c2/0.3)^6)

    # molecular decay
    node.load.A *= 0.92
    node.load.B *= 0.90
end

# ======================================================
# === Kodaira & Prolate
# ======================================================

function local_kodaira(i,q;ε=1e-8)
    nbrs = Int[]
    for e in q.edges
        (e.src==i || e.dst==i) && push!(nbrs, e.src==i ? e.dst : e.src)
    end
    length(nbrs)<2 && return -Inf

    mass = count(b->abs(b.z)>ε, values(q.nodes[i].bands))
    hh2 = sum(q.edges[k].weight^2 for k in q.adjacency[i])
    return log(hh2*mass + ε)
end

prolate_gap(v) = maximum(v)-minimum(v)

# ======================================================
# === Simulation
# ======================================================

function run_quiver_sim(N, avg_deg, tmax)
    nodes = init_nodes(N)
    q = MoritaQuiver(nodes, MoritaEdge[], [])
    init_edges!(q, avg_deg)

    α = zeros(tmax); gap = zeros(tmax)

    molA = MoleculeSchedule(3,10,0.6)
    molB = MoleculeSchedule(12,25,0.8)

    for t in 1:tmax
        # edge chemistry
        for e in q.edges
            e.load.A += level(t,molA)
            e.load.B += level(t,molB)
        end

        propagate_edge_loads!(q)

        for n in q.nodes
            update_node_geometry!(n)
        end

        for (i,n) in enumerate(q.nodes)
            for (k,b) in n.bands
                coup = sum(q.edges[j].weight *
                           q.nodes[q.edges[j].src].state *
                           q.nodes[q.edges[j].src].bands[k].z
                           for j in q.adjacency[i]; init=0+0im)
                update_hopf!(b,n,coup,0.02)
            end
        end

        prune_edges!(q)

        α[t] = mean(abs(n.bands[:alpha].z) for n in q.nodes)
        κ = [local_kodaira(i,q) for i in 1:N]
        gap[t] = prolate_gap(κ)
    end

    generate_simulation_plots(1:tmax, a_h, b_h, g_h, t_h, gap, eta_h)
    savefig("toy_quiver4.png")

    return α, gap
end
tmax = 20
NODES = 200
region_map = Dict(
    :PFC => collect(1:60),
    :BG => collect(61:120),
    :Amygdala => collect(121:200)
)
run_quiver_sim(NODES, 5, tmax)