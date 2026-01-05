using LinearAlgebra
using Statistics
using Plots
using Catlab
using SparseArrays
using Distributions  # For Poisson algebra
using Random


# === MORITA ALGEBRA CORE ===

struct MoritaAlgebra
    # Local data: (A, M, B) where M is A-B bimodule
    A::Matrix{ComplexF64}      # Endomorphism algebra at source
    B::Matrix{ComplexF64}      # Endomorphism algebra at target  
    M::Matrix{ComplexF64}      # Bimodule (stalk transport)
    
    # Derived invariants
    HH0::Float64               # Center/momentum map
    HH1::Float64               # Derivations/infinitesimal automorphisms
    HH2::Float64               # Deformations/obstructions
end

struct PoissonStalk
    position::Vector{Float64}     # Geometric position
    phase::Float64                # U(1) phase
    amplitude::Float64            # Oscillator amplitude
    momentum::Vector{Float64}     # Canonical momentum
    poisson_bracket::Matrix{Float64}  # Symplectic structure
end

# === VESSEL AS MORITA CONTEXT ===

struct VesselMorita
    id::Int
    source::PoissonStalk
    target::PoissonStalk
    bimodule::Matrix{ComplexF64}      # M: source → target
    bimodule_op::Matrix{ComplexF64}   # Mᴼᴾ: target → source
    morita_data::MoritaAlgebra
end

function rand_symplectic(n::Int)
    # Generate a random skew-symmetric matrix (Poisson tensor)
    # For n=6 (3 positions + 3 momenta)
    J = zeros(n, n)
    for i in 1:n
        for j in i+1:n
            val = randn()
            J[i,j] = val
            J[j,i] = -val
        end
    end
    # Ensure it's non-degenerate
    J += 0.1 * I  # Small regularization
    return J
end

function schur_complement(A::Matrix{Float64}, constraints::Matrix{Float64})
    # Compute Schur complement for Dirac reduction
    n = size(A, 1)
    m = size(constraints, 1)
    
    # Build extended matrix
    M = [A constraints';
         constraints zeros(m, m)]
    
    # Invert using block matrix formula
    if m > 0
        A_inv = pinv(A)
        S = constraints * A_inv * constraints'
        return A - A_inv * constraints' * pinv(S) * constraints * A_inv
    else
        return A
    end
end

function constraints(triangle::Vector{PoissonStalk})
    # Constraints for triangle reduction (closure conditions)
    n = length(triangle)
    dim = length(triangle[1].position)
    
    # Closure: sum of edge vectors = 0
    C = zeros(dim, n*dim)
    for i in 1:n
        C[:, (i-1)*dim+1:i*dim] = I(dim)
    end
    return C
end

function compute_massey_product(A::Matrix{ComplexF64}, 
                                B::Matrix{ComplexF64}, 
                                M::Matrix{ComplexF64})
    # Simplified Massey product for Morita context
    n = size(A, 1)
    AB = A * B
    MA = M * A
    BM = B * M
    
    # Massey triple product: ⟨[A], [B], [M]⟩
    massey = AB * M - A * BM + MA * B
    return massey
end

function compute_poisson_curvature(source::PoissonStalk, target::PoissonStalk)
    # Symplectic curvature = failure of Darboux theorem globally
    ω_source = pinv(source.poisson_bracket + 1e-8I)
    ω_target = pinv(target.poisson_bracket + 1e-8I)
    
    # Parallel transport curvature
    return norm(log(ω_source * ω_target'))
end

function poisson_bracket(stalk1::PoissonStalk, stalk2::PoissonStalk)
    # Compute Poisson bracket between two stalks
    Δx = stalk1.position - stalk2.position
    dist = norm(Δx)
    
    if dist < 1e-10
        return 0.0
    end
    
    # Symplectic form evaluated at difference
    ω1 = stalk1.poisson_bracket
    ω2 = stalk2.poisson_bracket
    ω_avg = (ω1 + ω2) / 2
    
    # Contract with position difference
    return Δx' * ω_avg * Δx / (dist^2 + 1e-8)
end

function ∂H_∂amplitude(stalk::PoissonStalk, H::Float64)
    # Derivative of Hamiltonian w.r.t amplitude
    return 2 * stalk.amplitude * H
end

function ∂H_∂phase(stalk::PoissonStalk, H::Float64)
    # Derivative of Hamiltonian w.r.t phase
    return 0.0  # Phase doesn't directly appear in simple H
end

function compute_noether_current(stalk::PoissonStalk, neighbors::Vector{PoissonStalk})
    # Compute Noether current from symmetries
    current = zeros(3)
    for nb in neighbors
        # Current proportional to phase difference
        phase_diff = nb.phase - stalk.phase
        pos_diff = nb.position - stalk.position
        current .+= sin(phase_diff) .* pos_diff
    end
    return current / (length(neighbors) + 1e-8)
end

function create_edge_morita(source::PoissonStalk, target::PoissonStalk)
    # Create a Morita algebra for an edge
    n = 3  # Dimension of simple algebra
    
    # Create random matrices for A and B (source and target algebras)
    A = randn(ComplexF64, n, n) + im * randn(n, n)
    B = randn(ComplexF64, n, n) + im * randn(n, n)
    
    # Make them approximately unitary
    A = A / norm(A)
    B = B / norm(B)
    
    # Bimodule M depends on stalk states
    phase_factor = exp(im * (target.phase - source.phase))
    amp_factor = sqrt(source.amplitude * target.amplitude)
    
    M = amp_factor * phase_factor * (A + B') / 2
    
    # Compute HH invariants
    HH0 = real(tr(A * B')) / (norm(A) * norm(B))
    
    crossed = [A M; M' B]
    HH1 = log(abs(det(crossed)) + 1e-10)
    
    massey = compute_massey_product(A, B, M)
    HH2 = norm(massey)
    
    return MoritaAlgebra(A, B, M, HH0, HH1, HH2)
end

function update_morita_from_stalks!(vessel::VesselMorita)
    # Update Morita algebra based on current stalk states
    phase_factor = exp(im * (vessel.target.phase - vessel.source.phase))
    amp_factor = sqrt(vessel.source.amplitude * vessel.target.amplitude)
    
    # Update bimodule
    vessel.bimodule .= amp_factor * phase_factor * 
                      (vessel.morita_data.A + vessel.morita_data.B') / 2
    vessel.bimodule_op .= conj(vessel.bimodule')
    
    # Recompute HH
    HH0, HH1, HH2 = compute_morita_HH(vessel)
    vessel.morita_data = MoritaAlgebra(
        vessel.morita_data.A,
        vessel.morita_data.B,
        vessel.bimodule,
        HH0, HH1, HH2
    )
end

function find_triangle_neighbors(vertex_idx::Int, complex)
    # Find all vertices connected to this one via triangles
    neighbors = Set{Int}()
    for tri in complex.triangles
        if vertex_idx in tri
            for v in tri
                if v != vertex_idx
                    push!(neighbors, v)
                end
            end
        end
    end
    return [complex.vertices[i] for i in neighbors]
end

function build_contraction_equivalence(vessel::VesselMorita)
    # Build Morita equivalence that contracts vessel to point
    A = vessel.morita_data.A
    B = vessel.morita_data.B
    M = vessel.bimodule
    
    # Construct equivalence: A ⊕ B ≃ ℂ via M
    n = size(A, 1)
    equivalence = zeros(ComplexF64, 2n, 2n)
    equivalence[1:n, 1:n] = A
    equivalence[n+1:end, n+1:end] = B
    equivalence[1:n, n+1:end] = M
    equivalence[n+1:end, 1:n] = M'
    
    return equivalence
end

function is_spherical_object(derived_end)
    # Simplified spherical test
    end_complex, d = derived_end
    
    # Check if complex is self-dual up to shift
    n = size(end_complex, 1)
    S = end_complex' * end_complex  # Approximate Serre functor
    
    # Check if S is close to identity (up to sign)
    return norm(S - I(n)) < 0.5
end

function twist_vessel_neighbors!(vessels::Vector{VesselMorita}, 
                                 vessel::VesselMorita, 
                                 equivalence::Matrix{ComplexF64})
    # Apply spherical twist to neighboring vessels
    for v in vessels
        if v.id != vessel.id
            # Check if v is connected to the contracted vessel
            if is_connected(v, vessel)
                # Apply twist: M_new = M_old * equivalence
                v.bimodule = v.bimodule * equivalence[1:size(v.bimodule,2), :]
                update_morita_from_stalks!(v)
            end
        end
    end
end

function is_connected(v1::VesselMorita, v2::VesselMorita)
    # Check if vessels share a vertex
    return (v1.source === v2.source || v1.source === v2.target ||
            v1.target === v2.source || v1.target === v2.target)
end

function find_triangle_clusters(triangles::Vector{Vector{Int}})
    # Group triangles that share edges
    clusters = Vector{Vector{Int}}()
    visited = falses(length(triangles))
    
    for i in 1:length(triangles)
        if !visited[i]
            cluster = Int[]
            stack = [i]
            
            while !isempty(stack)
                t = pop!(stack)
                if !visited[t]
                    visited[t] = true
                    push!(cluster, t)
                    
                    # Find neighbors sharing 2 vertices
                    for j in 1:length(triangles)
                        if !visited[j] && j != t
                            shared = intersect(triangles[t], triangles[j])
                            if length(shared) >= 2
                                push!(stack, j)
                            end
                        end
                    end
                end
            end
            
            if !isempty(cluster)
                push!(clusters, cluster)
            end
        end
    end
    
    return clusters
end

function compute_cluster_connection(cluster1::Vector{Int},
    cluster2::Vector{Int},
    complex)
    # Compute connection strength between two triangle clusters
    strength = 0.0

    for t1 in cluster1
        for t2 in cluster2
            tri1 = complex.triangles[t1]
            tri2 = complex.triangles[t2]

            # Find shared vertices
            shared = intersect(tri1, tri2)
            if !isempty(shared)
                # Connection strength based on number of shared vertices
                strength += length(shared)
            end
        end
    end

    return strength
end

function compute_momentum_map(stalk::PoissonStalk)
    # Compute momentum map for Marsden-Weinstein reduction
    # For simple case: J(x,p) = p (linear momentum)
    return stalk.momentum
end

function marsden_weinstein_reduction(stalk::PoissonStalk, J::Vector{Float64}, value::Float64)
    # Perform symplectic reduction at J = value

    # For now, just return a simplified stalk
    return PoissonStalk(
        stalk.position,
        stalk.phase,
        stalk.amplitude,
        zeros(3),  # Reduced momentum
        stalk.poisson_bracket  # Keep same Poisson structure
    )
end

function project_to_poisson(matrix::Matrix{Float64})
    # Project matrix to satisfy Poisson identities
    n = size(matrix, 1)
    
    # Ensure skew-symmetry
    poisson = (matrix - matrix') / 2
    
    # Ensure Jacobi identity approximately
    for i in 1:n
        for j in 1:n
            for k in 1:n
                # Jacobi: {x_i, {x_j, x_k}} + cyclic = 0
                jacobi = (poisson[i,j]*poisson[j,k] + 
                         poisson[j,k]*poisson[k,i] + 
                         poisson[k,i]*poisson[i,j])
                if abs(jacobi) > 0.1
                    # Adjust to reduce Jacobi violation
                    adjustment = jacobi / 3
                    poisson[i,j] -= adjustment
                    poisson[j,i] += adjustment
                end
            end
        end
    end
    
    return poisson
end

function find_poisson_singularities(complex)
    # Simplified singularity detection
    singularities = []
    
    for (i, stalk) in enumerate(complex.vertices)
        # Check rank of Poisson matrix
        rank_poisson = rank(stalk.poisson_bracket)
        
        if rank_poisson < 4 || rank_poisson % 2 != 0
            push!(singularities, Dict(
                :type => :poisson_degeneracy,
                :vertex => i,
                :rank => rank_poisson
            ))
        end
    end
    
    return singularities
end

function resolve_poisson_singularities!(complex, singularities::Vector)
    for sing in singularities
        if sing[:type] == :poisson_degeneracy
            i = sing[:vertex]
            stalk = complex.vertices[i]
            
            # Add small perturbation to improve rank
            perturbation = 0.01 * randn(size(stalk.poisson_bracket))
            stalk.poisson_bracket += perturbation - perturbation'
            
            # Ensure it's still Poisson
            stalk.poisson_bracket = project_to_poisson(stalk.poisson_bracket)
        end
    end
end


function compute_derived_endomorphisms(vessel::VesselMorita)
    # Compute derived endomorphisms for spherical test
    A = vessel.morita_data.A
    B = vessel.morita_data.B
    M = vessel.bimodule
    
    # Endomorphism complex
    end_complex = [A zeros(size(A)); zeros(size(B)) B]
    
    # Differential via M
    d = [zeros(size(A)) M; M' zeros(size(B))]
    
    return (end_complex, d)
end



# === TRIANGLE POISSON REDUCTION ===

function triangle_to_point(triangle::Vector{PoissonStalk})
    # Geometric center (Weil restriction)
    center = mean([s.position for s in triangle])
    
    # Phase coherence (gerbe data)
    phases = [s.phase for s in triangle]
    coherence = exp(im * (phases[1] - phases[2] + phases[3]))
    
    # Combined momentum (Noether theorem)
    total_momentum = sum([s.momentum for s in triangle])
    
    # Reduced Poisson bracket (Dirac reduction)
    ω_local = sum([s.poisson_bracket for s in triangle])
    ω_reduced = schur_complement(ω_local, constraints(triangle))
    
    return PoissonStalk(
        center,
        angle(coherence),
        mean([s.amplitude for s in triangle]),
        total_momentum,
        ω_reduced
    )
end

# === MORITA EQUIVALENCE & HH COMPUTATION ===

function compute_morita_HH(vessel::VesselMorita)
    A = vessel.morita_data.A
    B = vessel.morita_data.B
    M = vessel.bimodule
    
    # HH⁰ = Center of algebra (Casimirs)
    HH0 = trace(A * B') / (norm(A) * norm(B))
    
    # HH¹ = Outer derivations (infinitesimal symmetries)
    # Compute via crossed product A ⋊ M
    crossed = [A M; M' B]
    HH1 = log(abs(det(crossed)))  # Volume growth of automorphisms
    
    # HH² = Deformations (Poisson obstruction)
    # For Morita context (A, M, B), obstructions live in:
    # HH²(A, End_B(M)) ⊕ HH²(B, End_A(M))
    
    # Compute via Massey products
    obstruction = compute_massey_product(A, B, M)
    HH2 = norm(obstruction)
    
    return (HH0, HH1, HH2)
end

# === POISSON FLOW DYNAMICS ===

function poisson_flow!(stalk::PoissonStalk, neighbors::Vector{PoissonStalk}, dt::Float64)
    # Hamiltonian from neighboring stalks
    H = 0.0
    for nb in neighbors
        # Interaction via Poisson bracket
        Δx = stalk.position - nb.position
        H += exp(-norm(Δx)^2) * stalk.amplitude * nb.amplitude
    end
    
    # Add local Hopf term
    H_local = stalk.amplitude^2 * (stalk.momentum[1] - stalk.amplitude^4)
    H += H_local
    
    # Poisson evolution: dF/dt = {F, H}
    dphase = 0.0
    damplitude = 0.0
    
    for nb in neighbors
        # Compute Poisson bracket with neighbor
        pb = poisson_bracket(stalk, nb)
        
        # Phase coupling (Josephson-like)
        dphase += pb * sin(nb.phase - stalk.phase)
        
        # Amplitude coupling
        damplitude += pb * (nb.amplitude - stalk.amplitude)
    end
    
    # Update with Hamiltonian flow
    stalk.phase += dt * (dphase + ∂H_∂amplitude(stalk, H))
    stalk.amplitude += dt * (damplitude - ∂H_∂phase(stalk, H))
    
    # Update momentum via Noether
    stalk.momentum .+= dt * compute_noether_current(stalk, neighbors)
    
    return stalk
end

# === MORITA MMP (MINIMAL MODEL PROGRAM) ===

struct MoritaExtremalRay
    vessel_id::Int
    HH2_obstruction::Float64
    poisson_curvature::Float64  # Symplectic curvature
    contraction_map::Matrix{ComplexF64}  # Morita equivalence to point
end

function find_morita_extremal_rays(vessels::Vector{VesselMorita})
    rays = MoritaExtremalRay[]
    
    for v in vessels
        # Compute HH² obstruction
        _, _, HH2 = compute_morita_HH(v)
        
        # Compute Poisson curvature (symplectic curvature form)
        curvature = compute_poisson_curvature(v.source, v.target)
        
        # Build contraction as Morita equivalence to point algebra
        contraction = build_contraction_equivalence(v)
        
        push!(rays, MoritaExtremalRay(v.id, HH2, curvature, contraction))
    end
    
    # Sort by obstruction magnitude
    sort!(rays, by = r -> r.HH2_obstruction * r.poisson_curvature, rev = true)
    return rays
end

function morita_contraction!(vessels::Vector{VesselMorita}, ray::MoritaExtremalRay)
    # Apply Morita equivalence that collapses vessel to point
    # This is a derived version of blowing down
    
    v_idx = findfirst(v -> v.id == ray.vessel_id, vessels)
    v = vessels[v_idx]
    
    # Replace vessel with its Morita trivialization
    # A ⋊ M ⋊ B ≃ ℂ (point algebra) via Morita equivalence
    
    # 1. Compute derived endomorphisms
    derived_end = compute_derived_endomorphisms(v)
    
    # 2. Check if it's spherical (Calabi-Yau condition)
    if is_spherical_object(derived_end)
        # 3. Perform spherical twist (Seidel-Thomas)
        twist_vessel_neighbors!(vessels, v, ray.contraction_map)
        
        # 4. Remove contracted vessel
        deleteat!(vessels, v_idx)
        
        return true
    end
    
    return false  # Not contractible
end

# === SIMPLICIAL REDUCTION: ONLY TRIANGLES → POINTS ===

struct TriangleComplex
    vertices::Vector{PoissonStalk}
    edges::Vector{Tuple{Int, Int, VesselMorita}}
    triangles::Vector{Vector{Int}}  # Indices of vertices forming triangles
    
    # Reduced complex (after triangle → point)
    reduced_vertices::Vector{PoissonStalk}
    reduced_edges::Vector{Tuple{Int, Int, Float64}}  # Simplified

    TriangleComplex(vertices, edges, triangles) = 
        new(vertices, edges, triangles, [], [])
end

function reduce_triangles!(complex::TriangleComplex)
    # Group triangles that share edges (form tetrahedron-free clusters)
    clusters = find_triangle_clusters(complex.triangles)
    
    reduced_vertices = PoissonStalk[]
    reduced_edges = Tuple{Int, Int, Float64}[]
    
    for cluster in clusters
        # Collect all stalks in cluster
        cluster_stalks = unique(vcat([complex.triangles[i] for i in cluster]...))
        stalks = [complex.vertices[i] for i in cluster_stalks]
        
        # Reduce to single Poisson stalk
        reduced = triangle_to_point(stalks)
        push!(reduced_vertices, reduced)
        
        # Compute connections to other clusters
        for other_cluster in clusters
            if other_cluster != cluster
                # Find minimal edge between clusters
                connection_strength = compute_cluster_connection(
                    cluster, other_cluster, complex
                )
                if connection_strength > 0
                    push!(reduced_edges, (
                        length(reduced_vertices),  # Current index
                        findfirst(==(other_cluster), clusters),
                        connection_strength
                    ))
                end
            end
        end
    end
    
    complex.reduced_vertices = reduced_vertices
    complex.reduced_edges = reduced_edges
end

# === KODAIRA DIMENSION VIA MORITA THEORY ===

function morita_kodaira_dimension(vessels::Vector{VesselMorita})
    # κ = log-growth of HH*(Morita algebra)
    
    HH_dims = Float64[]
    for v in vessels
        HH0, HH1, HH2 = compute_morita_HH(v)
        push!(HH_dims, HH0 + HH1 + HH2)
    end
    
    # Kodaira dimension = logarithmic growth rate
    if isempty(HH_dims)
        return -Inf
    end
    
    # Sort by dimension and fit power law
    sorted_dims = sort(HH_dims)
    xs = log.(1:length(sorted_dims))
    ys = log.(sorted_dims .+ 1e-10)
    
    # Linear fit for growth rate
    coeff = cov(xs, ys) / var(xs)
    return coeff
end

# === MAIN SIMULATION WITH MORITA-POISSON DYNAMICS ===

function run_morita_simulation(N_vertices=200, N_triangles=100, steps=100)
    
    println("Initializing Poisson-Morita network...")
    
    # 1. Create Poisson stalks (vertices)
    vertices = [PoissonStalk(
        rand(3) .- 0.5,      # position in [-0.5, 0.5]³
        rand() * 2π,         # phase
        0.5 + 0.5rand(),     # amplitude
        randn(3),            # momentum
        rand_symplectic(6)   # 6x6 Poisson matrix (3 position, 3 momentum)
    ) for _ in 1:N_vertices]
    
    # 2. Create triangles (simplicial structure)
    triangles = [rand(1:N_vertices, 3) for _ in 1:N_triangles]
    
    # 3. Create vessels (edges with Morita data)
    vessels = VesselMorita[]
    vessel_counter = 1
    for (i, tri) in enumerate(triangles)
        # Create vessels for triangle edges
        for (a, b) in [(1,2), (2,3), (3,1)]
            source = vertices[tri[a]]
            target = vertices[tri[b]]
            
            # Create Morita algebra for this edge
            morita_result = create_edge_morita(source, target)
            
            push!(vessels, VesselMorita(
                vessel_counter,
                source, target,
                morita_result.M,  # Use M from the result
                conj(morita_result.M'),
                morita_result
            ))
            vessel_counter += 1
        end
    end
    
    # Build edges for complex
    edges = []
    for (i, v) in enumerate(vessels)
        # Find indices of source and target in vertices
        source_idx = findfirst(x -> x === v.source, vertices)
        target_idx = findfirst(x -> x === v.target, vertices)
        if source_idx !== nothing && target_idx !== nothing
            push!(edges, (source_idx, target_idx, v))
        end
    end
    
    complex = TriangleComplex(vertices, edges, triangles)
    
    # Simulation histories
    κ_history = Float64[]
    HH2_history = Float64[]
    flow_history = []

    
    for step in 1:steps
        println("\nStep $step:")
        
        # A. Poisson flow evolution - create new vertices array
        new_vertices = similar(vertices)
        for (i, v) in enumerate(vertices)
            neighbors = find_triangle_neighbors(i, complex)
            if !isempty(neighbors)
                stalk_copy = PoissonStalk(
                    copy(v.position),
                    v.phase,
                    v.amplitude,
                    copy(v.momentum),
                    copy(v.poisson_bracket)
                )
                new_vertices[i] = poisson_flow!(stalk_copy, neighbors, 0.01)
            else
                new_vertices[i] = v
            end
        end
        vertices = new_vertices
        
        # B. Update Morita algebras (stalks changed)
        for v in vessels
            # Update source/target references
            v.source = vertices[findfirst(x -> x === v.source, vertices)]
            v.target = vertices[findfirst(x -> x === v.target, vertices)]
            update_morita_from_stalks!(v)
        end
        
        # C. Compute Morita invariants
        κ = morita_kodaira_dimension(vessels)
        push!(κ_history, κ)
        
        # D. Compute total HH² obstruction
        total_HH2 = 0.0
        for v in vessels
            _, _, HH2 = compute_morita_HH(v)
            total_HH2 += HH2
        end
        push!(HH2_history, total_HH2)
        
        # E. Record flow observables
        total_flow = 0.0
        for v in vessels
            total_flow += abs(v.bimodule[1,1])
        end
        push!(flow_history, total_flow)
        
        println("  κ = $(round(κ, digits=3)), HH² = $(round(total_HH2, digits=3))")
        
        # F. Periodic Morita MMP
        if step % 10 == 0
            println("  Performing Morita MMP step...")
            
            # Find extremal rays (high obstruction vessels)
            rays = find_morita_extremal_rays(vessels)
            
            # Contract up to 5% most obstructed
            n_contract = max(1, Int(floor(0.05 * length(vessels))))
            contracted = 0
            
            for i in 1:min(n_contract, length(rays))
                if morita_contraction!(vessels, rays[i])
                    contracted += 1
                end
            end
            
            println("  Contracted $contracted vessels")
            
            # Reduce triangles if enough contraction occurred
            if contracted > 0
                reduce_triangles!(complex)
                println("  Reduced to $(length(complex.reduced_vertices)) vertices")
            end
        end
        
        # G. Detect Poisson singularities (symplectic leaves)
        singularities = find_poisson_singularities(complex)
        if !isempty(singularities)
            resolve_poisson_singularities!(complex, singularities)
        end
    end
    
    # 5. Visualization and analysis
    plot_results(κ_history, HH2_history, flow_history, complex)
    # Simple plot
    p1 = plot(1:steps, κ_history, label="κ(t)", title="Kodaira Dimension")
    p2 = plot(1:steps, HH2_history, label="HH²", title="Total Obstruction")
    p3 = plot(1:steps, flow_history, label="Flow", title="Network Flow")
    
    plot(p1, p2, p3, layout=(1,3), size=(1200, 400))
    savefig("morita_test.png")
    
    return complex, (κ_history, HH2_history, flow_history)
end

# === POISSON SINGULARITY DETECTION ===

function find_poisson_singularities(complex::TriangleComplex)
    singularities = []
    
    # Check Poisson bracket degeneracy (symplectic leaves)
    for (i, stalk) in enumerate(complex.vertices)
        # Rank of Poisson matrix should be even (symplectic)
        rank_poisson = rank(stalk.poisson_bracket)
        
        if rank_poisson % 2 != 0 || rank_poisson < 4
            # Degenerate Poisson structure = singularity
            push!(singularities, Dict(
                :type => :poisson_degeneracy,
                :vertex => i,
                :rank => rank_poisson,
                :expected => 6  # Full rank for 3D
            ))
        end
        
        # Check Casimir functions (functions that Poisson-commute with everything)
        num_casimirs = 6 - rank_poisson  # By Darboux theorem
        if num_casimirs > 2
            # Too many conserved quantities = integrable but singular
            push!(singularities, Dict(
                :type => :excessive_casimirs,
                :vertex => i,
                :num_casimirs => num_casimirs
            ))
        end
    end
    
    return singularities
end

function resolve_poisson_singularities!(complex::TriangleComplex, singularities::Vector)
    # Resolve via Marsden-Weinstein reduction
    
    for sing in singularities
        if sing[:type] == :poisson_degeneracy
            # Perform symplectic reduction
            i = sing[:vertex]
            stalk = complex.vertices[i]
            
            # Compute momentum map
            J = compute_momentum_map(stalk)
            
            # Reduce at regular value
            reduced_stalk = marsden_weinstein_reduction(stalk, J, 0.0)
            
            # Update in complex
            complex.vertices[i] = reduced_stalk
            
        elseif sing[:type] == :excessive_casimirs
            # Too many conserved quantities - need to break symmetry
            i = sing[:vertex]
            stalk = complex.vertices[i]
            
            # Add small perturbation to break extra symmetries
            perturbation = 0.01 * randn(size(stalk.poisson_bracket))
            stalk.poisson_bracket += perturbation + perturbation'
            
            # Renormalize to maintain Poisson identity
            stalk.poisson_bracket = project_to_poisson(stalk.poisson_bracket)
        end
    end
end

# === VISUALIZATION ===

function plot_results(κ_history, HH2_history, flow_history, complex)
    # 1. Kodaira dimension evolution
    p1 = plot(1:length(κ_history), κ_history,
              label="Morita κ(t)", xlabel="Step", ylabel="Kodaira Dimension",
              lw=2, title="Growth of Deformation Space")
    
    # 2. HH² obstruction evolution
    p2 = plot(1:length(HH2_history), HH2_history,
              label="Total HH²", xlabel="Step", ylabel="Obstruction",
              lw=2, color=:red, title="Associativity Obstruction")
    
    # 3. Flow amplitude
    p3 = plot(1:length(flow_history), flow_history,
              label="Total Flow", xlabel="Step", ylabel="Amplitude",
              lw=2, color=:green, title="Network Flow")
    
    # 4. Poisson structure visualization
    if !isempty(complex.reduced_vertices)
        positions = [v.position for v in complex.reduced_vertices]
        x = [p[1] for p in positions]
        y = [p[2] for p in positions]
        z = [p[3] for p in positions]
        
        p4 = scatter(x, y, z,
                     marker_z=[norm(v.momentum) for v in complex.reduced_vertices],
                     markersize=6, color=:viridis,
                     title="Reduced Poisson Network")
    else
        p4 = plot(legend=false, title="No reduced vertices")
    end
    
    plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))
    savefig("morita_poisson_simulation.png")
end

# === UTILITY FUNCTIONS (TO IMPLEMENT) ===

function compute_massey_product(A, B, M)
    # Implement Massey product for Morita context
    # Measures higher associativity obstructions
    AB = A * B
    MA = M * A
    BM = B * M
    
    # Massey triple product: ⟨[A], [B], [M]⟩
    return AB * M - A * BM + MA * B
end

function compute_poisson_curvature(source::PoissonStalk, target::PoissonStalk)
    # Symplectic curvature = failure of Darboux theorem globally
    ω_source = inv(source.poisson_bracket + 1e-8I)
    ω_target = inv(target.poisson_bracket + 1e-8I)
    
    # Parallel transport curvature
    return norm(log(ω_source * ω_target'))
end

function is_spherical_object(derived_end)
    # Check Calabi-Yau condition: HH*(End) ≅ HH*(point)[-d]
    # For vessel to be spherical (Seidel-Thomas)
    
    # Compute Serre functor
    S = compute_serre_functor(derived_end)
    
    # Check S ≅ [-d] (shift by dimension)
    return norm(S - (-1)^d) < 0.1
end

# === RUN SIMULATION ===

if abspath(PROGRAM_FILE) == @__FILE__
    println("="^60)
    println("Morita-Poisson Network Simulation")
    println("Truncated at triangles → HH² via Poisson algebra")
    println("="^60)
    
    complex, histories = run_morita_simulation(150, 80, 80)
    
    println("\n" * "="^60)
    println("Simulation Complete")
    println("Final vertices: $(length(complex.vertices))")
    println("Final reduced: $(length(complex.reduced_vertices))")
    println("Final κ: $(round(histories[1][end], digits=3))")
end

