# NeuroSheaf_TopologicalPhaseTransitions.jl
# Phase transitions as topological changes in Hochschild cohomology

using LinearAlgebra
using SparseArrays
using TensorOperations
using HomotopyContinuation
using Arpack

module TopologicalPhaseTransitions

export TopologicalPhase, MonodromyOperator, PicardGroup, KodairaDimension,
       compute_monodromy, compute_picard_group, detect_topology_change,
       resolve_singularity_blowup, compute_kodaira_dimension,
       sheaf_cohomology_change, compute_chern_classes,
       topological_phase_diagram

# ==================== TOPOLOGICAL PHASE STRUCTURE ====================

struct TopologicalPhase
    phase_id::Symbol  # :opiate, :critical, :transition, :norcain
    topological_invariant::Dict{Symbol, Any}
    
    # Algebraic topology data
    fundamental_group::Matrix{Int}  # π₁ representation
    homology_groups::Vector{Matrix{Int}}  # Hₖ
    cohomology_ring::Dict{Tuple{Int,Int}, Int}  # Cup product structure
    
    # Complex geometry data
    chern_classes::Vector{Float64}  # c₁, c₂, ...
    kodaira_dimension::Float64  # κ ∈ {-∞, 0, 1, 2, ...}
    picard_number::Int  # ρ = rank(Pic)
    
    # Hochschild data
    HH_structure::Dict{Int, Array{ComplexF64}}  # HH⁰, HH¹, HH², HH³
    deformation_space::Matrix{Float64}  # Tangent space to moduli space
    
    function TopologicalPhase(phase::Symbol, algebra_dim::Int=3)
        # Phase-specific topological invariants
        
        if phase == :opiate
            # Opiate phase: trivial topology (contractible)
            π₁ = [1 0; 0 1]  # Trivial fundamental group
            H = [Matrix(1I, 1, 1)]  # H₀ = ℤ
            
            # Trivial cohomology ring
            cohom_ring = Dict((0,0) => 1)
            
            # Chern classes (trivial bundle)
            chern = [0.0, 0.0, 0.0]
            κ = -Inf  # Kodaira dimension -∞ (Fano-like)
            ρ = 1     # Picard number 1
            
            # Hochschild: only center is non-trivial
            HH0 = ones(ComplexF64, algebra_dim)
            HH1 = zeros(ComplexF64, algebra_dim, algebra_dim)
            HH2 = zeros(ComplexF64, algebra_dim, algebra_dim, algebra_dim, algebra_dim)
            HH3 = zeros(ComplexF64, algebra_dim, algebra_dim, algebra_dim, algebra_dim, algebra_dim)
            
            HH_struct = Dict(0 => HH0, 1 => HH1, 2 => HH2, 3 => HH3)
            
        elseif phase == :critical
            # Critical phase: non-trivial π₁ (S¹-like)
            π₁ = [0 1; -1 0]  # ℤ representation
            H = [Matrix(1I, 1, 1), zeros(1, 1)]  # H₀ = ℤ, H₁ = ℤ
            
            # Cohomology ring of circle
            cohom_ring = Dict((0,0) => 1, (1,1) => 0)
            
            # Non-trivial Chern class
            chern = [1.0, 0.0, 0.0]  # c₁ = 1 (line bundle)
            κ = 0.0  # Kodaira dimension 0 (Calabi-Yau)
            ρ = 2    # Picard number 2
            
            # Hochschild: non-trivial derivations
            HH0 = ones(ComplexF64, algebra_dim)
            HH1 = randn(ComplexF64, algebra_dim, algebra_dim)  # Outer derivations
            HH2 = randn(ComplexF64, algebra_dim, algebra_dim, algebra_dim, algebra_dim) * 0.1
            HH3 = zeros(ComplexF64, algebra_dim, algebra_dim, algebra_dim, algebra_dim, algebra_dim)
            
            HH_struct = Dict(0 => HH0, 1 => HH1, 2 => HH2, 3 => HH3)
            
        elseif phase == :transition
            # Transition phase: non-commutative π₁
            π₁ = [1 2; 2 1]  # Non-abelian representation
            H = [Matrix(1I, 1, 1), zeros(2, 2), Matrix(1I, 1, 1)]  # H₀ = ℤ, H₂ = ℤ
            
            # More complex cohomology ring
            cohom_ring = Dict((0,0) => 1, (2,2) => 1, (0,2) => 1, (2,0) => 1)
            
            # Higher Chern classes
            chern = [2.0, 1.0, 0.5]
            κ = 1.0  # Kodaira dimension 1
            ρ = 3    # Picard number 3
            
            # Hochschild: non-trivial HH² (deformations)
            HH0 = ones(ComplexF64, algebra_dim)
            HH1 = randn(ComplexF64, algebra_dim, algebra_dim)
            HH2 = randn(ComplexF64, algebra_dim, algebra_dim, algebra_dim, algebra_dim) * 0.5
            HH3 = randn(ComplexF64, algebra_dim, algebra_dim, algebra_dim, algebra_dim, algebra_dim) * 0.1
            
            HH_struct = Dict(0 => HH0, 1 => HH1, 2 => HH2, 3 => HH3)
            
        else  # :norcain
            # Norcain phase: different topology (maybe ℂP¹-like)
            π₁ = [1 0; 0 1]  # Simply connected
            H = [Matrix(1I, 1, 1), zeros(2, 2), Matrix(1I, 1, 1)]  # H₀ = H₂ = ℤ
            
            # Cohomology ring of ℂP¹
            cohom_ring = Dict((0,0) => 1, (2,2) => 1, (0,2) => 1, (2,0) => 1)
            
            # Chern classes of ℂP¹ bundle
            chern = [2.0, 1.0, 0.0]
            κ = -Inf  # Fano again
            ρ = 1     # Picard number 1
            
            # Hochschild: different deformation pattern
            HH0 = ones(ComplexF64, algebra_dim)
            HH1 = zeros(ComplexF64, algebra_dim, algebra_dim)  # Trivial derivations
            HH2 = -randn(ComplexF64, algebra_dim, algebra_dim, algebra_dim, algebra_dim) * 0.3  # Anti-opiate
            HH3 = zeros(ComplexF64, algebra_dim, algebra_dim, algebra_dim, algebra_dim, algebra_dim)
            
            HH_struct = Dict(0 => HH0, 1 => HH1, 2 => HH2, 3 => HH3)
        end
        
        # Deformation space (moduli space tangent space)
        # This encodes how the phase can deform
        if phase == :critical
            # Critical phase has large deformation space
            deformation_dim = algebra_dim^2 + 1
        else
            deformation_dim = algebra_dim
        end
        deformation_space = randn(deformation_dim, deformation_dim)
        
        topological_invariant = Dict(
            :π₁ => π₁,
            :H => H,
            :cohomology_ring => cohom_ring,
            :chern => chern,
            :κ => κ,
            :ρ => ρ,
            :HH => HH_struct
        )
        
        new(phase, topological_invariant, π₁, H, cohom_ring, 
            chern, κ, ρ, HH_struct, deformation_space)
    end
end

# ==================== MONODROMY OPERATOR (TOPOLOGICAL DEFECTS) ====================

struct MonodromyOperator
    # Monodromy around topological defects
    # When traversing a loop around a defect, states get multiplied by M
    
    matrix::Matrix{ComplexF64}
    defect_type::Symbol  # :vortex, :disclination, :monopole, :frank
    winding_number::Int
    berry_phase::Float64  # Geometric phase accumulation
    
    # Holonomy data
    parallel_transport::Function
    curvature_form::Matrix{ComplexF64}
    
    function MonodromyOperator(defect::Symbol, algebra_dim::Int=3)
        if defect == :vortex
            # Vortex: phase winding by 2π
            M = diagm([exp(2π*im*k/algebra_dim) for k in 0:algebra_dim-1])
            winding = 1
            berry = 2π
            
        elseif defect == :disclination
            # Disclination: orientation reversal
            M = [0 1 0; -1 0 0; 0 0 1][1:algebra_dim, 1:algebra_dim]
            winding = 2
            berry = π
            
        elseif defect == :monopole
            # Magnetic monopole: non-abelian holonomy
            M = exp(im * π/2 * (randn(algebra_dim, algebra_dim) + randn(algebra_dim, algebra_dim)'))
            winding = 4
            berry = π/2
            
        else  # :frank (Frank defect in liquid crystals)
            M = Matrix(1.0I, algebra_dim, algebra_dim)
            for i in 1:algebra_dim-1
                M[i, i+1] = 0.5
                M[i+1, i] = -0.5
            end
            winding = 1
            berry = π/4
        end
        
        # Parallel transport along path γ: P exp(∫A)
        function parallel_transport(path::Vector{Matrix{Float64}})
            # Path-ordered exponential of connection A
            U = Matrix(1.0I, algebra_dim, algebra_dim)
            for (i, A) in enumerate(path)
                U *= exp(im * A)
            end
            return U
        end
        
        # Curvature F = dA + A∧A
        A = randn(algebra_dim, algebra_dim) + im*randn(algebra_dim, algebra_dim)
        curvature = A * A' - A' * A  # [A, A†] simplified
        
        new(M, defect, winding, berry, parallel_transport, curvature)
    end
end

function compute_monodromy(phase::TopologicalPhase, loop::Vector{Int})
    # Compute monodromy around a loop in the sheaf
    
    π₁ = phase.fundamental_group
    n = size(π₁, 1)
    
    # Product of group elements along loop
    M = Matrix(1.0I, n, n)
    for g in loop
        if 1 <= g <= n
            M = M * π₁  # Using same generator for simplicity
        end
    end
    
    return M
end

# ==================== PICARD GROUP & LINE BUNDLES ====================

struct PicardGroup
    # Pic(X) = H¹(X, O^*) ≅ group of line bundles
    rank::Int  # Picard number ρ
    generators::Vector{Matrix{ComplexF64}}  # Generator line bundles
    intersection_matrix::Matrix{Int}  # Cup product on H¹,¹
    
    # First Chern classes of generators
    c1_generators::Vector{Vector{Float64}}
    
    function PicardGroup(phase::TopologicalPhase)
        ρ = phase.picard_number
        
        # Generators as U(1) connections
        generators = []
        c1_generators = []
        
        for i in 1:ρ
            # Create a line bundle generator
            if i == 1
                # Trivial bundle
                L = Matrix(1.0I, 3, 3)
                c1 = zeros(3)
            else
                # Non-trivial bundle with curvature
                L = exp(im * randn(3, 3))
                c1 = randn(3)  # First Chern class
            end
            push!(generators, L)
            push!(c1_generators, c1)
        end
        
        # Intersection matrix (simplified)
        intersection = zeros(Int, ρ, ρ)
        for i in 1:ρ, j in 1:ρ
            if i == j
                intersection[i,j] = 1  # Self-intersection
            else
                intersection[i,j] = rand(-1:1)  # Random intersections
            end
        end
        
        new(ρ, generators, intersection, c1_generators)
    end
end

function compute_picard_group(phase::TopologicalPhase)
    return PicardGroup(phase)
end

# ==================== KODAIRA DIMENSION ====================

struct KodairaDimension
    # κ(X) = growth rate of plurigenera P_m = dim H⁰(X, K^⊗m)
    value::Float64  # -∞, 0, 1, 2, ...
    plurigenera::Dict{Int, Int}  # P_m for m = 1,2,3,...
    canonical_bundle::Matrix{ComplexF64}  # K_X
    
    function KodairaDimension(phase::TopologicalPhase)
        κ = phase.kodaira_dimension
        
        # Canonical bundle (determinant of cotangent bundle)
        if isinf(κ) && κ < 0
            # Fano: anti-canonical bundle ample
            K = exp(im * randn(3, 3))  # Random curvature
        elseif κ == 0
            # Calabi-Yau: K trivial
            K = Matrix(1.0I, 3, 3)
        else
            # General type: K ample
            K = exp(im * 2 * randn(3, 3))
        end
        
        # Plurigenera P_m = dim H⁰(X, K^⊗m)
        plurigenera = Dict{Int, Int}()
        for m in 1:3
            if isinf(κ) && κ < 0
                P_m = 0  # Fano: no holomorphic sections for m > 0
            elseif κ == 0
                P_m = 1  # Calabi-Yau: constant sections
            else
                P_m = Int(floor(m^κ))  # Growth ~ m^κ
            end
            plurigenera[m] = P_m
        end
        
        new(κ, plurigenera, K)
    end
end

function compute_kodaira_dimension(phase::TopologicalPhase)
    return KodairaDimension(phase)
end

# ==================== TOPOLOGY CHANGE DETECTION ====================

function detect_topology_change(phase1::TopologicalPhase, phase2::TopologicalPhase)
    # Detect if there's a topology change between phases
    
    changes = Dict{Symbol, Bool}()
    
    # 1. Check fundamental group
    π1_change = !is_similar_matrix(phase1.fundamental_group, phase2.fundamental_group, 0.1)
    changes[:π1] = π1_change
    
    # 2. Check homology
    homology_change = false
    if length(phase1.homology_groups) != length(phase2.homology_groups)
        homology_change = true
    else
        for k in 1:min(length(phase1.homology_groups), length(phase2.homology_groups))
            if !is_similar_matrix(phase1.homology_groups[k], phase2.homology_groups[k], 0.2)
                homology_change = true
                break
            end
        end
    end
    changes[:homology] = homology_change
    
    # 3. Check Chern classes
    chern_change = norm(phase1.chern_classes - phase2.chern_classes) > 0.5
    changes[:chern] = chern_change
    
    # 4. Check Picard number
    picard_change = phase1.picard_number != phase2.picard_number
    changes[:picard] = picard_change
    
    # 5. Check Kodaira dimension
    kodaira_change = abs(phase1.kodaira_dimension - phase2.kodaira_dimension) > 0.5
    changes[:kodaira] = kodaira_change
    
    # 6. Check Hochschild cohomology
    HH_change = false
    for k in 0:3
        if haskey(phase1.HH_structure, k) && haskey(phase2.HH_structure, k)
            HH1 = phase1.HH_structure[k]
            HH2 = phase2.HH_structure[k]
            if norm(HH1 - HH2) / (norm(HH1) + norm(HH2) + 1e-10) > 0.3
                HH_change = true
                break
            end
        end
    end
    changes[:HH] = HH_change
    
    # Overall topology change if any invariant changes
    overall_change = any(values(changes))
    
    return overall_change, changes
end

function is_similar_matrix(A, B, tolerance::Float64)
    # Check if matrices are similar up to conjugation
    if size(A) != size(B)
        return false
    end
    
    # Check eigenvalues
    λA = sort(eigvals(A))
    λB = sort(eigvals(B))
    
    return norm(λA - λB) < tolerance * max(norm(λA), norm(λB))
end

# ==================== SINGULARITY RESOLUTION ====================

function resolve_singularity_blowup(singular_phase::TopologicalPhase)
    # Resolve topological singularities via blowup
    
    println("Resolving singularities in $(singular_phase.phase_id) phase...")
    
    # Check what type of singularity we have
    singularity_type = classify_singularity(singular_phase)
    
    if singularity_type == :conifold
        # Conifold singularity: blow up to small resolution
        println("  Conifold singularity → small resolution")
        return conifold_resolution(singular_phase)
        
    elseif singularity_type == :orbifold
        # Orbifold singularity: blow up to smooth manifold
        println("  Orbifold singularity → smooth blowup")
        return orbifold_resolution(singular_phase)
        
    elseif singularity_type == :node
        # Nodal singularity: blow up to smooth
        println("  Nodal singularity → blowup")
        return nodal_resolution(singular_phase)
        
    else
        # Already smooth
        println("  Phase is already smooth")
        return singular_phase
    end
end

function classify_singularity(phase::TopologicalPhase)
    # Classify the type of topological singularity
    
    # Check Chern classes
    c1 = phase.chern_classes[1]
    
    # Check fundamental group
    π₁ = phase.fundamental_group
    
    # Check Kodaira dimension
    κ = phase.kodaira_dimension
    
    if norm(π₁ - Matrix(1.0I, size(π₁))) > 0.5 && abs(c1) > 0.8
        return :conifold
    elseif det(π₁) != 1 && κ == 0
        return :orbifold
    elseif rank(π₁) < size(π₁, 1)
        return :node
    else
        return :smooth
    end
end

function conifold_resolution(phase::TopologicalPhase)
    # Resolve conifold singularity: T*S³ → O(-1)⊕O(-1) → ℂP¹
    
    # New phase after resolution
    new_phase = deepcopy(phase)
    
    # Update fundamental group (simplifies)
    new_phase.fundamental_group = Matrix(1.0I, 2, 2)
    
    # Update homology
    new_phase.homology_groups = [Matrix(1I, 1, 1), zeros(2, 2), Matrix(1I, 1, 1)]
    
    # Update Chern classes
    new_phase.chern_classes = [2.0, 1.0, 0.0]  # ℂP¹ bundle
    
    # Update Kodaira dimension
    new_phase.kodaira_dimension = -Inf  # Fano
    
    # Update Picard number
    new_phase.picard_number = 1
    
    # Update Hochschild cohomology
    alg_dim = length(phase.HH_structure[0])
    new_phase.HH_structure[2] = zeros(ComplexF64, alg_dim, alg_dim, alg_dim, alg_dim)  # Smoother
    
    new_phase.phase_id = Symbol("$(phase.phase_id)_resolved")
    
    return new_phase
end

function orbifold_resolution(phase::TopologicalPhase)
    # Resolve orbifold singularity
    
    new_phase = deepcopy(phase)
    
    # Make fundamental group trivial
    new_phase.fundamental_group = Matrix(1.0I, size(phase.fundamental_group))
    
    # Increase Picard number (exceptional divisors added)
    new_phase.picard_number = phase.picard_number + 1
    
    # Chern classes change
    new_phase.chern_classes = phase.chern_classes .+ [0.5, 0.2, 0.0]
    
    new_phase.phase_id = Symbol("$(phase.phase_id)_resolved")
    
    return new_phase
end

function nodal_resolution(phase::TopologicalPhase)
    # Resolve nodal singularity
    
    new_phase = deepcopy(phase)
    
    # Restore full rank to fundamental group
    n = size(phase.fundamental_group, 1)
    new_phase.fundamental_group = Matrix(1.0I, n, n)
    
    # Smooth out Hochschild cohomology
    for k in 0:3
        if haskey(new_phase.HH_structure, k)
            HH = new_phase.HH_structure[k]
            # Reduce norm (smoother)
            new_phase.HH_structure[k] = HH * 0.7
        end
    end
    
    new_phase.phase_id = Symbol("$(phase.phase_id)_resolved")
    
    return new_phase
end

# ==================== SHEAF COHOMOLOGY CHANGE ====================

function sheaf_cohomology_change(sheaf1::Dict, sheaf2::Dict)
    # Compute change in sheaf cohomology Hⁱ(X, ℱ)
    
    changes = Dict{Int, Float64}()
    
    for i in 0:3
        if haskey(sheaf1, i) && haskey(sheaf2, i)
            # Compare cohomology groups
            H1 = sheaf1[i]
            H2 = sheaf2[i]
            
            if H1 isa Matrix && H2 isa Matrix
                # Compare as matrices
                change = norm(H1 - H2) / (norm(H1) + norm(H2) + 1e-10)
                changes[i] = change
            else
                # Compare dimensions
                dim1 = length(H1)
                dim2 = length(H2)
                changes[i] = abs(dim1 - dim2) / max(dim1, dim2)
            end
        end
    end
    
    return changes
end

# ==================== CHERN CLASS COMPUTATION ====================

function compute_chern_classes(phase::TopologicalPhase, curvature::Matrix{ComplexF64})
    # Compute Chern classes from curvature form
    
    n = size(curvature, 1)
    
    # Chern character: ch = Tr(exp(iF/2π))
    F = curvature
    I_F = im * F / (2π)
    
    # Compute exponential via Taylor series
    exp_F = Matrix(1.0I, n, n)
    term = Matrix(1.0I, n, n)
    
    for k in 1:5  # 5 terms enough for approximation
        term = term * I_F / k
        exp_F += term
    end
    
    # Chern character components
    ch = real(tr(exp_F))
    
    # Convert to Chern classes (simplified)
    c1 = real(tr(F)) / (2π)
    c2 = (real(tr(F*F)) - real(tr(F))^2) / (8π^2)
    c3 = real(det(F)) / ((2π)^3 * factorial(3))
    
    return [c1, c2, c3]
end

# ==================== TOPOLOGICAL PHASE DIAGRAM ====================

function topological_phase_diagram(phases::Vector{TopologicalPhase})
    # Create phase diagram with topology transitions
    
    n = length(phases)
    adjacency = zeros(Bool, n, n)
    transition_types = Dict{Tuple{Int,Int}, Symbol}()
    
    for i in 1:n, j in 1:n
        if i != j
            topology_change, changes = detect_topology_change(phases[i], phases[j])
            
            if topology_change
                adjacency[i,j] = true
                
                # Classify transition type
                if changes[:π1] && changes[:homology]
                    transition_types[(i,j)] = :topological
                elseif changes[:chern] && !changes[:π1]
                    transition_types[(i,j)] = :differentiable
                elseif changes[:HH] && !changes[:chern]
                    transition_types[(i,j)] = :algebraic
                else
                    transition_types[(i,j)] = :mixed
                end
            end
        end
    end
    
    return adjacency, transition_types
end

# ==================== TOPOLOGICAL CHARGE CONSERVATION ====================

function compute_topological_charge(phase::TopologicalPhase)
    # Compute topological charge (Chern number, monopole charge, etc.)
    
    charges = Dict{Symbol, Float64}()
    
    # 1. Chern number (first Chern class integrated)
    c1 = phase.chern_classes[1]
    charges[:chern] = c1
    
    # 2. Monopole charge from π₁
    π₁ = phase.fundamental_group
    if size(π₁, 1) > 1
        # Compute winding from determinant phase
        charges[:monopole] = angle(det(π₁)) / (2π)
    else
        charges[:monopole] = 0.0
    end
    
    # 3. Defect charge from Hochschild
    HH2 = phase.HH_structure[2]
    if !isempty(HH2)
        # Rough measure of non-associativity as topological charge
        charges[:associator] = norm(HH2)
    else
        charges[:associator] = 0.0
    end
    
    # 4. Kodaira-Spencer class (obstruction to deformation)
    κ = phase.kodaira_dimension
    charges[:kodaira_spencer] = isinf(κ) ? -1.0 : κ
    
    return charges
end

function check_topological_charge_conservation(phase1::TopologicalPhase, 
                                               phase2::TopologicalPhase)
    # Check if topological charges are conserved (modulo integers)
    
    charges1 = compute_topological_charge(phase1)
    charges2 = compute_topological_charge(phase2)
    
    conserved = Dict{Symbol, Bool}()
    differences = Dict{Symbol, Float64}()
    
    for key in keys(charges1)
        if haskey(charges2, key)
            diff = abs(charges1[key] - charges2[key])
            differences[key] = diff
            
            # Charge conservation modulo integers (for Chern numbers)
            if key == :chern
                conserved[key] = abs(diff - round(diff)) < 0.01
            else
                conserved[key] = diff < 0.1
            end
        end
    end
    
    all_conserved = all(values(conserved))
    
    return all_conserved, conserved, differences
end

end  # module TopologicalPhaseTransitions

# ==================== SIMULATION OF TOPOLOGY-CHANGING PHASE TRANSITIONS ====================

function simulate_topological_phase_transitions()
    println("="^60)
    println("TOPOLOGY-CHANGING PHASE TRANSITIONS")
    println("Hochschild cohomology as topological invariant")
    println("="^60)
    
    # Create the four topological phases
    phases = []
    phase_names = [:opiate, :critical, :transition, :norcain]
    
    for name in phase_names
        phase = TopologicalPhaseTransitions.TopologicalPhase(name, 3)
        push!(phases, phase)
        println("\nCreated $(name) phase:")
        println("  Fundamental group size: $(size(phase.fundamental_group))")
        println("  Chern classes: $(round.(phase.chern_classes, digits=3))")
        println("  Kodaira dimension: $(phase.kodaira_dimension)")
        println("  Picard number: $(phase.picard_number)")
    end
    
    # Analyze topology changes between phases
    println("\n" * "="^60)
    println("TOPOLOGY CHANGE ANALYSIS")
    println("="^60)
    
    topology_changes = []
    for i in 1:length(phases)
        for j in i+1:length(phases)
            change, details = TopologicalPhaseTransitions.detect_topology_change(phases[i], phases[j])
            
            if change
                println("\nTopology change detected: $(phase_names[i]) → $(phase_names[j])")
                for (invariant, changed) in details
                    if changed
                        println("  ✓ $(invariant) changed")
                    end
                end
                push!(topology_changes, (i, j, details))
            else
                println("\nNo topology change: $(phase_names[i]) → $(phase_names[j])")
            end
        end
    end
    
    # Check topological charge conservation
    println("\n" * "="^60)
    println("TOPOLOGICAL CHARGE CONSERVATION")
    println("="^60)
    
    for i in 1:length(phases)-1
        conserved, which, diffs = TopologicalPhaseTransitions.check_topological_charge_conservation(
            phases[i], phases[i+1]
        )
        
        println("\n$(phase_names[i]) → $(phase_names[i+1]):")
        if conserved
            println("  ✓ All topological charges conserved")
        else
            println("  ⚠ Some charges not conserved:")
            for (charge, cons) in which
                if !cons
                    println("    - $(charge): diff = $(round(diffs[charge], digits=3))")
                end
            end
        end
    end
    
    # Create phase diagram
    println("\n" * "="^60)
    println("TOPOLOGICAL PHASE DIAGRAM")
    println("="^60)
    
    adjacency, transition_types = TopologicalPhaseTransitions.topological_phase_diagram(phases)
    
    println("\nPhase transitions (with topology changes):")
    for i in 1:length(phases), j in 1:length(phases)
        if adjacency[i,j]
            println("  $(phase_names[i]) → $(phase_names[j]): $(transition_types[(i,j)])")
        end
    end
    
    # Simulate a topology-changing transition with resolution
    println("\n" * "="^60)
    println("SINGULARITY RESOLUTION SIMULATION")
    println("="^60)
    
    # Start with critical phase (has singularities)
    critical_phase = phases[findfirst(==(:critical), phase_names)]
    
    # Classify its singularity
    singularity = TopologicalPhaseTransitions.classify_singularity(critical_phase)
    println("Critical phase has $(singularity) singularity")
    
    # Resolve it
    resolved_phase = TopologicalPhaseTransitions.resolve_singularity_blowup(critical_phase)
    
    # Check if resolution worked
    change, _ = TopologicalPhaseTransitions.detect_topology_change(critical_phase, resolved_phase)
    if change
        println("Resolution changed topology!")
        println("New phase: $(resolved_phase.phase_id)")
        println("New Chern classes: $(round.(resolved_phase.chern_classes, digits=3))")
        println("New Picard number: $(resolved_phase.picard_number)")
    end
    
    # Compute monodromy for each phase
    println("\n" * "="^60)
    println("MONODROMY OPERATORS (TOPOLOGICAL DEFECTS)")
    println("="^60)
    
    for (i, phase) in enumerate(phases)
        # Create a loop in the sheaf
        loop = [1, 2, 3, 2, 1]  # Simple loop
        
        M = TopologicalPhaseTransitions.compute_monodromy(phase, loop)
        
        println("\n$(phase_names[i]) phase monodromy:")
        println("  Determinant: $(round(det(M), digits=3))")
        println("  Eigenvalues: $(round.(eigvals(M), digits=3))")
        
        # Check if monodromy is non-trivial
        if norm(M - Matrix(1.0I, size(M))) > 0.1
            println("  ⚠ Non-trivial monodromy (topological defect present)")
        else
            println("  ✓ Trivial monodromy (no defects)")
        end
    end
    
    # Compute Picard groups and Kodaira dimensions
    println("\n" * "="^60)
    println("COMPLEX GEOMETRY INVARIANTS")
    println("="^60)
    
    for (i, phase) in enumerate(phases)
        picard = TopologicalPhaseTransitions.compute_picard_group(phase)
        kodaira = TopologicalPhaseTransitions.compute_kodaira_dimension(phase)
        
        println("\n$(phase_names[i]) phase:")
        println("  Picard group rank: $(picard.rank)")
        println("  Intersection matrix: $(picard.intersection_matrix)")
        println("  Kodaira dimension: $(kodaira.value)")
        println("  Plurigenera P_m: $(kodaira.plurigenera)")
    end
    
    return phases, topology_changes
end

# Run simulation
if abspath(PROGRAM_FILE) == @__FILE__
    println("Simulating topology-changing phase transitions...")
    phases, changes = simulate_topological_phase_transitions()
    
    println("\n" * "="^60)
    println("MATHEMATICAL SUMMARY")
    println("="^60)
    println("Phase transitions are TOPOLOGY CHANGES when:")
    println("1. Fundamental group π₁ changes (homotopy type)")
    println("2. Homology groups Hₖ change")
    println("3. Chern classes cₖ change (differentiable structure)")
    println("4. Picard number ρ changes (complex structure)")
    println("5. Kodaira dimension κ changes (birationally)")
    println("6. Hochschild cohomology HH⁺ changes (algebraic structure)")
    println()
    println("Monodromy detects topological defects")
    println("Singularities resolved by blowups")
    println("Topological charges (mostly) conserved")
    println("="^60)
end

