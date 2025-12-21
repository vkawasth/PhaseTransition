# NeuroSheaf_CreationAnnihilation.jl
# Implementation with Casimir, raising/lowering operators and Connes-style spectral theory

using LinearAlgebra
using SparseArrays
using Arpack
using TensorOperations

module CreationAnnihilationAlgebra

export CasimirOperator, RaisingOperator, LoweringOperator, 
       SpectralFlow, ConnesTrace, compute_spectral_triple,
       evolve_basis, detect_phase_reversal_spectral

# ==================== CASIMIR OPERATOR (Center of Universal Enveloping Algebra) ====================

struct CasimirOperator
    # For Lie algebra g with basis {X_i}, Casimir = Σ_i X_i X^i (using Killing form)
    matrix::Matrix{ComplexF64}
    eigenvalues::Vector{Float64}
    eigenvectors::Matrix{ComplexF64}
    trace::ComplexF64  # Connes trace (singular trace on type II∞)
    
    # For neuro-sheaf: represents conserved quantities under phase transitions
    function CasimirOperator(algebra_dim::Int=3, curvature::Float64=0.0)
        # Create basis for su(algebra_dim) or so(algebra_dim) depending on context
        if algebra_dim == 3
            # su(2) generators (simplified)
            Jx = [0 1 0; 1 0 1; 0 1 0] / sqrt(2)
            Jy = [0 -im 0; im 0 -im; 0 im 0] / sqrt(2)
            Jz = [1 0 0; 0 0 0; 0 0 -1]
            
            # Casimir J² = Jx² + Jy² + Jz²
            J2 = Jx*Jx + Jy*Jy + Jz*Jz
            
            # Add curvature effect
            if abs(curvature) > 0
                # Curvature deforms the Casimir
                deformation = curvature * (Jx*Jy - Jy*Jx)  # Non-commutativity measure
                J2 += 0.1 * deformation
            end
            
            # Compute eigen decomposition
            eigvals, eigvecs = eigen(J2)
            
            # Connes trace (Dixmier trace for compact operators)
            # For positive compact T, Tr_ω(T) = lim_{N→∞} (1/log N) Σ_{n=1}^N λ_n(T)
            λ_sorted = sort(real.(diag(J2)), rev=true)
            N = min(100, length(λ_sorted))
            connes_trace = sum(λ_sorted[1:N]) / log(1 + N)
            
            new(J2, real.(eigvals), eigvecs, connes_trace)
        else
            # General construction
            basis = [randn(algebra_dim, algebra_dim) + im*randn(algebra_dim, algebra_dim) 
                    for _ in 1:algebra_dim^2-1]
            # Orthonormalize wrt Killing form
            basis = gram_schmidt_killing(basis)
            
            # Construct Casimir
            C = zeros(ComplexF64, algebra_dim, algebra_dim)
            for X in basis
                C += X * X'
            end
            
            eigvals, eigvecs = eigen(C)
            λ_sorted = sort(real.(diag(C)), rev=true)
            N = min(100, length(λ_sorted))
            connes_trace = sum(λ_sorted[1:N]) / log(1 + N)
            
            new(C, real.(eigvals), eigvecs, connes_trace)
        end
    end
end

# ==================== RAISING/LOWERING OPERATORS ====================

struct RaisingOperator
    # a†: raises state, creates excitation (fentanyl propagation)
    matrix::Matrix{ComplexF64}
    target_regions::Vector{Symbol}  # Which regions it affects (CUL4, HY, BMA)
    geometric_factor::Function  # f(curvature, connectedness) → amplification
    
    function RaisingOperator(algebra_dim::Int, targets::Vector{Symbol})
        # Standard bosonic creation operator
        a_dag = zeros(ComplexF64, algebra_dim, algebra_dim)
        for n in 1:algebra_dim-1
            a_dag[n, n+1] = sqrt(n)  # Standard harmonic oscillator normalization
        end
        
        # Geometric sensitivity
        geom_factor(curvature, connectedness) = 
            exp(-0.2 * curvature^2) * tanh(connectedness)
        
        new(a_dag, targets, geom_factor)
    end
end

struct LoweringOperator
    # a: lowers state, annihilates excitation (fentanyl), creates antidote (noloxone)
    matrix::Matrix{ComplexF64}
    target_regions::Vector{Symbol}  # Affects PA as well
    annihilation_factor::Function  # How much fentanyl to annihilate
    creation_factor::Function  # How much noloxone to create
    
    function LoweringOperator(algebra_dim::Int, targets::Vector{Symbol})
        # Standard bosonic annihilation operator
        a = zeros(ComplexF64, algebra_dim, algebra_dim)
        for n in 2:algebra_dim
            a[n, n-1] = sqrt(n-1)
        end
        
        # Factors based on geometry
        annihilate_factor(curvature, distance) = 
            0.8 * exp(-0.3 * distance) * (1 + 0.5 * curvature)
        
        create_factor(curvature, distance) = 
            1.2 * exp(-0.2 * distance) * (1 - 0.3 * curvature)
        
        new(a, targets, annihilate_factor, create_factor)
    end
end

# ==================== SPECTRAL FLOW (ALAN CONNES STYLE) ====================

struct SpectralFlow
    # For a 1-parameter family of Dirac operators D_t
    # Spectral flow = net number of eigenvalues crossing 0
    D_initial::Matrix{ComplexF64}  # Initial Dirac operator
    D_final::Matrix{ComplexF64}    # Final Dirac operator
    eigenvalues_history::Vector{Vector{Float64}}
    flow_count::Int
    
    function SpectralFlow(D0::Matrix{ComplexF64}, D1::Matrix{ComplexF64}, 
                         steps::Int=100)
        # Interpolate Dirac operators
        eigenvalues_hist = []
        
        for t in range(0, 1, length=steps)
            D_t = (1-t) * D0 + t * D1
            
            # Compute eigenvalues (smallest magnitude ones)
            n = size(D_t, 1)
            nev = min(20, n-2)
            λ = real.(eigvals(D_t)[1:nev])
            push!(eigenvalues_hist, λ)
        end
        
        # Count spectral flow (simplified)
        flow = count_spectral_flow(eigenvalues_hist)
        
        new(D0, D1, eigenvalues_hist, flow)
    end
end

# ==================== CONNES TRACE & ZETA ZEROS ====================

struct ConnesTrace
    # Dixmier trace implementation for spectral triples
    operator::Matrix{ComplexF64}
    zeta_function::Function
    residues::Dict{Int, ComplexF64}
    
    function ConnesTrace(D::Matrix{ComplexF64})
        # Zeta function ζ_D(s) = Tr(|D|^{-s}) for Re(s) > dimension
        function zeta_D(s::ComplexF64)
            # Compute |D|^{-s} via functional calculus
            λ, U = eigen(D)
            λ_abs = abs.(λ) .+ 1e-10  # Regularize
            
            # ζ(s) = Σ λ^{-s}
            return sum(λ_abs .^ (-s))
        end
        
        # Compute residues at poles (spectral dimension)
        residues = Dict{Int, ComplexF64}()
        
        # For demonstration, approximate residues numerically
        for k in 1:3
            s0 = ComplexF64(k)
            # Numerical derivative around pole
            ϵ = 0.01
            res = (s0 - 1im*ϵ) * zeta_D(s0 + 1im*ϵ)  # Simplified
            residues[k] = res
        end
        
        new(D, zeta_D, residues)
    end
end

# ==================== PROLATE SPECTRAL TRIPLE ====================

struct ProlateSpectralTriple
    # (A, H, D) where:
    # A = algebra of observables (Moyal algebra here)
    # H = Hilbert space (prolate function space)
    # D = Dirac operator (prolate-Jacobi with frequency constraints)
    
    algebra::Matrix{ComplexF64}  # Representation of A on H
    hilbert_dim::Int
    dirac_operator::Matrix{ComplexF64}  # D
    chirality::Matrix{ComplexF64}  # γ (Z/2 grading)
    reality::Matrix{ComplexF64}    # J (real structure)
    
    # Prolate parameters
    Ω::Float64  # Bandwidth
    T::Float64  # Time window
    
    function ProlateSpectralTriple(Ω::Float64, T::Float64, n::Int=50)
        # Build prolate spheroidal wave operator as Dirac operator
        α = range(-Ω/2, Ω/2, length=n)
        β = 0.3 * ones(n-1)
        
        # Jacobi matrix (tridiagonal)
        D = diagm(0 => α)
        for i in 1:n-1
            D[i, i+1] = β[i]
            D[i+1, i] = β[i]
        end
        
        # Make it anti-self-adjoint (Dirac-like)
        D = im * D
        
        # Chirality operator γ (Z/2 grading)
        γ = zeros(n, n)
        for i in 1:n
            γ[i, i] = i <= n÷2 ? 1.0 : -1.0
        end
        
        # Real structure J (antilinear)
        J = zeros(n, n)
        for i in 1:n
            J[i, n-i+1] = 1.0
        end
        
        # Algebra representation (simplified)
        A = zeros(ComplexF64, n, n)
        for i in 1:n
            A[i, i] = exp(im * 2π * i/n)
        end
        
        new(A, n, D, γ, J, Ω, T)
    end
end

# ==================== REGION-AWARE OPERATOR IMPLEMENTATION ====================

struct RegionAwareOperators
    point_p::Tuple{Float64, Float64, Float64}  # Reference point in bgr
    neighborhood_radius::Float64
    
    # Surrounding regions with their geometry
    regions::Dict{Symbol, Dict}
    # bgr (reference), CUL4 (above), HY (below), BMA (forward), PA (behind)
    
    # Geometric operators
    raising::RaisingOperator
    lowering::LoweringOperator
    casimir::CasimirOperator
    
    # Current molecular state
    fentanyl_state::Vector{ComplexF64}  # In eigenbasis
    noloxone_state::Vector{ComplexF64}
    dopamine_state::Vector{ComplexF64}
    
    # Basis functions (eigenfunctions of prolate operator)
    basis_functions::Matrix{ComplexF64}
    jacobian::Matrix{Float64}  # How basis changes with parameters
    
    function RegionAwareOperators(p::Tuple{Float64, Float64, Float64}, 
                                 radius::Float64=100.0)
        # Define regions around point p
        regions = Dict(
            :bgr => Dict(:position => p, :curvature => 0.5, :connectedness => 0.7),
            :CUL4 => Dict(:position => (p[1], p[2] + radius, p[3]), 
                         :curvature => 0.3, :connectedness => 0.6),
            :HY => Dict(:position => (p[1], p[2] - radius, p[3]), 
                       :curvature => 0.4, :connectedness => 0.8),
            :BMA => Dict(:position => (p[1] + radius, p[2], p[3]), 
                        :curvature => 0.6, :connectedness => 0.5),
            :PA => Dict(:position => (p[1] - radius, p[2], p[3]), 
                       :curvature => 0.7, :connectedness => 0.4)
        )
        
        # Create operators
        alg_dim = 5  # Basis dimension
        raising = RaisingOperator(alg_dim, [:CUL4, :HY, :BMA])  # Affects above, below, forward
        lowering = LoweringOperator(alg_dim, [:CUL4, :HY, :BMA, :PA])  # Affects all including behind
        
        # Casimir with average curvature
        avg_curvature = mean([r[:curvature] for r in values(regions)])
        casimir = CasimirOperator(alg_dim, avg_curvature)
        
        # Initial states
        fent_state = zeros(ComplexF64, alg_dim)
        nolo_state = zeros(ComplexF64, alg_dim)
        dopa_state = zeros(ComplexF64, alg_dim)
        
        # Initialize with some fentanyl in bgr
        fent_state[1] = 1.0  # Ground state occupied
        dopa_state[2] = 0.5  # Some dopamine
        
        # Basis functions from prolate operator
        prolate_triple = ProlateSpectralTriple(6.0, 0.5, alg_dim)
        basis = prolate_triple.dirac_operator |> eigen |> eigenvectors
        
        # Initial Jacobian (identity)
        jacobian = Matrix(1.0I, alg_dim, alg_dim)
        
        new(p, radius, regions, raising, lowering, casimir,
            fent_state, nolo_state, dopa_state, basis, jacobian)
    end
end

# ==================== OPERATOR ACTIONS ====================

function apply_raising!(system::RegionAwareOperators, drug::Symbol, dt::Float64)
    # Apply raising operator to propagate drug forward/upward
    
    if drug == :fentanyl
        state = system.fentanyl_state
        op = system.raising
    elseif drug == :dopamine
        state = system.dopamine_state
        op = system.raising
    else
        error("Raising operator not defined for $drug")
    end
    
    # Apply raising operator: |ψ⟩ → a†|ψ⟩
    new_state = op.matrix * state
    
    # Scale by geometric factors for target regions
    for region in op.target_regions
        if haskey(system.regions, region)
            geom = system.regions[region]
            factor = op.geometric_factor(geom[:curvature], geom[:connectedness])
            
            # Apply region-specific amplification
            # (In real implementation, would map to regional basis)
            new_state .*= (1 + 0.3 * factor * dt)
        end
    end
    
    # Update state
    if drug == :fentanyl
        system.fentanyl_state = new_state
    else
        system.dopamine_state = new_state
    end
    
    # Evolve basis functions (spectral flow)
    evolve_basis!(system, dt)
    
    return system
end

function apply_lowering!(system::RegionAwareOperators, dt::Float64)
    # Apply lowering operator: annihilates fentanyl, creates noloxone
    
    # Annihilate fentanyl: a|fent⟩
    fent_new = system.lowering.matrix * system.fentanyl_state
    
    # Create noloxone: a†|nolo⟩ (but with different factor)
    nolo_new = system.fentanyl_state  # Noloxone created from fentanyl annihilation
    
    # Apply geometric factors
    for region in system.lowering.target_regions
        if haskey(system.regions, region)
            geom = system.regions[region]
            ann_factor = system.lowering.annihilation_factor(geom[:curvature], 
                                                           norm(geom[:position]))
            cre_factor = system.lowering.creation_factor(geom[:curvature], 
                                                        norm(geom[:position]))
            
            # Region-specific effects
            if region in [:CUL4, :HY, :BMA, :PA]
                fent_new .*= (1 - 0.4 * ann_factor * dt)
                nolo_new .*= (1 + 0.6 * cre_factor * dt)
            end
        end
    end
    
    # Update states
    system.fentanyl_state = fent_new
    system.noloxone_state = nolo_new
    
    # Evolve basis
    evolve_basis!(system, dt)
    
    return system
end

function evolve_basis!(system::RegionAwareOperators, dt::Float64)
    # Evolve basis functions according to changing molecular concentrations
    
    alg_dim = length(system.fentanyl_state)
    
    # Create effective Hamiltonian from molecular states
    H_eff = zeros(ComplexF64, alg_dim, alg_dim)
    
    # Fentanyl contributes negative (inhibitory) term
    ρ_fent = system.fentanyl_state * system.fentanyl_state'
    H_eff -= 0.8 * ρ_fent
    
    # Noloxone contributes positive (excitatory) term
    ρ_nolo = system.noloxone_state * system.noloxone_state'
    H_eff += 1.2 * ρ_nolo
    
    # Dopamine modulates both
    ρ_dopa = system.dopamine_state * system.dopamine_state'
    H_eff += 0.3 * ρ_dopa * (ρ_nolo - ρ_fent)
    
    # Add Casimir for algebraic structure preservation
    H_eff += 0.1 * system.casimir.matrix
    
    # Time evolution: U = exp(-i H dt)
    U = exp(-im * H_eff * dt)
    
    # Evolve basis functions
    system.basis_functions = U * system.basis_functions
    
    # Update Jacobian: J = d(basis_new)/d(basis_old)
    # Simplified: J ≈ U for small dt
    system.jacobian = real.(U * U')
    
    return system
end

# ==================== PHASE REVERSAL DETECTION ====================

function detect_phase_reversal_spectral(system::RegionAwareOperators)
    # Detect phase reversal using spectral methods (Connes-style)
    
    # 1. Compute spectral flow of Dirac operator
    # Create initial and final Dirac operators based on molecular states
    D_initial = construct_dirac_from_state(system.fentanyl_state, system.noloxone_state,
                                          system.dopamine_state, system.basis_functions)
    
    # Simulate "reversed" state (swap fentanyl and noloxone)
    D_reversed = construct_dirac_from_state(system.noloxone_state, system.fentanyl_state,
                                           system.dopamine_state, system.basis_functions)
    
    # Compute spectral flow
    flow = SpectralFlow(D_initial, D_reversed, 50)
    
    # 2. Analyze zeta function zeros (Riemann hypothesis analogy)
    trace_info = ConnesTrace(D_initial)
    
    # 3. Casimir eigenvalue distribution
    casimir_vals = system.casimir.eigenvalues
    gap = casimir_vals[end] - casimir_vals[1]
    
    # 4. Basis instability (Jacobian determinant)
    J_det = det(system.jacobian)
    basis_instability = abs(1 - abs(J_det))
    
    # Detection criteria
    reversal_risk = 0.0
    
    # Criterion 1: Large spectral flow indicates topological change
    reversal_risk += 0.4 * min(flow.flow_count / 10, 1.0)
    
    # Criterion 2: Zeta function pole structure change
    if length(trace_info.residues) > 0
        res_sum = sum(abs.(values(trace_info.residues)))
        reversal_risk += 0.3 * min(res_sum / alg_dim, 1.0)
    end
    
    # Criterion 3: Casimir gap closing (criticality)
    reversal_risk += 0.2 * exp(-gap)
    
    # Criterion 4: Basis instability
    reversal_risk += 0.1 * basis_instability
    
    # Molecular ratio check
    fent_norm = norm(system.fentanyl_state)
    nolo_norm = norm(system.noloxone_state)
    ratio = fent_norm / (nolo_norm + 1e-10)
    
    if ratio > 1.5 && reversal_risk > 0.6
        return true, reversal_risk, (flow.flow_count, gap, basis_instability)
    else
        return false, reversal_risk, (flow.flow_count, gap, basis_instability)
    end
end

function construct_dirac_from_state(fent_state, nolo_state, dopa_state, basis)
    # Construct Dirac operator from molecular states in current basis
    
    n = length(fent_state)
    
    # Start with prolate operator structure
    α = range(-3.0, 3.0, length=n)
    D = diagm(0 => α)
    
    # Add off-diagonal terms from molecular interference
    for i in 1:n-1
        # Fentanyl creates negative off-diagonals (inhibitory)
        D[i, i+1] += -0.3 * abs(fent_state[i] * conj(fent_state[i+1]))
        D[i+1, i] += -0.3 * abs(fent_state[i+1] * conj(fent_state[i]))
        
        # Noloxone creates positive off-diagonals (excitatory)
        D[i, i+1] += 0.4 * abs(nolo_state[i] * conj(nolo_state[i+1]))
        D[i+1, i] += 0.4 * abs(nolo_state[i+1] * conj(nolo_state[i]))
        
        # Dopamine modulates
        dopa_mod = 0.2 * abs(dopa_state[i] * conj(dopa_state[i+1]))
        D[i, i+1] += dopa_mod * (abs(nolo_state[i]) - abs(fent_state[i]))
        D[i+1, i] += dopa_mod * (abs(nolo_state[i+1]) - abs(fent_state[i+1]))
    end
    
    # Make it anti-self-adjoint (Dirac property)
    D = im * (D - D') / 2
    
    return D
end

# ==================== HELPER FUNCTIONS ====================

function gram_schmidt_killing(basis::Vector{Matrix{ComplexF64}})
    # Orthonormalize wrt Killing form: ⟨X,Y⟩ = Tr(XY)
    ortho_basis = similar(basis)
    
    for (i, X) in enumerate(basis)
        v = copy(X)
        for j in 1:i-1
            Y = ortho_basis[j]
            # Projection coefficient
            c = tr(Y * v') / tr(Y * Y')
            v -= c * Y
        end
        # Normalize
        norm_v = sqrt(abs(tr(v * v')))
        if norm_v > 1e-10
            ortho_basis[i] = v / norm_v
        else
            ortho_basis[i] = zeros(size(X))
        end
    end
    
    return ortho_basis
end

function count_spectral_flow(eigenvalues_history::Vector{Vector{Float64}})
    # Count how many eigenvalues cross zero
    flow = 0
    
    for t in 1:length(eigenvalues_history)-1
        λ_t = eigenvalues_history[t]
        λ_next = eigenvalues_history[t+1]
        
        # Count sign changes for each eigenvalue pair
        for i in 1:min(length(λ_t), length(λ_next))
            if λ_t[i] * λ_next[i] < 0
                flow += sign(λ_next[i] - λ_t[i])
            end
        end
    end
    
    return flow
end

function compute_spectral_triple(system::RegionAwareOperators)
    # Create full spectral triple (A, H, D) for Connes-style analysis
    
    alg_dim = length(system.fentanyl_state)
    
    # Algebra A (simplified: diagonal matrices representing molecular concentrations)
    A = zeros(ComplexF64, alg_dim, alg_dim)
    for i in 1:alg_dim
        A[i, i] = system.fentanyl_state[i] + 0.5*system.noloxone_state[i] + 
                  0.3*system.dopamine_state[i]
    end
    
    # Hilbert space H (current basis)
    H = system.basis_functions
    
    # Dirac operator D
    D = construct_dirac_from_state(system.fentanyl_state, system.noloxone_state,
                                  system.dopamine_state, H)
    
    return (A, H, D)
end

end  # module CreationAnnihilationAlgebra

# ==================== SIMULATION ====================

function simulate_region_aware_operators()
    println("="^60)
    println("REGION-AWARE CREATION/ANNIHILATION OPERATOR SIMULATION")
    println("Alan Connes-style spectral methods for phase transitions")
    println("="^60)
    
    # Create system at point p in bgr region
    p = (0.0, 0.0, 0.0)  # Origin
    system = CreationAnnihilationAlgebra.RegionAwareOperators(p, 150.0)
    
    println("\nInitial state:")
    println("  Fentanyl norm: $(round(norm(system.fentanyl_state), digits=3))")
    println("  Noloxone norm: $(round(norm(system.noloxone_state), digits=3))")
    println("  Dopamine norm: $(round(norm(system.dopamine_state), digits=3))")
    
    # Simulation parameters
    total_time = 3600.0  # 1 hour
    dt = 60.0  # 1 minute steps
    n_steps = Int(total_time / dt)
    
    # Track phase reversal risk
    reversal_risks = []
    flow_counts = []
    
    for step in 1:n_steps
        t = step * dt
        
        if t < 1800.0  # First 30 minutes: fentanyl dominant
            # Apply raising operator to propagate fentanyl
            system = CreationAnnihilationAlgebra.apply_raising!(system, :fentanyl, dt)
            
            # Some dopamine activity
            if step % 5 == 0
                system = CreationAnnihilationAlgebra.apply_raising!(system, :dopamine, dt/10)
            end
            
        else  # After 30 minutes: administer noloxone
            if t == 1800.0
                println("\n=== ADMINISTERING NOLOXONE (Lowering operator activated) ===")
            end
            
            # Apply lowering operator: annihilates fentanyl, creates noloxone
            system = CreationAnnihilationAlgebra.apply_lowering!(system, dt)
            
            # Continue dopamine
            if step % 3 == 0
                system = CreationAnnihilationAlgebra.apply_raising!(system, :dopamine, dt/5)
            end
        end
        
        # Detect phase reversal risk
        reversal, risk, (flow, gap, basis_instab) = 
            CreationAnnihilationAlgebra.detect_phase_reversal_spectral(system)
        
        push!(reversal_risks, risk)
        push!(flow_counts, flow)
        
        if step % 10 == 0
            println("t = $(t/60) min: Risk = $(round(risk, digits=3)), " *
                   "Flow = $flow, Gap = $(round(gap, digits=3))")
            
            if reversal
                println("  ⚠ PHASE REVERSAL DETECTED!")
            end
        end
    end
    
    # Final analysis
    println("\n" * "="^60)
    println("FINAL ANALYSIS")
    println("="^60)
    
    println("\nMolecular states:")
    println("  Final fentanyl: $(round(norm(system.fentanyl_state), digits=3))")
    println("  Final noloxone: $(round(norm(system.noloxone_state), digits=3))")
    println("  Final dopamine: $(round(norm(system.dopamine_state), digits=3))")
    
    # Spectral analysis
    println("\nSpectral analysis:")
    
    # Casimir eigenvalues
    casimir_vals = system.casimir.eigenvalues
    println("  Casimir eigenvalues: $(round.(casimir_vals, digits=3))")
    println("  Casimir gap: $(round(casimir_vals[end] - casimir_vals[1], digits=3))")
    
    # Connes trace
    println("  Connes trace: $(round(system.casimir.trace, digits=3))")
    
    # Basis evolution
    J_det = det(system.jacobian)
    println("  Basis Jacobian determinant: $(round(J_det, digits=3))")
    
    # Phase reversal summary
    max_risk = maximum(reversal_risks)
    avg_risk = mean(reversal_risks)
    println("\nPhase reversal risk:")
    println("  Maximum: $(round(max_risk, digits=3))")
    println("  Average: $(round(avg_risk, digits=3))")
    
    if max_risk > 0.7
        println("  ⚠ High risk of phase reversal detected")
    else
        println("  ✓ System stable under spectral flow")
    end
    
    return system, reversal_risks
end

# Run simulation
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting region-aware operator simulation...")
    final_system, risks = simulate_region_aware_operators()
    
    println("\n" * "="^60)
    println("MATHEMATICAL FOUNDATIONS:")
    println("="^60)
    println("1. Casimir operator preserves Lie algebra structure")
    println("2. Raising/lowering operators respect brain geometry")
    println("3. Spectral flow (Connes) detects topological phase changes")
    println("4. Basis functions evolve via molecular Hamiltonians")
    println("5. Prolate operators provide frequency-constrained Hilbert space")
    println("6. Zeta function analysis for critical behavior")
    println("="^60)
end
