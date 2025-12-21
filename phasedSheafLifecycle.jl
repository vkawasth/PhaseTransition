struct Stalk
    id::Int
    region::Symbol           # :VTA, :NAc, :PFC, etc.
    
    # Local state (presymplectic vector space)
    p::Vector{Float64}       # 5D probability vector (C0)
    ω::Matrix{Float64}       # Presymplectic form (degenerate)
    
    # Hopf oscillator (local dynamics)
    z::ComplexF64           # Oscillator state
    μ::Float64              # Bifurcation parameter
    ω_natural::Float64      # Natural frequency
    band::Symbol            # :theta, :alpha, :beta, :gamma
    
    # MorAl algebra (Moyal deformation only when activated)
    ħ::Float64              # Deformation parameter (0 normally)
    A::Matrix{Float64}      # Algebra multiplication table
    ⋆_cache::Dict{Tuple{Vector{Float64}, Vector{Float64}}, Vector{Float64}}
    
    # Viral dormancy analogy
    dormant::Bool           # True = dormant (normal), False = active (opiate/dopamine)
    trigger_threshold::Float64
    activation_level::Float64
    
    # Local moral module (VTA/NAc/PFC specific)
    moral_constraint::Function
end

struct NeuroSheaf
    G::SimpleGraph           # Brain graph (3.5M nodes, 5.5M edges)
    stalks::Vector{Stalk}    # 3.5M stalks
    
    # Restriction maps (directed, time-dependent)
    # ρ[(i,j)] maps stalk i → stalk j along edge
    restrictions::Dict{Tuple{Int,Int}, Matrix{Float64}}
    
    # Semilocal projections (only when flooded)
    projections::Dict{Tuple{Int,Int}, Matrix{Float64}}
    
    # Sonnin tower structure
    towers::Vector{Vector{Int}}  # Partition of stalks into coarsening levels
    sonnin_parameters::Vector{Float64}  # p for each tower level
end

# SHEAFS DONE

function check_local_trigger!(stalk::Stalk, t::Float64, dt::Float64)
    # Analogous to virus checking environment to break dormancy
    
    # Trigger conditions (local only!):
    # 1. Opiate molecule binds (external input)
    # 2. Neighbor activation exceeds threshold (viral quorum sensing)
    # 3. Local metabolic state crosses threshold
    
    trigger_prob = 0.0
    
    # Condition 1: Direct opiate binding
    if has_opiate_at(stalk.id, t)
        trigger_prob += 0.8 * stalk.activation_level
    end
    
    # Condition 2: Local oscillator reaches critical amplitude
    if abs(stalk.z) > stalk.trigger_threshold
        trigger_prob += 0.5
    end
    
    # Condition 3: Probability vector entropy drops (order emerges)
    entropy = -sum(stalk.p .* log.(stalk.p .+ 1e-10))
    if entropy < 0.3
        trigger_prob += 0.3
    end
    
    # Stochastic triggering
    if rand() < trigger_prob * dt
        activate_stalk!(stalk)
        return true
    end
    
    return false
end

function activate_stalk!(stalk::Stalk)
    # BREAK DORMANCY - like virus starting replication
    
    stalk.dormant = false
    
    # 1. Activate Moyal algebra (ħ > 0)
    stalk.ħ = 0.1  # Non-commutative deformation activates
    
    # 2. Shift Hopf oscillator to supercritical
    stalk.μ = 0.5  # Above bifurcation (μ > 0)
    
    # 3. Release "viral payload" = dopamine
    stalk.activation_level = 1.0
    
    # 4. Change moral constraint (behavior change)
    if stalk.region == :VTA
        stalk.moral_constraint = vta_activated_constraint
    elseif stalk.region == :NAc
        stalk.moral_constraint = nac_activated_constraint
    end
    
    println("Stalk $(stalk.id) in $(stalk.region) ACTIVATED at $(time())")
end


function sheaf_time_step!(sheaf::NeuroSheaf, t::Float64, dt::Float64)
    # PHASE 1: Check local triggers (VIRAL SENSING)
    activated_stalks = Int[]
    
    @threads for i in 1:length(sheaf.stalks)
        stalk = sheaf.stalks[i]
        if check_local_trigger!(stalk, t, dt)
            push!(activated_stalks, i)
        end
    end
    
    # PHASE 2: Local dynamics (Hopf + Moyal)
    @threads for i in 1:length(sheaf.stalks)
        stalk = sheaf.stalks[i]
        
        # Hopf dynamics
        if !stalk.dormant
            # Activated: full dynamics with Moyal deformation
            dz = hopf_dynamics_moyal(stalk, sheaf, i, t)
        else
            # Dormant: simple dynamics
            dz = (stalk.μ + im*stalk.ω_natural - abs2(stalk.z)) * stalk.z
        end
        
        stalk.z += dt * dz
    end
    
    # PHASE 3: Propagate via restriction maps (ANATOMICAL SPREAD)
    @threads for e in edges(sheaf.G)
        i, j = src(e), dst(e)
        
        if haskey(sheaf.restrictions, (i,j))
            ρ = sheaf.restrictions[(i,j)]
            
            # Apply restriction (stalk i → stalk j)
            source_state = get_stalk_state(sheaf.stalks[i])
            transferred = ρ * source_state
            
            # Modify target stalk
            integrate_restriction!(sheaf.stalks[j], transferred, dt)
        end
    end
    
    # PHASE 4: Semilocal projections if flooded (FUNCTIONAL SPREAD)
    flooded_regions = find_flooded_regions(sheaf, activated_stalks)
    
    if !isempty(flooded_regions)
        @threads for (i,j) in keys(sheaf.projections)
            if i in flooded_regions && j in flooded_regions
                π = sheaf.projections[(i,j)]
                source_state = get_stalk_state(sheaf.stalks[i])
                projected = π * source_state
                
                integrate_projection!(sheaf.stalks[j], projected, dt)
            end
        end
    end
    
    return activated_stalks
end


function build_tower_from_activations(sheaf::NeuroSheaf, 
                                      activated_stalks::Vector{Int})
    # Tower emerges FROM LOCAL ACTIVATIONS, not predefined!
    
    # Level 0: All stalks (fine)
    # Level 1: Clusters of activated stalks
    # Level 2: Regions with significant activation
    # Level 3: Global patterns
    
    # Find connected components of activated stalks
    activated_graph = induced_subgraph(sheaf.G, activated_stalks)
    components = connected_components(activated_graph)
    
    # Build tower levels based on activation density
    tower = Vector{Vector{Int}}()
    
    # Level 0: Individual stalks (3.5M)
    push!(tower, collect(1:length(sheaf.stalks)))
    
    # Level 1: Activated clusters
    push!(tower, components)
    
    # Level 2: Merge clusters within same region
    regional_clusters = Dict{Symbol, Vector{Int}}()
    for comp in components
        region = sheaf.stalks[comp[1]].region
        push!(get!(regional_clusters, region, []), comp...)
    end
    push!(tower, collect(values(regional_clusters)))
    
    # Level 3: Whole brain (76 regions)
    region_groups = group_by_region(sheaf)
    push!(tower, region_groups)
    
    # Compute Sonnin parameters for each level
    sonnin_params = compute_sonnin_parameters(sheaf, tower)
    
    return tower, sonnin_params
end

function compute_sonnin_parameters(sheaf::NeuroSheaf, tower::Vector{Vector{Int}})
    # Sonnin parameter p measures approximation detail needed
    # p ≈ 1: Need fine detail (many active stalks)
    # p ≫ 1: Coarse approximation sufficient
    
    params = Float64[]
    
    for (level_idx, level_groups) in enumerate(tower)
        # For each group in this level
        group_errors = Float64[]
        
        for group in level_groups
            # Approximation error: how well group represents its members
            if level_idx < length(tower)  # Not the finest level
                # Compare to next finer level
                finer_groups = tower[level_idx+1]
                child_groups = find_children(group, finer_groups)
                
                # Error = variance within group
                states = [get_stalk_state(sheaf.stalks[i]) for i in group]
                error = mean(var.(states))
                push!(group_errors, error)
            end
        end
        
        # Sonnin parameter for this level
        if !isempty(group_errors)
            avg_error = mean(group_errors)
            # p ~ 1/error (more error = need more detail = lower p)
            p = clamp(2.0 / (avg_error + 0.1), 1.0, 10.0)
            push!(params, p)
        else
            push!(params, 2.0)  # Default
        end
    end
    
    return params
end

function compute_local_associator(stalk::Stalk)
    # C2 cochain: associator φ(a,b,c) = a ⋆ (b ⋆ c) - (a ⋆ b) ⋆ c
    # Only non-zero when Moyal algebra active (ħ > 0)
    
    if stalk.ħ ≈ 0
        return zeros(5, 5, 5)  # 5D algebra from p-vector
    end
    
    φ = zeros(5, 5, 5)
    
    # For basis vectors e_i, e_j, e_k
    for i in 1:5, j in 1:5, k in 1:5
        e_i = basis_vector(i, 5)
        e_j = basis_vector(j, 5)
        e_k = basis_vector(k, 5)
        
        # Moyal star product
        star_ij = moyal_product(e_i, e_j, stalk.ħ, stalk.A)
        star_jk = moyal_product(e_j, e_k, stalk.ħ, stalk.A)
        
        # Compute associator
        lhs = moyal_product(e_i, star_jk, stalk.ħ, stalk.A)
        rhs = moyal_product(star_ij, e_k, stalk.ħ, stalk.A)
        
        φ[i,j,k] = lhs - rhs
    end
    
    return φ
end

function propagate_associator_sheaf!(sheaf::NeuroSheaf, 
                                     activated_stalks::Vector{Int})
    # GV/BV coarsening: merge stalks with compatible associators
    
    # Group stalks by similar associator (up to coboundary)
    groups = group_by_cohomology_class(sheaf, activated_stalks)
    
    # Perform blow-up/blow-down
    for group in groups
        if should_blow_up(group, sheaf)
            blow_up_stalks!(sheaf, group)
        elseif should_blow_down(group, sheaf)
            blow_down_stalks!(sheaf, group)
        end
    end
    
    # Update restriction maps using surviving HH1 derivations
    update_restrictions_from_H1!(sheaf)
end

function group_by_cohomology_class(sheaf::NeuroSheaf, stalks::Vector{Int})
    # Group stalks whose associators differ by coboundary
    groups = Vector{Vector{Int}}()
    visited = falses(length(stalks))
    
    for (i, idx_i) in enumerate(stalks)
        if !visited[i]
            group = [idx_i]
            visited[i] = true
            
            φ_i = compute_local_associator(sheaf.stalks[idx_i])
            
            for (j, idx_j) in enumerate(stalks[i+1:end])
                if !visited[i+j]
                    φ_j = compute_local_associator(sheaf.stalks[idx_j])
                    
                    # Check if φ_j = φ_i + δη (coboundary)
                    if is_coboundary_difference(φ_i, φ_j)
                        push!(group, idx_j)
                        visited[i+j] = true
                    end
                end
            end
            
            push!(groups, group)
        end
    end
    
    return groups
end


# Hardy-Titchmarsh Transform
function propagate_dopamine_tower!(tower::NeuroSonninTower, t::Float64, dt::Float64)
    # Semilocal propagation: δ increases with dopamine
    for k in 1:length(tower.levels)
        level = tower.levels[k]
        
        # Semilocal Hardy-Titchmarsh transform
        δ = 1.0 + 2.0 * tower.dopamine_levels[k]  # Locality parameter
        
        # Apply to spectral measure
        for region_idx in 1:length(level.region_partition)
            # Transform spectral measure
            old_μ = level.spectral_measure[region_idx]
            new_μ = apply_semilocal_transform(old_μ, δ)
            
            # Compute Radon-Nikodym derivative
            rn_derivative = new_μ ./ (old_μ .+ 1e-10)
            
            # Check boundedness condition (both dm_S/dm and dm/dm_S bounded)
            lower_bound = minimum(rn_derivative)
            upper_bound = maximum(rn_derivative)
            
            if 0.1 < lower_bound < upper_bound < 10.0
                level.spectral_measure[region_idx] = new_μ
            else
                # Phase transition detected!
                log_transition(t, k, region_idx)
            end
        end
    end
end

function apply_semilocal_transform(μ::Vector{Float64}, δ::Float64)
    # Hardy-Titchmarsh kernel
    N = length(μ)
    transformed = zeros(N)
    
    for i in 1:N
        total = 0.0
        for j in 1:N
            # Semilocal kernel K_δ(i,j)
            r = abs(i-j)/N
            kernel = exp(-r^2/(2δ^2)) * (r ≤ 2δ ? 1.0 : 0.0)
            total += kernel * μ[j]
        end
        transformed[i] = total
    end
    
    return transformed
end

# HOPF with Moyal Algebra
struct HopfOscillator
    # Phase space coordinates (complex amplitude)
    z::ComplexF64
    z̄::ComplexF64
    
    # Hopf parameters
    μ::Float64      # Bifurcation parameter
    ω::Float64      # Natural frequency
    β::Float64      # Nonlinear saturation
    
    # Moyal algebra parameters
    ħ::Float64                         # Deformation parameter
    use_moyal::Bool                    # Active in drug regions
    star_cache::Dict{Tuple{Int,Int}, ComplexF64}  # Precomputed ⋆ products
end

function moyal_product(f::Function, g::Function, ħ::Float64)
    # Moyal star product: f ⋆ g = fg + (iħ/2){f,g} + O(ħ²)
    # For polynomial functions
    return (x,p) -> begin
        f_val = f(x,p)
        g_val = g(x,p)
        
        # Poisson bracket {f,g} = ∂f/∂x ∂g/∂p - ∂f/∂p ∂g/∂x
        ∇f = ForwardDiff.gradient(f, [x,p])
        ∇g = ForwardDiff.gradient(g, [x,p])
        poisson = ∇f[1]*∇g[2] - ∇f[2]*∇g[1]
        
        return f_val*g_val + (im*ħ/2)*poisson
    end
end

function hopf_dynamics_moyal!(osc::HopfOscillator, coupling::ComplexF64, dt::Float64)
    # Hamiltonian formulation
    H(z, z̄) = ω*abs2(z) + (μ/2)*abs2(z) - (β/4)*abs4(z)
    
    if osc.use_moyal
        # Moyal bracket dynamics: ∂W/∂t = {{H, W}}_⋆
        # For small ħ: reduces to Poisson + O(ħ²)
        {H, W} = 2im*(∂H∂z̄*∂W∂z - ∂H∂z*∂W∂z̄)  # Poisson bracket
        
        # Moyal correction
        ⋆_correction = (ħ^2/24)*(∂³H∂z³*∂³W∂z̄³ - ∂³H∂z̄³*∂³W∂z³)
        
        dzdt = {H, z} + ħ^2 * ⋆_correction
    else
        # Classical Hopf normal form
        dzdt = (osc.μ + im*osc.ω - osc.β*abs2(osc.z))*osc.z
    end
    
    osc.z += dt * (dzdt + coupling)
end
using LinearAlgebra, SpecialFunctions, Polynomials

struct SonninSpace
    p::Float64                # Sonnin parameter 1 ≤ p ≤ ∞
    max_degree::Int           # Maximum polynomial degree
    basis::Vector{Polynomial} # Orthogonal polynomial basis
    measure::Function         # dμ(ω) spectral measure
end

function approximation_error(f::Function, 
                             space::SonninSpace, 
                             degree::Int)
    # Project f onto polynomials up to given degree
    coeffs = zeros(degree+1)
    
    for k in 0:degree
        # Compute coefficient via inner product
        integrand(ω) = f(ω) * space.basis[k+1](ω) * space.measure(ω)
        coeffs[k+1], _ = quadgk(integrand, ω_min, ω_max)
    end
    
    # Construct approximation
    approx(ω) = sum(coeffs[k+1] * space.basis[k+1](ω) for k in 0:degree)
    
    # Compute L² error
    error_func(ω) = abs2(f(ω) - approx(ω)) * space.measure(ω)
    E, _ = quadgk(error_func, ω_min, ω_max)
    
    return sqrt(E)
end

function sonnin_norm(f::Function, space::SonninSpace)
    # Compute ∑ n^(p-1) E_n(f)^p
    total = 0.0
    
    for n in 1:space.max_degree
        E_n = approximation_error(f, space, n)
        
        if space.p < Inf
            total += (n^(space.p-1)) * (E_n^space.p)
        else
            # For p = ∞, norm is sup_n E_n
            total = max(total, E_n)
        end
    end
    
    if space.p < Inf
        return total^(1/space.p)
    else
        return total
    end
end

function dopamine_sonnin_transition(f::Function, 
                                    D_values::Vector{Float64})
    # Track how Sonnin parameter changes with dopamine
    
    results = []
    
    for D in D_values
        # Metaplectic transformation of f
        f_D = apply_metaplectic(f, D)
        
        # Find optimal Sonnin parameter for f_D
        p_opt = find_optimal_sonnin_parameter(f_D)
        
        # Compute norm in optimal space
        space = SonninSpace(p_opt, 20, legendre_basis, gaussian_measure)
        norm_val = sonnin_norm(f_D, space)
        
        push!(results, (D, p_opt, norm_val))
    end
    
    return results
end

function find_optimal_sonnin_parameter(f::Function)
    # Minimize norm + complexity cost
    best_p = 2.0  # Start with Hilbert space
    best_cost = Inf
    
    for p in 1:0.1:10
        space = SonninSpace(p, 20, legendre_basis, gaussian_measure)
        norm_val = sonnin_norm(f, space)
        
        # Complexity cost: higher p = finer approximation needed
        complexity_cost = exp(p-1)  # Exponential cost for fine detail
        
        total_cost = norm_val + 0.1 * complexity_cost
        
        if total_cost < best_cost
            best_cost = total_cost
            best_p = p
        end
    end
    
    return best_p
end

using LinearAlgebra, QuadGK

struct BoundedRNDerivative
    original_measure::Function  # μ(A) = ∫_A dμ
    deformed_measure::Function  # μ_D(A) = ∫_A dμ_D
    density::Function          # f(ω) = dμ_D/dμ(ω)
    c::Float64                 # Lower bound: f(ω) ≥ c
    C::Float64                 # Upper bound: f(ω) ≤ C
end

function compute_rn_derivative(original_weights::Vector{Float64},
                               deformed_weights::Vector{Float64},
                               eigenvalues::Vector{Float64})
    # For discrete spectrum
    n = length(original_weights)
    density_vals = zeros(n)
    
    for i in 1:n
        if original_weights[i] > 0
            density_vals[i] = deformed_weights[i] / original_weights[i]
        else
            density_vals[i] = 1.0  # Convention
        end
    end
    
    c = minimum(density_vals)
    C = maximum(density_vals)
    
    # Interpolating function for continuous representation
    density_func = ω -> begin
        # Find nearest eigenvalue
        idx = argmin(abs.(eigenvalues .- ω))
        return density_vals[idx]
    end
    
    return BoundedRNDerivative(
        A -> sum(original_weights[eigenvalues .∈ [A]]),
        A -> sum(deformed_weights[eigenvalues .∈ [A]]),
        density_func,
        c, C
    )
end

function check_phase_transition(rn::BoundedRNDerivative, 
                                threshold_low::Float64=0.01,
                                threshold_high::Float64=100.0)
    # Phase transition when bounds become extreme
    if rn.c < threshold_low || rn.C > threshold_high
        return true, (rn.c, rn.C)
    else
        return false, (rn.c, rn.C)
    end
end

function metaplectic_to_rn(M::Matrix{ComplexF64},  # Metaplectic transformation
                           J::Matrix{Float64})     # Jacobi matrix
    # Diagonalize Jacobi matrix
    λ, V = eigen(J)
    n = size(J, 1)
    
    # Original weights: squared eigenvector components
    w_original = abs2.(V[1, :])  # Assuming p₀ = [1,0,0,...]ᵀ
    
    # Deformed weights under M
    p0 = zeros(ComplexF64, n)
    p0[1] = 1.0
    p0_deformed = M * p0
    
    # Project onto eigenvectors
    w_deformed = zeros(Float64, n)
    for i in 1:n
        w_deformed[i] = abs(dot(V[:, i], p0_deformed))^2
    end
    
    return compute_rn_derivative(w_original, w_deformed, λ)
end








function reduce_waves_locally!(sheaf::NeuroSheaf, 
                               subsiding_regions::Vector{Symbol})
    # CRITICAL: Waves don't interact, just get reduced locally
    
    @threads for stalk in sheaf.stalks
        if stalk.region in subsiding_regions
            # Exponential decay of oscillator
            decay = exp(-0.1 * dt)  # Time constant 10ms
            stalk.z *= decay
            
            # Reduce probability vector entropy (become more deterministic)
            if entropy(stalk.p) > 0.1
                # Move toward basis vector
                max_idx = argmax(stalk.p)
                stalk.p .= 0.1 * stalk.p
                stalk.p[max_idx] = 0.9
            end
            
            # DO NOT transfer energy to other stalks!
            # This is not conservation of energy, it's dissipation
        end
    end
    
    # Update observable waves at region level
    for region in subsiding_regions
        region_stalks = [s for s in sheaf.stalks if s.region == region]
        region_power = sum(abs2.(getfield.(region_stalks, :z)))
        
        # Store for visualization
        record_wave_power(region, :theta, region_power)  # Simplified
    end
end

function simulate_viral_triggering(total_time=10.0, dt=0.01)
    # Initialize sheaf with 3.5M stalks
    sheaf = initialize_neuro_sheaf(
        num_stalks=3_500_000,
        num_edges=5_500_000,
        regions=76
    )
    
    # All stalks start DORMANT (like latent virus)
    for stalk in sheaf.stalks
        stalk.dormant = true
        stalk.ħ = 0.0  # Commutative algebra
        stalk.μ = -0.1  # Subcritical (stable fixed point)
    end
    
    # Simulation loop
    wave_history = Dict{Symbol, Vector{Float64}}()
    activation_history = Vector{Int}[]
    
    for t in 0:dt:total_time
        # 1. Introduce opiate at specific locations (VTA, NAc)
        if t ≈ 2.0
            introduce_opiate!(sheaf, [:VTA, :NAc])
        end
        
        # 2. Local triggering (like virus sensing environment)
        activated = sheaf_time_step!(sheaf, t, dt)
        push!(activation_history, activated)
        
        # 3. Build tower FROM activations (emergent)
        tower, sonnin_params = build_tower_from_activations(sheaf, activated)
        sheaf.towers = tower
        sheaf.sonnin_parameters = sonnin_params
        
        # 4. Hochschild coarsening
        propagate_associator_sheaf!(sheaf, activated)
        
        # 5. Compute observable waves (coarse level only)
        if !isempty(tower) && length(tower) >= 3
            coarse_groups = tower[end]  # Coarsest level (regions)
            waves = compute_coarse_waves(sheaf, coarse_groups)
            
            for (region, wave_powers) in waves
                wave_history[region] = wave_powers
            end
        end
        
        # 6. Check for phase transitions (executive function)
        if check_phase_transition(sheaf, t)
            # Trigger norepinephrine response (reversal)
            apply_norepinephrine!(sheaf)
            
            # Waves reduce WITHOUT interaction
            reduce_waves_locally!(sheaf, [:PFC, :NAc])  # Executive regions subside
        end
    end
    
    return sheaf, wave_history, activation_history
end

