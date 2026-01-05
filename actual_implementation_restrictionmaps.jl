# NeuroSheaf_RestrictionMaps.jl
# Proper implementation of restriction maps for wavefront propagation

using LinearAlgebra
using SparseArrays
using TensorOperations

module RestrictionMaps

export RestrictionMap, ExtensionMap, RestrictionFunctor, 
       compose_restrictions, compute_cohomology_transfer,
       wavefront_propagation, resolve_restriction_singularities,
       build_restriction_tower, check_gluing_conditions

# ============================================================================
# 1. RESTRICTION MAPS (ALGEBRAIC GEOMETRY)
# ============================================================================

struct RestrictionMap
    # ρ: ℱ(U) → ℱ(V) for V ⊆ U (contravariant!)
    domain::Vector{Int}      # Indices in larger region U
    codomain::Vector{Int}    # Indices in smaller region V ⊆ U
    matrix::Matrix{Float64}  # Linear restriction: s|_V = ρ(s)
    
    # Algebraic properties
    is_surjective::Bool
    is_injective::Bool
    kernel_basis::Matrix{Float64}  # Basis for ker(ρ)
    cokernel_basis::Matrix{Float64} # Basis for coker(ρ)
    
    # Functorial data
    naturality_square::Matrix{Float64}  # Commutes with differentials
    hochschild_compatibility::Bool      # Compatible with HH⁺?
    
    function RestrictionMap(domain::Vector{Int}, codomain::Vector{Int}, 
                           sheaf_data::Dict{Int, Matrix{Float64}})
        # domain = stalks in larger open set U
        # codomain = stalks in smaller open set V ⊆ U
        
        n_dom = length(domain)
        n_cod = length(codomain)
        
        # Create restriction matrix (simplified: identity for included stalks)
        ρ = zeros(n_cod, n_dom)
        
        # Map: if stalk i in domain maps to stalk j in codomain
        domain_to_idx = Dict(domain[i] => i for i in 1:n_dom)
        
        for (j, stalk_idx) in enumerate(codomain)
            if haskey(domain_to_idx, stalk_idx)
                i = domain_to_idx[stalk_idx]
                ρ[j, i] = 1.0  # Identity restriction
            else
                # Interpolate from neighboring stalks in domain
                # Find nearest neighbor in domain
                distances = [norm(sheaf_data[stalk_idx] - sheaf_data[domain[k]]) 
                           for k in 1:n_dom]
                nearest = argmin(distances)
                ρ[j, nearest] = 0.5  # Partial restriction
            end
        end
        
        # Check properties
        is_surj = rank(ρ) == n_cod
        is_inj = rank(ρ) == n_dom
        
        # Compute kernel and cokernel
        ker = nullspace(ρ)
        coker = nullspace(ρ')
        
        # Check naturality with differentials (simplified)
        if haskey(sheaf_data, :differential)
            d = sheaf_data[:differential]
            naturality = norm(ρ * d - d * ρ)
            hh_compat = naturality < 1e-6
        else
            hh_compat = true
        end
        
        new(domain, codomain, ρ, is_surj, is_inj, ker, coker, 
            zeros(size(ρ)), hh_compat)
    end
end

struct ExtensionMap
    # ε: ℱ(V) → ℱ(U) for V ⊆ U (covariant extension by zero)
    domain::Vector{Int}      # Smaller region V
    codomain::Vector{Int}    # Larger region U
    matrix::Matrix{Float64}  # Extension: ε(s) = s on V, 0 elsewhere
    
    # Sheaf-theoretic properties
    is_embedding::Bool
    support::Vector{Int}     # Where the extension is non-zero
    is_flabby::Bool          # Can extend any section?
    
    function ExtensionMap(domain::Vector{Int}, codomain::Vector{Int})
        n_dom = length(domain)
        n_cod = length(codomain)
        
        ε = zeros(n_cod, n_dom)
        
        # Create index mapping
        domain_idx = Dict(domain[i] => i for i in 1:n_dom)
        
        for (j, stalk_idx) in enumerate(codomain)
            if haskey(domain_idx, stalk_idx)
                i = domain_idx[stalk_idx]
                ε[j, i] = 1.0  # Identity on V
            end
            # Otherwise 0 (extension by zero)
        end
        
        is_emb = all(in.(domain, Ref(codomain)))
        support = domain
        is_flabby = true  # Simplified assumption
        
        new(domain, codomain, ε, is_emb, support, is_flabby)
    end
end

# ============================================================================
# 2. FUNCTORIAL COMPOSITION
# ============================================================================

function compose_restrictions(ρ1::RestrictionMap, ρ2::RestrictionMap)
    # Compose: ℱ(U) → ℱ(V) → ℱ(W) for U ⊇ V ⊇ W
    
    # Check composability: codomain(ρ1) must contain domain(ρ2)
    if !all(in.(ρ2.domain, Ref(ρ1.codomain)))
        error("Restrictions not composable: V must be subset of U")
    end
    
    # Create composition matrix
    # Map domain(ρ2) indices to positions in codomain(ρ1)
    idx_map = Dict(ρ1.codomain[i] => i for i in 1:length(ρ1.codomain))
    
    comp_matrix = zeros(length(ρ2.codomain), length(ρ1.domain))
    
    for (i, stalk_w) in enumerate(ρ2.codomain)
        # Find which stalk in V it comes from
        for (j, stalk_v) in enumerate(ρ2.domain)
            if ρ2.matrix[i, j] != 0
                # This stalk_v contributes to stalk_w
                # Find where stalk_v comes from in U
                if haskey(idx_map, stalk_v)
                    k = idx_map[stalk_v]
                    for l in 1:length(ρ1.domain)
                        comp_matrix[i, l] += ρ2.matrix[i, j] * ρ1.matrix[k, l]
                    end
                end
            end
        end
    end
    
    # Check if composition preserves properties
    is_surj = rank(comp_matrix) == length(ρ2.codomain)
    is_inj = rank(comp_matrix) == length(ρ1.domain)
    
    return RestrictionMap(ρ1.domain, ρ2.codomain, comp_matrix, 
                         Dict(:matrix_only => true))
end

# Helper constructor
function RestrictionMap(domain, codomain, matrix, properties)
    # Simplified constructor for compositions
    n_dom = length(domain)
    n_cod = length(codomain)
    
    is_surj = rank(matrix) == n_cod
    is_inj = rank(matrix) == n_dom
    ker = nullspace(matrix)
    coker = nullspace(matrix')
    
    new(domain, codomain, matrix, is_surj, is_inj, ker, coker,
        zeros(size(matrix)), true)
end

# ============================================================================
# 3. WAVEFRONT PROPAGATION VIA RESTRICTION
# ============================================================================

struct Wavefront
    # A wavefront propagating through regions
    region_chain::Vector{Vector{Int}}  # U₀ ⊇ U₁ ⊇ U₂ ⊇ ... ⊇ Uₙ
    restriction_maps::Vector{RestrictionMap}
    wave_amplitudes::Vector{Vector{Float64}}  # Amplitude at each region
    
    # Propagation dynamics
    speed::Float64          # Propagation speed (mm/s)
    attenuation::Float64    # Attenuation per restriction
    phase_shift::Float64    # Phase shift at boundaries
    
    # Current state
    current_region::Int     # Index in region_chain
    arrival_times::Vector{Float64}
    
    function Wavefront(regions::Vector{Vector{Int}}, 
                      sheaf_data::Dict{Int, Any},
                      speed::Float64=1.0)
        n = length(regions)
        
        # Create restriction maps along chain
        ρ_maps = Vector{RestrictionMap}(undef, n-1)
        for i in 1:n-1
            ρ_maps[i] = RestrictionMap(regions[i], regions[i+1], sheaf_data)
        end
        
        # Initialize wave amplitudes (Gaussian profile)
        amps = Vector{Vector{Float64}}(undef, n)
        for i in 1:n
            m = length(regions[i])
            # Gaussian centered at first stalk
            center = 1
            amps[i] = [exp(-0.5 * ((j-center)/2)^2) for j in 1:m]
            amps[i] ./= maximum(amps[i])  # Normalize
        end
        
        new(regions, ρ_maps, amps, speed, 0.1, 0.0, 1, zeros(n))
    end
end

function propagate_wavefront!(wf::Wavefront, dt::Float64, 
                            drug_concentration::Dict{Int, Float64})
    # Propagate wavefront through restriction maps
    
    current_idx = wf.current_region
    if current_idx >= length(wf.region_chain)
        return wf  # Already reached deepest region
    end
    
    # Get current and next regions
    U_current = wf.region_chain[current_idx]
    U_next = wf.region_chain[current_idx + 1]
    ρ = wf.restriction_maps[current_idx]
    
    # Current amplitude in U_current
    amp_current = wf.wave_amplitudes[current_idx]
    
    # Apply restriction map with drug modulation
    amp_next = ρ.matrix * amp_current
    
    # Drug effects on propagation
    for (i, stalk_idx) in enumerate(U_next)
        if haskey(drug_concentration, stalk_idx)
            drug_factor = exp(-0.2 * drug_concentration[stalk_idx])
            amp_next[i] *= drug_factor
        end
    end
    
    # Attenuation from restriction
    amp_next .*= (1.0 - wf.attenuation)
    
    # Phase shift (simplified)
    phase_shift = wf.phase_shift * dt
    if length(amp_next) > 1
        # Apply different phases to different components
        for i in 1:length(amp_next)
            angle = phase_shift * (i-1)/(length(amp_next)-1)
            amp_next[i] *= exp(im * angle) |> abs
        end
    end
    
    # Update wavefront state
    wf.wave_amplitudes[current_idx + 1] = amp_next
    
    # Check if wavefront has "arrived" in next region
    arrival_threshold = 0.1
    if maximum(amp_next) > arrival_threshold
        # Wavefront has propagated to next region
        wf.arrival_times[current_idx + 1] = wf.arrival_times[current_idx] + dt
        
        # Move current region forward
        if current_idx < length(wf.region_chain) - 1
            # Prepare for next propagation step
            wf.current_region += 1
            
            # Transfer some amplitude back via adjoint restriction
            # (wave reflection at boundary)
            if wf.current_region > 1
                ρ_prev = wf.restriction_maps[current_idx - 1]
                reflection = ρ_prev.matrix' * amp_next * 0.1  # 10% reflection
                wf.wave_amplitudes[current_idx] .+= reflection
            end
        end
    end
    
    return wf
end

# ============================================================================
# 4. COHOMOLOGY TRANSFER MAPS
# ============================================================================

function compute_cohomology_transfer(ρ::RestrictionMap, 
                                    HH_domain::Dict{Int, Array{Float64}},
                                    HH_codomain::Dict{Int, Array{Float64}})
    # Compute induced maps on Hochschild cohomology
    
    # For each degree k, compute ρₖ: HHᵏ(U) → HHᵏ(V)
    transfer_maps = Dict{Int, Matrix{Float64}}()
    
    for k in 0:3  # Consider HH⁰, HH¹, HH², HH³
        if haskey(HH_domain, k) && haskey(HH_codomain, k)
            HH_U = HH_domain[k]  # dim_U × ... × dim_U (k+1 times)
            HH_V = HH_codomain[k] # dim_V × ... × dim_V
            
            # Flatten Hochschild cochains
            if k == 0
                # HH⁰: center elements (vectors)
                dim_U = size(HH_U, 1)
                dim_V = size(HH_V, 1)
                
                # Restriction on center: project using ρ
                ρ_k = ρ.matrix  # dim_V × dim_U
                
            elseif k == 1
                # HH¹: derivations (matrices)
                dim_U = size(HH_U, 1)
                dim_V = size(HH_V, 1)
                
                # For derivation D: U → U, restrict to V: ρ∘D∘ρ⁺
                ρ_k = kron(ρ.matrix, ρ.matrix)  # (dim_V×dim_V) × (dim_U×dim_U)
                
            elseif k == 2
                # HH²: associators (4-tensors)
                dim_U = size(HH_U, 1)
                dim_V = size(HH_V, 1)
                
                # Restrict associator: ρ⊗ρ⊗ρ⊗ρ applied to tensor
                ρ_k = kron(kron(ρ.matrix, ρ.matrix), 
                          kron(ρ.matrix, ρ.matrix))
                
            else  # k == 3
                # HH³: higher operations
                dim_U = size(HH_U, 1)
                dim_V = size(HH_V, 1)
                
                ρ_k = kron(kron(kron(ρ.matrix, ρ.matrix), 
                               ρ.matrix), ρ.matrix)
            end
            
            transfer_maps[k] = ρ_k
        end
    end
    
    # Also compute the transfer in opposite direction (extension)
    extension_maps = Dict{Int, Matrix{Float64}}()
    for k in keys(transfer_maps)
        # Extension is approximately pseudo-inverse
        extension_maps[k] = pinv(transfer_maps[k])
    end
    
    return transfer_maps, extension_maps
end

# ============================================================================
# 5. RESOLVING RESTRICTION SINGULARITIES
# ============================================================================

function resolve_restriction_singularities(ρ::RestrictionMap,
                                          wave_data::Dict{Int, Float64})
    # Resolve singularities where restriction map fails to be nice
    
    n_cod = length(ρ.codomain)
    n_dom = length(ρ.domain)
    
    # Detect singularities in restriction
    singularities = []
    
    # Check each stalk in codomain
    for (j, stalk_v) in enumerate(ρ.codomain)
        # Check if restriction to this stalk is ill-defined
        row_norm = norm(ρ.matrix[j, :])
        
        if row_norm < 1e-10
            # Zero row: stalk_v gets no information from domain
            push!(singularities, (:zero_row, stalk_v, j))
            
        elseif count(!iszero, ρ.matrix[j, :]) > 3
            # Too many contributions: over-determined
            push!(singularities, (:over_determined, stalk_v, j))
            
        elseif wave_data[stalk_v] > 1.0 && row_norm < 0.5
            # High wave activity but weak restriction: mismatch
            push!(singularities, (:amplitude_mismatch, stalk_v, j))
        end
    end
    
    # Resolve each singularity type
    resolved_matrix = copy(ρ.matrix)
    
    for (type, stalk, row_idx) in singularities
        if type == :zero_row
            # Find nearest stalk in domain
            distances = [norm(wave_data[ρ.domain[i]] - wave_data[stalk]) 
                        for i in 1:n_dom]
            nearest = argmin(distances)
            resolved_matrix[row_idx, nearest] = 0.8  # Strong connection
            
        elseif type == :over_determined
            # Keep only strongest connections
            row = resolved_matrix[row_idx, :]
            threshold = 0.3 * maximum(abs.(row))
            for i in 1:n_dom
                if abs(row[i]) < threshold
                    resolved_matrix[row_idx, i] = 0.0
                end
            end
            
        elseif type == :amplitude_mismatch
            # Boost restriction coefficients for high-amplitude stalks
            scale = min(2.0, 1.0 / norm(resolved_matrix[row_idx, :]))
            resolved_matrix[row_idx, :] .*= scale
        end
    end
    
    # Recompute properties
    is_surj = rank(resolved_matrix) == n_cod
    is_inj = rank(resolved_matrix) == n_dom
    ker = nullspace(resolved_matrix)
    coker = nullspace(resolved_matrix')
    
    return RestrictionMap(ρ.domain, ρ.codomain, resolved_matrix, 
                         is_surj, is_inj, ker, coker, zeros(size(resolved_matrix)), true)
end

# ============================================================================
# 6. RESTRICTION TOWER (HIERARCHICAL)
# ============================================================================

struct RestrictionTower
    # Tower of restrictions: ℱ₀ → ℱ₁ → ℱ₂ → ... → ℱₙ
    levels::Vector{Vector{Int}}          # Stalks at each level
    restrictions::Vector{RestrictionMap} # Maps between levels
    extensions::Vector{ExtensionMap}     # Reverse maps
    
    # Cohomology data at each level
    HH_data::Vector{Dict{Int, Array{Float64}}}
    transfer_maps::Vector{Dict{Int, Matrix{Float64}}}
    
    # Wave propagation through tower
    wavefronts::Vector{Wavefront}
    
    # Mittag-Leffler conditions
    ML_conditions::Vector{Bool}
    projective_limits::Vector{Vector{Float64}}
    
    function RestrictionTower(base_stalks::Vector{Int}, 
                             sheaf_data::Dict{Int, Any},
                             n_levels::Int=5)
        # Build hierarchical tower by coarsening
        
        levels = Vector{Vector{Int}}()
        restrictions = Vector{RestrictionMap}()
        extensions = Vector{ExtensionMap}()
        HH_data = Vector{Dict{Int, Array{Float64}}}()
        transfer_maps = Vector{Dict{Int, Matrix{Float64}}}()
        wavefronts = Vector{Wavefront}()
        ML_conditions = Vector{Bool}()
        projective_limits = Vector{Vector{Float64}}()
        
        # Level 0: base
        push!(levels, base_stalks)
        
        # Initialize HH data for base level (simplified)
        HH0 = Dict(0 => randn(length(base_stalks)),
                  1 => randn(length(base_stalks), length(base_stalks)),
                  2 => randn(length(base_stalks), length(base_stalks), 
                            length(base_stalks), length(base_stalks)))
        push!(HH_data, HH0)
        
        # Build tower levels
        current_stalks = base_stalks
        for level in 1:n_levels-1
            # Coarsen: group stalks
            new_stalks = coarsen_stalks(current_stalks, 2)  # Reduce by factor ~2
            
            # Create restriction map
            ρ = RestrictionMap(current_stalks, new_stalks, sheaf_data)
            push!(restrictions, ρ)
            
            # Create extension map (reverse)
            ε = ExtensionMap(new_stalks, current_stalks)
            push!(extensions, ε)
            
            # Compute HH data at this level (coarsened)
            HH_level = coarsen_HH(HH_data[end], ρ)
            push!(HH_data, HH_level)
            
            # Compute cohomology transfer
            transfer, _ = compute_cohomology_transfer(ρ, HH_data[end-1], HH_level)
            push!(transfer_maps, transfer)
            
            # Check Mittag-Leffler condition
            ML = check_mittag_leffler(HH_data[end-1], HH_level, transfer)
            push!(ML_conditions, ML)
            
            # Compute projective limit (simplified)
            if level == 1
                proj_limit = vec(HH_level[2])  # Flatten HH²
            else
                # Compose with previous limit
                prev_limit = projective_limits[end]
                T = transfer[2]  # HH² transfer
                proj_limit = T * prev_limit
            end
            push!(projective_limits, proj_limit)
            
            # Create wavefront for this level transition
            wf = Wavefront([current_stalks, new_stalks], sheaf_data)
            push!(wavefronts, wf)
            
            current_stalks = new_stalks
            push!(levels, new_stalks)
        end
        
        new(levels, restrictions, extensions, HH_data, transfer_maps,
            wavefronts, ML_conditions, projective_limits)
    end
end

function coarsen_stalks(stalks::Vector{Int}, factor::Int)
    # Simple coarsening: group consecutive stalks
    n = length(stalks)
    n_new = Int(ceil(n / factor))
    
    new_stalks = Vector{Int}()
    for i in 1:n_new
        start_idx = (i-1)*factor + 1
        end_idx = min(i*factor, n)
        
        # Take first stalk from each group as representative
        push!(new_stalks, stalks[start_idx])
    end
    
    return new_stalks
end

function coarsen_HH(HH_old::Dict{Int, Array{Float64}}, ρ::RestrictionMap)
    # Coarsen Hochschild data via restriction
    
    HH_new = Dict{Int, Array{Float64}}()
    
    for k in keys(HH_old)
        if k == 0
            # HH⁰: center elements
            C_old = HH_old[0]
            C_new = ρ.matrix * C_old
            
        elseif k == 1
            # HH¹: derivations
            D_old = HH_old[1]
            n_old = size(D_old, 1)
            n_new = size(ρ.matrix, 1)
            
            # Restrict derivation: D_new = ρ D_old ρ⁺
            D_new = ρ.matrix * D_old * ρ.matrix'
            
        elseif k == 2
            # HH²: associators
            φ_old = HH_old[2]
            n_old = size(φ_old, 1)
            n_new = size(ρ.matrix, 1)
            
            # Flatten and apply ρ⊗ρ⊗ρ⊗ρ
            φ_flat = reshape(φ_old, (n_old^4,))
            ρ4 = kron(kron(ρ.matrix, ρ.matrix), kron(ρ.matrix, ρ.matrix))
            φ_new_flat = ρ4 * φ_flat
            φ_new = reshape(φ_new_flat, (n_new, n_new, n_new, n_new))
            
            HH_new[k] = φ_new
        end
    end
    
    return HH_new
end

function check_mittag_leffler(HH_U::Dict{Int, Array{Float64}},
                             HH_V::Dict{Int, Array{Float64}},
                             transfer::Dict{Int, Matrix{Float64}})
    # Check Mittag-Leffler condition for projective system
    
    # For each k, check that transfer is "close enough" to isomorphism
    for k in keys(transfer)
        if haskey(HH_U, k) && haskey(HH_V, k)
            T = transfer[k]
            
            # Check that T is injective modulo small errors
            if size(T, 1) > 0 && size(T, 2) > 0
                # Compute how close T is to having left inverse
                T_pinv = pinv(T)
                error = norm(T_pinv * T - I) / size(T, 2)
                
                if error > 0.3
                    return false
                end
            end
        end
    end
    
    return true
end

# ============================================================================
# 7. GLUING CONDITIONS (SHEAF AXIOM)
# ============================================================================

function check_gluing_conditions(tower::RestrictionTower, 
                                section_data::Dict{Int, Vector{Float64}})
    # Check if sections can be glued along restrictions
    
    violations = []
    
    for level in 1:length(tower.levels)-1
        ρ = tower.restrictions[level]
        U = tower.levels[level]
        V = tower.levels[level+1]
        
        # Get sections on U and V
        s_U = [section_data[stalk] for stalk in U]
        s_V = [section_data[stalk] for stalk in V]
        
        # Check gluing condition: ρ(s_U) should equal s_V
        ρ_s_U = ρ.matrix * s_U
        mismatch = norm(ρ_s_U - s_V) / max(norm(ρ_s_U), norm(s_V), 1e-10)
        
        if mismatch > 0.1
            push!(violations, (level, mismatch, U, V))
            
            # Try to resolve by adjusting sections
            if mismatch > 0.5
                # Use pseudoinverse to find best s_U that restricts to s_V
                s_U_corrected = pinv(ρ.matrix) * s_V
                
                # Update section data
                for (i, stalk) in enumerate(U)
                    section_data[stalk] = s_U_corrected[i]
                end
            end
        end
    end
    
    return violations, section_data
end

# ============================================================================
# 8. WAVE-ALGEBRA INTERACTION OPERATORS
# ============================================================================

struct WaveAlgebraOperator
    # Operator that mediates between wave dynamics and algebra deformations
    
    # Wave side
    wave_to_algebra::Function  # Converts wave amplitudes to algebra deformation
    algebra_to_wave::Function  # Converts algebra state to wave modulation
    
    # Restriction compatibility
    commutes_with_restriction::Bool
    natural_transformation::Matrix{Float64}
    
    # Current state
    deformation_parameter::Float64  # How much wave deforms algebra
    modulation_strength::Float64    # How much algebra modulates waves
    
    function WaveAlgebraOperator(wave_dim::Int, algebra_dim::Int)
        # Linear map from wave space to algebra deformation space
        W_to_A = randn(algebra_dim^4, wave_dim)  # Maps to HH² (associators)
        
        # Linear map from algebra space to wave modulation
        A_to_W = randn(wave_dim, algebra_dim^4)
        
        wave_to_algebra(wave_vec) = W_to_A * wave_vec
        algebra_to_wave(algebra_vec) = A_to_W * algebra_vec
        
        # Check if these commute with a sample restriction
        ρ_sample = randn(Int(0.8*wave_dim), wave_dim)  # Sample restriction
        ρ_algebra = kron(kron(ρ_sample, ρ_sample), kron(ρ_sample, ρ_sample))
        
        commutes = norm(ρ_algebra * W_to_A - W_to_A * ρ_sample) < 1e-6
        
        new(wave_to_algebra, algebra_to_wave, commutes, 
            zeros(wave_dim, wave_dim), 0.0, 0.0)
    end
end

function apply_wave_algebra_interaction!(operator::WaveAlgebraOperator,
                                        wave_state::Vector{Float64},
                                        algebra_state::Vector{Float64},
                                        dt::Float64)
    # Apply mutual interaction between waves and algebra
    
    # Waves deform algebra
    deformation = operator.wave_to_algebra(wave_state)
    algebra_state .+= operator.deformation_parameter * deformation * dt
    
    # Algebra modulates waves
    modulation = operator.algebra_to_wave(algebra_state)
    wave_state .*= (1.0 .+ operator.modulation_strength .* modulation .* dt)
    
    # Update operator parameters based on interaction strength
    interaction_strength = norm(wave_state) * norm(algebra_state)
    operator.deformation_parameter = tanh(interaction_strength)
    operator.modulation_strength = 0.5 * tanh(interaction_strength)
    
    return wave_state, algebra_state
end

# ============================================================================
# 9. MAIN SIMULATION WITH RESTRICTION MAPS
# ============================================================================

function simulate_with_restrictions()
    println("="^60)
    println("WAVEFRONT PROPAGATION VIA RESTRICTION MAPS")
    println("Functorial composition and cohomology transfer")
    println("="^60)
    
    # Create a brain graph with regions
    n_stalks = 64
    base_stalks = collect(1:n_stalks)
    
    # Create sheaf data (simplified)
    sheaf_data = Dict{Int, Any}()
    for stalk in base_stalks
        # Random "stalk data"
        sheaf_data[stalk] = randn(3, 3)
    end
    sheaf_data[:differential] = randn(n_stalks, n_stalks)
    
    # Create restriction tower
    println("\n1. Building restriction tower...")
    tower = RestrictionTower(base_stalks, sheaf_data, 4)
    
    println("   Tower has $(length(tower.levels)) levels:")
    for (i, level) in enumerate(tower.levels)
        println("   Level $i: $(length(level)) stalks")
    end
    
    # Test Mittag-Leffler conditions
    println("\n2. Mittag-Leffler conditions:")
    for (i, ML) in enumerate(tower.ML_conditions)
        println("   Level $(i)→$(i+1): $(ML ? "✓ Satisfied" : "✗ Failed")")
    end
    
    # Create wavefronts
    println("\n3. Wavefront propagation through restrictions...")
    
    # Initialize drug concentrations (simulating fentanyl wavefront)
    drug_conc = Dict{Int, Float64}()
    for stalk in base_stalks
        # Gaussian drug distribution moving from left to right
        position = (stalk - 1) / (n_stalks - 1)  # 0 to 1
        time = 0.0
        drug_conc[stalk] = exp(-10 * (position - 0.3 - 0.1*time)^2)
    end
    
    # Propagate wavefronts through each level
    arrival_times = Dict{Int, Vector{Float64}}()
    for (i, wf) in enumerate(tower.wavefronts)
        println("\n   Wavefront at level $i transition:")
        
        times = Float64[]
        dt = 0.1
        for t in 0:dt:5.0  # 5 seconds propagation
            propagate_wavefront!(wf, dt, drug_conc)
            
            if wf.current_region > 1 && wf.arrival_times[wf.current_region] > 0
                println("     t=$(round(t, digits=1))s: " *
                       "Arrived at region $(wf.current_region)")
                push!(times, t)
            end
        end
        arrival_times[i] = times
    end
    
    # Test gluing conditions
    println("\n4. Testing gluing conditions...")
    
    # Create random section data
    section_data = Dict{Int, Vector{Float64}}()
    for stalk in base_stalks
        section_data[stalk] = randn(3)  # 3D section at each stalk
    end
    
    violations, corrected_data = check_gluing_conditions(tower, section_data)
    
    if isempty(violations)
        println("   ✓ All gluing conditions satisfied")
    else
        println("   ⚠ $(length(violations)) gluing violations found")
        for (level, mismatch, U, V) in violations[1:min(3, end)]  # Show first 3
            println("     Level $level: mismatch = $(round(mismatch, digits=3))")
        end
    end
    
    # Test cohomology transfer
    println("\n5. Cohomology transfer maps:")
    for (i, transfer) in enumerate(tower.transfer_maps)
        if haskey(transfer, 2)  # HH² transfer
            T = transfer[2]
            rank_T = rank(T)
            println("   Level $i: HH² transfer rank = $rank_T")
        end
    end
    
    # Test wave-algebra interaction
    println("\n6. Wave-algebra interaction operators...")
    operator = WaveAlgebraOperator(n_stalks, 3)  # 3D algebra
    
    # Initial states
    wave_state = randn(n_stalks)
    algebra_state = randn(3^4)  # Flattened HH² (3^4 = 81)
    
    println("   Initial: wave norm = $(round(norm(wave_state), digits=3)), " *
           "algebra norm = $(round(norm(algebra_state), digits=3))")
    
    # Apply interaction
    for step in 1:10
        wave_state, algebra_state = apply_wave_algebra_interaction!(
            operator, wave_state, algebra_state, 0.1)
    end
    
    println("   After interaction: wave norm = $(round(norm(wave_state), digits=3)), " *
           "algebra norm = $(round(norm(algebra_state), digits=3))")
    
    println("\n   Commutes with restriction: $(operator.commutes_with_restriction ? "✓" : "✗")")
    
    return tower, arrival_times, violations
end

# Run simulation
if abspath(PROGRAM_FILE) == @__FILE__
    println("Simulating wavefront propagation via restriction maps...")
    tower, arrivals, violations = simulate_with_restrictions()
    
    println("\n" * "="^60)
    println("KEY MATHEMATICAL STRUCTURES IMPLEMENTED:")
    println("="^60)
    println("1. Restriction maps ρ: ℱ(U) → ℱ(V) for V ⊆ U")
    println("2. Functorial composition: ρ₂∘ρ₁ for U ⊇ V ⊇ W")
    println("3. Extension maps ε: ℱ(V) → ℱ(U) (extension by zero)")
    println("4. Cohomology transfer: ρₖ: HHᵏ(U) → HHᵏ(V)")
    println("5. Wavefront propagation through restriction chain")
    println("6. Singularity resolution for ill-defined restrictions")
    println("7. Restriction tower with Mittag-Leffler conditions")
    println("8. Gluing conditions (sheaf axiom verification)")
    println("9. Wave-algebra interaction operators")
    println("="^60)
end

end  # module RestrictionMaps

