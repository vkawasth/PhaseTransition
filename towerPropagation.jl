struct MetaplecticTower
    levels::Vector{Int}  # Polynomial degrees at each level: [N₀, N₁, ..., Nₗ]
    sheaves::Vector{NeuroSheaf}  # Sheaf at each level
    metaplectic_generators::Vector{Matrix{ComplexF64}}  # Θ⁽ᵏ⁾
    dopamine_levels::Vector{Float64}  # D⁽ᵏ⁾(t)
    projection_maps::Vector{Matrix{Float64}}  # πₖ: ℱ⁽ᵏ⁾ → ℱ⁽ᵏ⁻¹⁾
end

function propagate_metaplectic_flow!(tower::MetaplecticTower, 
                                     base_dopamine::Float64,
                                     dt::Float64)
    # 1. Update finest level (k=0)
    D0 = base_dopamine
    M0 = exp(im * D0 * tower.metaplectic_generators[1])
    apply_metaplectic!(tower.sheaves[1], M0)
    tower.dopamine_levels[1] = D0
    
    # 2. Propagate upward through tower
    for k in 2:length(tower.levels)
        # Induce dopamine at this level (scaling law)
        Dk = scaling_law(D0, k)
        
        # Compute induced metaplectic transformation
        # Option A: Direct computation from finer level
        π = tower.projection_maps[k-1]  # πₖ: level k-1 → k
        Mk_induced = π' * M0 * π  # Pushforward
        
        # Option B: Use level-specific generator
        Mk_direct = exp(im * Dk * tower.metaplectic_generators[k])
        
        # Blend: Mk = α Mk_induced + (1-α) Mk_direct
        α = coherence_parameter(tower.sheaves[k], tower.sheaves[k-1])
        Mk = α * Mk_induced + (1-α) * Mk_direct
        
        # Apply to sheaf at level k
        apply_metaplectic!(tower.sheaves[k], Mk)
        tower.dopamine_levels[k] = Dk
    end
end

function scaling_law(D0::Float64, k::Int)
    # Critical scaling: D_c^(k) = λ^k D_c^(0)
    λ = 0.8  # For downward propagation (coarser levels transition first)
    return D0 * λ^k
end
