function apply_dopamine_metaplectic!(state::Vector{ComplexF64},
                                     transformer::MetaplecticTransformer,
                                     dopamine_increment::Float64)
    
    # Update dopamine parameter
    transformer.dopamine_parameter += dopamine_increment
    
    # Recompute transformation matrix with new D
    D = transformer.dopamine_parameter
    N = size(transformer.transformation_matrix, 1)
    
    # Generator matrix (simplified)
    Θ = zeros(ComplexF64, 2N, 2N)
    for n in 1:N
        Θ[n, N+n] = 1.0
        Θ[N+n, n] = 1.0
    end
    
    # New metaplectic transformation
    U_new = exp(im * D * Θ)
    
    # Apply to state
    state .= U_new * state
    
    return state
end

function simulate_phase_transition(total_time::Float64, dt::Float64)
    # Create band bases
    theta_basis = create_band_basis(:theta, 10)
    alpha_basis = create_band_basis(:alpha, 10)
    
    # Initial state: alpha dominant
    N = 10
    state = zeros(ComplexF64, 2N)
    state[N+1:2N] = randn(ComplexF64, N)  # Alpha modes excited
    state[N+1:2N] ./= norm(state[N+1:2N])
    
    # Create metaplectic transformer
    transformer = create_interband_metaplectic(theta_basis, alpha_basis, 0.0)
    
    # Storage
    time_points = 0:dt:total_time
    R_history = Float64[]
    dopamine_level = 0.0
    
    for (i, t) in enumerate(time_points)
        # Gradually increase dopamine
        if t > 1.0 && dopamine_level < 2.0
            dopamine_increment = 0.1 * dt
            dopamine_level += dopamine_increment
            apply_dopamine_metaplectic!(state, transformer, dopamine_increment)
        end
        
        # Compute dominance
        E_theta, E_alpha, R = compute_band_dominance(state, theta_basis, alpha_basis)
        push!(R_history, R)
        
        # Add noise/small fluctuations
        state .+= 0.01 * randn(ComplexF64, 2N)
        state ./= norm(state)
    end
    
    return time_points, R_history, dopamine_level
end
