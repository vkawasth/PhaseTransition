function compute_band_dominance(state::Vector{ComplexF64},
                                theta_basis::BandPolynomialBasis,
                                alpha_basis::BandPolynomialBasis)
    # Project state onto each band basis
    # State is in combined Hilbert space
    
    N_theta = theta_basis.degree
    N_alpha = alpha_basis.degree
    
    # Extract theta and alpha components
    theta_component = state[1:N_theta]
    alpha_component = state[N_theta+1:N_theta+N_alpha]
    
    # Compute energies
    E_theta = norm(theta_component)^2
    E_alpha = norm(alpha_component)^2
    
    # Dominance parameter
    if E_theta + E_alpha > 0
        R = (E_theta - E_alpha) / (E_theta + E_alpha)
    else
        R = 0.0
    end
    
    return (E_theta, E_alpha, R)
end

function detect_phase_transition(R_history::Vector{Float64}, 
                                 threshold::Float64=0.5)
    # R = (E_theta - E_alpha)/(E_theta + E_alpha)
    # R > threshold: theta dominant (executive mode)
    # R < -threshold: alpha dominant (default mode)
    
    transitions = []
    
    for t in 2:length(R_history)
        if (R_history[t-1] < threshold && R_history[t] >= threshold) ||
           (R_history[t-1] > -threshold && R_history[t] <= -threshold)
            push!(transitions, (t, R_history[t]))
        end
    end
    
    return transitions
end
