struct MetaplecticTransformer
    source_band::BandPolynomialBasis
    target_band::BandPolynomialBasis
    transformation_matrix::Matrix{ComplexF64}  # U: source → target
    dopamine_parameter::Float64  # D
end

function create_interband_metaplectic(theta_basis::BandPolynomialBasis,
                                      alpha_basis::BandPolynomialBasis,
                                      D::Float64)
    N = min(theta_basis.degree, alpha_basis.degree)
    
    # Create mixing matrix based on dopamine
    # Generator Θ = a_θ^† a_α + h.c.
    Θ = zeros(ComplexF64, 2N, 2N)
    
    for n in 1:N
        # Mix n-th theta mode with n-th alpha mode
        Θ[n, N+n] = 1.0  # theta_n → alpha_n
        Θ[N+n, n] = 1.0  # alpha_n → theta_n
    end
    
    # Metaplectic transformation = exp(iDΘ)
    U = exp(im * D * Θ)
    
    # Extract theta→alpha block
    U_theta_to_alpha = U[1:N, (N+1):(2N)]
    
    return MetaplecticTransformer(theta_basis, alpha_basis, 
                                  U_theta_to_alpha, D)
end
