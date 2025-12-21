using FastGaussQuadrature, LinearAlgebra, SpecialFunctions

struct BandPolynomialBasis
    band::Symbol  # :theta, :alpha, :beta, :gamma
    ω_min::Float64
    ω_max::Float64
    degree::Int
    points::Vector{Float64}  # Quadrature points
    weights::Vector{Float64} # Quadrature weights
    polynomials::Matrix{Float64}  # ψ_n(ω_m) at points
    jacobi_matrix::Matrix{Float64}  # J^b
end

function create_band_basis(band::Symbol, degree::Int)
    # Define frequency range for band
    ranges = Dict(
        :theta => (4.0, 8.0),
        :alpha => (8.0, 13.0),
        :beta => (13.0, 30.0),
        :gamma => (30.0, 100.0)
    )
    ω_min, ω_max = ranges[band]
    
    # Use Legendre polynomials mapped to [ω_min, ω_max]
    # Gaussian quadrature points
    N = degree + 1
    ξ, w = gausslegendre(N)
    
    # Map from [-1,1] to [ω_min, ω_max]
    points = (ω_max - ω_min)/2 * ξ .+ (ω_max + ω_min)/2
    weights = (ω_max - ω_min)/2 * w
    
    # Compute Legendre polynomials at mapped points
    P = zeros(N, N)
    for n in 0:degree
        # Legendre polynomial of degree n
        coeffs = zeros(N)
        coeffs[n+1] = 1
        P[:, n+1] = legendre.(points, n)  # Need Legendre polynomial function
    end
    
    # Orthonormalize
    for n in 1:N
        # Gram-Schmidt
        for m in 1:(n-1)
            proj = dot(P[:,m], P[:,n] .* weights)
            P[:,n] -= proj * P[:,m]
        end
        P[:,n] ./= sqrt(dot(P[:,n], P[:,n] .* weights))
    end
    
    # Compute Jacobi matrix (tridiagonal)
    J = zeros(N, N)
    for n in 1:N
        # Multiplication by ω in this basis
        ωψ = points .* P[:,n]
        for m in 1:N
            J[m,n] = dot(P[:,m], ωψ .* weights)
        end
    end
    
    return BandPolynomialBasis(band, ω_min, ω_max, degree, 
                               points, weights, P, J)
end
