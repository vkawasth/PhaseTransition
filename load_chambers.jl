# load_chambers.jl — shared loader for the actual chambers.tsv format
# Include this at the top of each plot script with:  include("load_chambers.jl")
#
# Actual columns:
# 1:id  2:time  3:dominant(text)  4:gap  5:obs  6:score  7:support_sum
# 8:perv_sum  9:bridgeland_phase  10:klein_constraint  11:lefschetz_proxy
# 12:gap_complex_re  13:gap_complex_im  14:plucker_phase

using DelimitedFiles

function load_chambers(filename="chambers.tsv")
    raw     = readdlm(filename, '\t', Any, skipstart=1)
    n       = size(raw, 1)

    snap    = Int.(raw[:, 1])
    # col 3 is text (dominant region) — skip
    gap     = parse.(Float64, string.(raw[:, 4]))
    obs     = parse.(Float64, string.(raw[:, 5]))   # m6 obstruction
    score   = parse.(Float64, string.(raw[:, 6]))
    support = parse.(Float64, string.(raw[:, 7]))
    perv    = parse.(Float64, string.(raw[:, 8]))
    bphase  = parse.(Float64, string.(raw[:, 9]))   # Bridgeland phase
    K       = parse.(Float64, string.(raw[:,10]))   # Klein constraint
    lefsch  = parse.(Float64, string.(raw[:,11]))   # Lefschetz proxy
    gap_re  = parse.(Float64, string.(raw[:,12]))   # Re(complex gap)
    gap_im  = parse.(Float64, string.(raw[:,13]))   # Im(complex gap)
    pphase  = parse.(Float64, string.(raw[:,14]))   # Plücker phase

    # Derived quantities
    # Schubert stratum from Klein constraint
    K_max   = maximum(abs.(K))
    stratum = [abs(K[i])/K_max < 0.05 ? 4 :
               abs(K[i])/K_max < 0.20 ? 3 :
               abs(K[i])/K_max < 0.50 ? 2 :
               abs(K[i])/K_max < 0.85 ? 1 : 0 for i in 1:n]

    # Wall crossings: local maxima of obs above 90th percentile
    thresh  = quantile(abs.(obs), 0.90)
    walls   = zeros(Int, n)
    for i in 2:n-1
        if abs(obs[i]) > thresh && abs(obs[i]) > abs(obs[i-1]) && abs(obs[i]) > abs(obs[i+1])
            walls[i] = 1
        end
    end

    # Synthetic Plücker coordinates from available data
    # K = q12*q34 - q13*q24 + q14*q23  (Klein constraint — we have this directly)
    # Approximate individual coords from phase and magnitude:
    #   q12 ≈ sqrt(|gap_re|) * cos(pphase)
    #   q34 ≈ sqrt(|gap_re|) * sin(pphase)
    #   q13 ≈ gap_re / (sqrt(|gap_re|)+1e-8)
    #   q24 ≈ gap_im / (sqrt(|gap_re|)+1e-8)
    #   q14 ≈ perv / (support+1e-8)
    #   q23 ≈ K + q13*q24 - q12*q34  (from Klein constraint)
    mag    = sqrt.(max.(abs.(gap_re), 0.0) .+ 1e-8)
    q12    = mag .* cos.(pphase)
    q34    = mag .* sin.(pphase)
    q13    = gap_re ./ (mag .+ 1e-8)
    q24    = gap_im ./ (mag .+ 1e-8)
    q14    = perv   ./ (support .+ 1e-8)
    q23    = K .+ q13 .* q24 .- q12 .* q34

    return (snap=snap, gap=gap, obs=obs, score=score, support=support,
            perv=perv, bphase=bphase, K=K, lefsch=lefsch,
            gap_re=gap_re, gap_im=gap_im, pphase=pphase,
            q12=q12, q13=q13, q14=q14, q23=q23, q24=q24, q34=q34,
            stratum=stratum, walls=walls, n=n)
end
