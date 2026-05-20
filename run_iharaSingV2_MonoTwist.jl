# run_iharaSingV2.jl
#
# Full production pipeline (derived (2,1)-stack analysis)
#   - Bridge (Ihara proxy, unified zeta)
#   - Singularity tracker (blow‑up events, perversity)
#   - Schober navigator (chambers, walls)
#   - Rees‑Grassmann bridge (exceptional divisor, Rees charts)
#   - Export of combined data (Plücker phase, Dehn error, prime ideal matrix)
#
# Usage: julia run_iharaSingV2.jl [folder] [max_files]
#   folder    : directory containing ainf_export_*.json (default ".")
#   max_files : limit to first N files for testing (default 0 = all)

using JSON3, LinearAlgebra, Statistics, CairoMakie
using Printf
using CSV
using DataFrames
using StaticArrays   # zero-alloc 4x4 monodromy

# Include modules (adjust paths if needed)
include("OnlineAssociahedronNavigatorV3.jl")
include("IharaAssociahedronBridgeV2.jl")
include("SingularityTrackerV2.jl")
include("SchobarNavigatorV2.jl")
include("ReesGrassmannBridge.jl")

using .OnlineAssociahedronNavigatorV3
using .IharaAssociahedronBridgeV2
using .SingularityTracker
using .SingularityTracker: TrackerResult

using .SchoberNavigatorV2
using .ReesGrassmannBridge

# NOTE **********************************************************************
# ORDER OF REGIONS MUST MATCH WHAT WE SEE FROM
# build_region_graph.py -- Shown below for PAL
# Regions : ['BLA', 'CA1sp', 'HPF', 'HY', 'LA', 'PAL', 'sAMY']
# Regions : ['BLA', 'CA1sp', 'HPF', 'HY', 'LA', 'LSX', 'sAMY']
# Regions: ['BLA', 'CA1sp', 'HPF', 'HY', 'LA', 'LSX', 'PAL', 'sAMY']
# ***************************************************************************
const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :LSX, :sAMY]
load_json(file) = JSON3.read(read(file,String))

# ============================================================================
# SNAPSHOT CACHE — single-pass JSON loading
#
# Every function that iterates snapshot_files calls load_cached(i) instead of
# load_json(path). The first call parses the file; subsequent calls return the
# already-parsed object. This reduces 5 sequential full-file passes to 1.
#
# Usage:
#   init_snapshot_cache!(snapshot_files, folder)   # call once in main()
#   snap = load_cached(i)                          # O(1) lookup everywhere
# ============================================================================
const SNAP_CACHE = Dict{Int, Any}()

function init_snapshot_cache!(snapshot_files, folder; verbose=true)
    empty!(SNAP_CACHE)
    n = length(snapshot_files)
    verbose && println("  Loading $n snapshots into memory cache...")
    t0 = time()
    fps = String[]
    sevs = Float64[]
    for (i, f) in enumerate(snapshot_files)
        snap = load_json(joinpath(folder, f))
        SNAP_CACHE[i] = snap
        push!(fps,  snapshot_fingerprint(snap))
        push!(sevs, snapshot_severity(snap))
        i % 200 == 0 && verbose && @printf("    loaded %d/%d\n", i, n)
    end
    elapsed = round(time() - t0, digits=1)
    n_unique = length(unique(fps))
    pct_stable = round(100.0*(1 - n_unique/max(n,1)), digits=1)
    verbose && println("  Cache ready: $n snapshots in $(elapsed)s")
    verbose && @printf("  Unique fingerprints: %d / %d  (%.1f%% stable duplicates)\n",
                       n_unique, n, pct_stable)
    return fps, sevs
end

load_cached(i::Int) = SNAP_CACHE[i]

# ============================================================================
# STATICARRAYS SO(4) MONODROMY — exact Rodrigues formula for Gr(2,4)
#
# Replaces: exp(theta*Omega) [Pade, ~40 us] + svd(M) [~15 us] per snapshot
# With:     Rodrigues on SMatrix{4,4} [~0.5 us], zero heap allocations
#
# For a rank-2 bivector Omega, Omega^3 = -alpha^2 * Omega exactly, so the
# infinite matrix exponential collapses to three terms. No approximation.
# ============================================================================

@inline function _build_bivector(q12,q13,q14,q23,q24,q34)
    klein = q12*q34 - q13*q24 + q14*q23
    nf = sqrt(q12^2+q13^2+q14^2+q23^2+q24^2+q34^2)
    nf > 1e-10 || return (@SMatrix zeros(4,4)), 0.0, 0.0, nf, klein
    s = 1/nf
    q12*=s; q13*=s; q14*=s; q23*=s; q24*=s; q34*=s
    Omega = @SMatrix [
         0.0   q12   q13   q14
        -q12   0.0   q23   q24
        -q13  -q23   0.0   q34
        -q14  -q24  -q34   0.0 ]
    return Omega, (pi/2)*nf, 1.0, nf, klein
end

@inline function _rodrigues(Omega::SMatrix{4,4,Float64,16}, theta::Float64)
    I4  = SMatrix{4,4,Float64,16}(I)
    Om2 = Omega * Omega
    tr2 = Om2[1,1]+Om2[2,2]+Om2[3,3]+Om2[4,4]   # = -2*alpha^2
    abs(tr2) < 1e-14 && return I4
    alpha = sqrt(-tr2/2)
    alpha < 1e-12 && return I4
    ta = theta*alpha
    ia = 1/alpha
    return I4 + sin(ta)*ia*Omega + (1-cos(ta))*(ia*ia)*Om2
end

function monodromy_fast(snap)::Matrix{Float64}
    q12=1.0; q13=0.0; q14=0.0; q23=0.0; q24=0.0; q34=0.0

    if haskey(snap,:prime_paths) && !isempty(snap[:prime_paths])
        tlw = 0.0
        for pp in snap[:prime_paths]
            if haskey(pp,:weight)
                w=Float64(pp[:weight]); (w>0&&isfinite(w)) && (tlw+=log10(w))
            end
        end
        if tlw>0
            q12 = 1.0+0.1*min(tlw/30,0.5)
            q34 = 0.1*min(length(snap[:prime_paths])/10,0.2)
        end
    end

    if haskey(snap,:prime_higher_ideals)
        ts=0.0
        for id in snap[:prime_higher_ideals]
            haskey(id,:total_support) && (ts+=Float64(id[:total_support]))
        end
        if ts>0
            q13=0.1*min(log10(ts+1),10.0)
            q24=0.05*min(log10(ts+1),5.0)
        end
    end

    for (mult,factor) in ((:m4,1.0),(:m5,2.0),(:m6,3.0))
        if haskey(snap,mult)
            n_nz=0
            for (_,v) in snap[mult]
                (v isa Number&&!iszero(v)&&isfinite(Float64(v))) && (n_nz+=1)
            end
            q14+=factor*min(n_nz/500,pi/6)/pi
            q23+=0.5*factor*min(n_nz/500,pi/6)/pi
        end
    end

    nf=sqrt(q12^2+q13^2+q14^2+q23^2+q24^2+q34^2)
    if nf>1e-10; s=1/nf; q12*=s;q13*=s;q14*=s;q23*=s;q24*=s;q34*=s; end

    Omega,theta,_,_,_ = _build_bivector(q12,q13,q14,q23,q24,q34)
    return Matrix{Float64}(_rodrigues(Omega,theta))
end

# ======================================================================
# STABILITY-AWARE DEDUPLICATION CACHE
#
# In the stable zone (chambers 20-700) the best tubing, prime paths,
# and monodromy are identical snapshot after snapshot.
# Computing eigvals/exp(Ω)/SVD 800× for identical inputs is the
# cause of the 2-day runtime.
#
# Strategy:
#   1. Fingerprint each snapshot by a cheap hash of its key fields
#   2. If fingerprint matches the previous K snapshots → skip full computation
#   3. Periodically (every FORCE_INTERVAL) recompute regardless
#   4. At blowup events (high severity) always recompute
#
# FORCE_INTERVAL: force recomputation every N stable snapshots
# LOOKBACK: number of previous snapshots to compare fingerprint against
# SEVERITY_THRESHOLD: always recompute when m6 norm exceeds this
# ======================================================================

const FORCE_INTERVAL     = 50    # recompute every 50 stable snapshots
const LOOKBACK           = 3     # compare against last 3 fingerprints
const SEVERITY_THRESHOLD = 1e6   # always recompute above this m6 norm

"""
Cheap fingerprint of a snapshot for deduplication.
Uses: top prime path signature + m6 total norm + n_prime_paths.
Fast to compute — no matrix algebra.
"""
function snapshot_fingerprint(snap)::String
    np = haskey(snap, :prime_paths) ? length(snap[:prime_paths]) : 0
    
    # Top path signature
    top_path = ""
    if haskey(snap, :prime_paths) && !isempty(snap[:prime_paths])
        pp = snap[:prime_paths]
        sorted_pp = sort(collect(pp), by=p->get(p,:weight,0.0), rev=true)
        if !isempty(sorted_pp) && haskey(sorted_pp[1], :path)
            full_path = join(sorted_pp[1][:path], "→")
            # Safe truncation
            chars = collect(full_path)
            top_path = length(chars) > 50 ? String(chars[1:50]) : full_path
        end
    end
    
    # m6 count
    m6_count = haskey(snap, :m6) ? length(snap[:m6]) : 0
    
    # Tubing signature
    tubing_sig = ""
    if haskey(snap, :prime_higher_ideals) && !isempty(snap[:prime_higher_ideals])
        first_ideal = snap[:prime_higher_ideals][1]
        if haskey(first_ideal, :closure)
            full_sig = join(sort(string.(first_ideal[:closure])), ",")
            chars = collect(full_sig)
            tubing_sig = length(chars) > 30 ? String(chars[1:30]) : full_sig
        end
    end
    
    return "$(np)|$(m6_count)|$(top_path)|$(tubing_sig)"
end
"""
Snapshot severity for deciding whether to force recomputation.
Returns m6 total support norm.
"""
function snapshot_severity(snap)::Float64
    haskey(snap, :m6) || return 0.0
    total = 0.0
    for (k, v) in snap[:m6]
        if v isa Number && isfinite(Float64(v))
            total += Float64(v)^2
        end
    end
    return sqrt(total)
end

"""
Decide whether to skip full computation for snapshot i.
Returns (skip::Bool, reason::String)

Skip when:
  - fingerprint matches last LOOKBACK snapshots
  - severity below threshold
  - not at a FORCE_INTERVAL boundary
  - not the first or last snapshot
"""
function should_skip(i::Int, n_total::Int,
                     fingerprint::String,
                     recent_fingerprints::Vector{String},
                     severity::Float64)::Tuple{Bool,String}
    # Never skip first, last, or crisis snapshots
    i == 1        && return (false, "first snapshot")
    i == n_total  && return (false, "last snapshot")
    severity > SEVERITY_THRESHOLD && return (false, "high severity $(round(severity,digits=0))")

    # Always recompute at force intervals
    i % FORCE_INTERVAL == 0 && return (false, "force interval")

    # Skip if fingerprint matches recent history
    length(recent_fingerprints) < LOOKBACK && return (false, "insufficient history")
    all(fp == fingerprint for fp in recent_fingerprints[end-min(LOOKBACK,length(recent_fingerprints))+1:end]) &&
        return (true, "identical to last $(LOOKBACK) snapshots")

    return (false, "fingerprint changed")
end

# ----------------------------------------------------------------------
# Helper: parse timestamp from filename (for ordering)
# ----------------------------------------------------------------------
function parse_timestamp(fname)
    m = match(r"([0-9]+\.[0-9]+)", basename(fname))
    return m === nothing ? 0.0 : parse(Float64, m.captures[1])
end

# ----------------------------------------------------------------------
# Compute Plücker phase from plucker_trajectory.json
# ----------------------------------------------------------------------
function compute_plucker_phase(file="plucker_trajectory.json")
    if !isfile(file)
        @warn "$file not found; Plücker phase will be missing."
        return nothing
    end
    data = JSON3.read(read(file, String))
    q12 = Float64.(data["q12"])
    q13 = Float64.(data["q13"])
    phase = atan.(q13, q12)
    return phase
end

# ----------------------------------------------------------------------
# Monodromy computation for Dehn error
# ----------------------------------------------------------------------

function build_monodromy_from_plucker(coords)
    """
    Build SO(4) monodromy matrix from Plücker coordinates.
    Implements the Gauss-Manin connection on Gr(2,4).
    """
    q12, q13, q14, q23, q24, q34 = coords
    
    # Verify Klein quadric constraint (monitors coherence)
    klein_constraint = q12*q34 - q13*q24 + q14*q23
    if abs(klein_constraint) > 1e-6
        # Non-zero constraint indicates ghost signal precursor
        @debug "Klein quadric deviation: $klein_constraint"
    end
    
    # Normalize to prevent numerical explosion
    norm_factor = sqrt(q12^2 + q13^2 + q14^2 + q23^2 + q24^2 + q34^2)
    if norm_factor > 1e-10
        q12 /= norm_factor; q13 /= norm_factor; q14 /= norm_factor
        q23 /= norm_factor; q24 /= norm_factor; q34 /= norm_factor
    else
        return Matrix{Float64}(I, 4, 4)  # Zero Plücker = identity monodromy
    end
    
    # Build skew-symmetric bivector matrix
    Ω = zeros(Float64, 4, 4)
    Ω[1,2] = q12; Ω[2,1] = -q12
    Ω[1,3] = q13; Ω[3,1] = -q13
    Ω[1,4] = q14; Ω[4,1] = -q14
    Ω[2,3] = q23; Ω[3,2] = -q23
    Ω[2,4] = q24; Ω[4,2] = -q24
    Ω[3,4] = q34; Ω[4,3] = -q34
    
    # Rotation angle = winding number × π × path length
    # For typical snapshot interval, use π/2
    θ = π/2 * norm_factor
    
    # Compute matrix exponential for SO(4)
    # exp(θΩ) using Rodrigues-like formula for 4D
    Ω2 = Ω * Ω
    Ω_norm = sqrt(sum(Ω[i,j]^2 for i=1:4 for j=1:4) / 2)
    
    if Ω_norm < 1e-10
        return Matrix{Float64}(I, 4, 4)
    end
    
    # For skew-symmetric in 4D: exp(θΩ) = I + sin(θΩ_norm) * (Ω/Ω_norm) + (1-cos(θΩ_norm)) * (Ω²/Ω_norm²)
    # But simpler: use matrix exponential from LinearAlgebra
    M = exp(θ * Ω)
    
    # Ensure orthogonal (numerical stability)
    M = real(M)
    
    # Orthogonalize via SVD for SO(4) guarantee
    U, _, Vt = svd(M)
    M = U * Vt
    
    # Ensure determinant = +1 (rotation, not reflection)
    if det(M) < 0
        M[:, end] = -M[:, end]
    end
    
    return M
end

# -ainf_mode computes only needed m3..m6 and not perversity gerstenhaber cup etc..
# following function falls back to monodrom_primepaths count based logic when 
# json files are not fully populated by simulation as we only call julia using -full
# for full blown events.

"""   Do not use Graph Monodromy, use Plucker's for winding...

# From OnlineAssociahedronNavigatorV3.jl
# This tracks the geometric monodromy of the associahedron
# As you navigate the moduli space of A∞ structures

function monodromy_navigator(path_state)
    # Tracks how tubing flips accumulate
    # Monodromy = product of transition matrices between chambers
    # Returns GL(n) matrix representing path-ordered product
end
What it measures:
    Geometric monodromy of the associahedron fibration
    How chamber transitions compose under path-ordered product
    Topological obstruction to global trivialization of the moduli space

From your compute_dehn_error
# This tracks algebraic monodromy from obstruction tensors

function monodromy_snapshot(snapshot)
    # Uses prime_paths weights + m4/m5/m6 obstructions
    # Returns SO(4) matrix for Gr(2,4) embedding
end

What it measures:
    Algebraic monodromy of the A∞ structure itself
    How prime paths generate monodromy in the Grassmannian
    Deformation obstruction to lifting to higher multiplications

"""

# Checks type etc...
# TEMPORARY DEBUG - Replace or add this before your monodromy function
function monodromy_debug(snap)
    println("=== DEBUG: monodromy called ===")
    println("  Type of snap: ", typeof(snap))
    println("  Keys: ", keys(snap))
    
    # Check if it's a Dict with Symbol keys
    if snap isa Dict{Symbol, Any}
        println("  ✓ Dict with Symbol keys")
        if haskey(snap, :prime_paths)
            println("  ✓ Has :prime_paths")
            pp = snap[:prime_paths]
            println("    Length: ", length(pp))
            if !isempty(pp)
                println("    First weight: ", pp[1][:weight])
            end
        else
            println("  ✗ Does NOT have :prime_paths")
        end
    else
        println("  ✗ Not a Dict{Symbol, Any}")
        println("  Actual type: ", typeof(snap))
    end
    
    # Return identity matrix
    N = 4
    return Matrix{Float64}(I, N, N)
end


function monodromy(snapshot)  # Remove the type restriction
    """
    Compute monodromy matrix from A∞ export snapshot.
    Uses prime_paths and obstruction data.
    """
    N = 4  # Gr(2,4) dimension
    θ_total = 0.0
    
    # ------------------------------------------------------------------
    # 1. Extract from prime_paths (using Symbol keys)
    # ------------------------------------------------------------------
    if haskey(snapshot, :prime_paths)
        prime_paths = snapshot[:prime_paths]
        
        if !isempty(prime_paths)
            total_log_weight = 0.0
            
            for pp in prime_paths
                # Access fields - JSON3 objects support Symbol indexing
                if haskey(pp, :weight)
                    w = Float64(pp[:weight])
                    if w > 0 && isfinite(w)
                        total_log_weight += log10(w)
                    end
                end
            end
            
            if total_log_weight > 0
                θ_geo = π * min(total_log_weight / 30.0, 0.5)
                θ_total += θ_geo
            end
            
            # Additional contribution from number of paths
            θ_total += 0.05 * π * min(length(prime_paths) / 10.0, 0.2)
        end
    end
    
    # ------------------------------------------------------------------
    # 2. Extract from prime_higher_ideals (Symbol keys)
    # ------------------------------------------------------------------
    if haskey(snapshot, :prime_higher_ideals)
        higher_ideals = snapshot[:prime_higher_ideals]
        total_support = 0.0
        
        for ideal in higher_ideals
            if haskey(ideal, :total_support)
                total_support += Float64(ideal[:total_support])
            end
        end
        
        if total_support > 0 && isfinite(total_support)
            θ_ideal = 0.1 * π * min(log10(total_support + 1), 10.0)
            θ_total += θ_ideal
        end
    end
    
    # ------------------------------------------------------------------
    # 3. Extract from higher obstructions (m3, m4, m5, m6)
    # ------------------------------------------------------------------
    for mult in [:m3, :m4, :m5, :m6]
        if haskey(snapshot, mult)
            obstructions = snapshot[mult]
            n_nonzero = 0
            
            # Count non-zero obstructions safely
            for (key, val) in obstructions
                if val isa Dict || val isa JSON3.Object
                    for (k, v) in val
                        if v isa Number && !iszero(v) && isfinite(v)
                            n_nonzero += 1
                        end
                    end
                elseif val isa Number && !iszero(val) && isfinite(val)
                    n_nonzero += 1
                end
            end
            
            # Scale by multiplier (higher m = higher obstruction)
            mult_factor = Dict(:m3 => 0.5, :m4 => 1.0, :m5 => 2.0, :m6 => 3.0)[mult]
            θ_total += mult_factor * min(n_nonzero / 500.0, π/6)
        end
    end
    
    # Cap the total angle
    θ_total = min(θ_total, π)
    
    # ------------------------------------------------------------------
    # Build SO(4) monodromy matrix
    # ------------------------------------------------------------------
    M = Matrix{Float64}(I, N, N)
    
    if θ_total > 1e-10
        c = cos(θ_total)
        s = sin(θ_total)
        
        # Primary rotation in 1-2 plane
        M[1,1] = c
        M[1,2] = -s
        M[2,1] = s
        M[2,2] = c
    end
    
    # Ensure orthogonal and determinant = +1
    try
        U, _, Vt = svd(M)
        M = U * Vt
        if det(M) < 0
            M[:, end] = -M[:, end]
        end
    catch
        # SVD failed, return identity
        M = Matrix{Float64}(I, N, N)
    end
    
    return M
end



# Test Monodromy program
# Temporary test function
function test_monodromy()
    snap = load_json("test_subset_3/ainf_export_1777770806.68.json")
    println("Keys type: ", typeof(keys(snap)))
    println("First key: ", first(keys(snap)))
    println("Has :prime_paths? ", haskey(snap, :prime_paths))
    
    # Get prime_paths
    pp = snap[:prime_paths]
    println("Number of prime paths: ", length(pp))
    println("First prime path: ", pp[1])
    println("First weight: ", pp[1][:weight])
    
    # Call monodromy
    M = monodromy(snap)
    println("Monodromy matrix:\n", M)
    return M
end


function monodromy_primepaths(snapshot::Dict)
    """
    Simplified monodromy from A∞ data.
    Uses prime path count as proxy for monodromy complexity.
    """
    N = 4
    
    # Count prime paths (indecomposable obstructions)
    n_prime_paths = length(get(snapshot, "prime_paths", []))
    n_higher_ideals = length(get(snapshot, "prime_higher_ideals", []))
    
    # Total obstruction indicator
    obstruction_index = n_prime_paths + n_higher_ideals
    
    # Monodromy angle increases with obstruction count
    θ = π * min(obstruction_index / 100.0, 1.0)
    
    # Build rotation matrix
    M = Matrix{Float64}(I, N, N)
    M[1,1] = cos(θ)
    M[1,2] = -sin(θ)
    M[2,1] = sin(θ)
    M[2,2] = cos(θ)
    
    return M
end

# ----------------------------------------------------------------------
# Compute Dehn twist error (monodromy coherence) per snapshot
# Monodromy matrix is built for each snapshot; we accumulate product over
# three dose intervals (based on dose times). For simplicity, we compute
# cumulative product and output norm(M_cumulative - I).
# ----------------------------------------------------------------------
function compute_dehn_error(snapshot_files, folder;
        precomputed_fps::Union{Vector{String},Nothing}=nothing,
        precomputed_sevs::Union{Vector{Float64},Nothing}=nothing)
    # OPTIMISED: uses SNAP_CACHE (no disk reads) + monodromy_fast (Rodrigues,
    # ~0.5 us) + fingerprint monodromy cache (skip duplicate configs).
    N         = 4
    M_total   = Matrix{Float64}(I, N, N)
    error_series     = Float64[]
    monodromy_series = []
    mono_cache       = Dict{String, Matrix{Float64}}()   # fp -> matrix
    recent_fps       = String[]
    last_M           = Matrix{Float64}(I, N, N)
    last_err         = 0.0
    n_total          = length(snapshot_files)
    n_skip=0; n_compute=0; n_cache=0

    println("\n  Dehn error across $n_total snapshots (cached)...")

    for i in 1:n_total
        snap = load_cached(i)
        fp   = precomputed_fps  !== nothing ? precomputed_fps[i]  : snapshot_fingerprint(snap)
        sev  = precomputed_sevs !== nothing ? precomputed_sevs[i] : snapshot_severity(snap)

        if i == 1
            println("  Sample keys: ", collect(keys(snap))[1:min(5,length(keys(snap)))])
        end

        skip, reason = should_skip(i, n_total, fp, recent_fps, sev)

        if skip
            push!(error_series, last_err)
            push!(monodromy_series, last_M)
            n_skip += 1
        elseif haskey(mono_cache, fp)
            M = mono_cache[fp]
            M_total = M * M_total
            err = norm(M_total - I)
            push!(error_series, err); push!(monodromy_series, M_total)
            last_M=M_total; last_err=err; n_cache+=1
        else
            try
                M = monodromy_fast(snap)
                mono_cache[fp] = M
                M_total = M * M_total
                err = norm(M_total - I)
                push!(error_series, err); push!(monodromy_series, M_total)
                last_M=M_total; last_err=err; n_compute+=1
                if 2.5 < err < 3.5
                    @info "  [GHOST] snapshot $i error=$(round(err,digits=3)) $reason"
                end
            catch e
                @warn "  monodromy_fast failed snapshot $i: $(e)"
                push!(error_series, NaN); push!(monodromy_series, nothing)
            end
        end

        push!(recent_fps, fp)
        length(recent_fps) > LOOKBACK+2 && popfirst!(recent_fps)

        if i % 100 == 0 || i == n_total
            saved = n_skip+n_cache
            @printf("    %d/%d  computed=%d cached=%d skipped=%d (%.0f%% saved)\n",
                    i, n_total, n_compute, n_cache, n_skip,
                    100*saved/max(i,1))
        end
    end
    println("  Dehn error done: $n_compute computed $n_cache cached $n_skip skipped")
    return error_series, monodromy_series
end

# ======================================================================
# PLOTTING FUNCTION
# ======================================================================
function plot_twisting_topology(plucker_phase, dehn_error, tracker, folder)
    
    # Early exit if no data
    if (plucker_phase === nothing || isempty(plucker_phase)) && 
    (isempty(dehn_error) || all(isnan, dehn_error))
        @warn "No valid data to plot"
        return nothing
    end

    fig = Figure(size=(1200, 800))

    # Panel A: Fixed unwrapping
    ax1 = Axis(fig[1, 1], title="A: Plücker Phase (Monodromy Winding)")
    if plucker_phase !== nothing && length(plucker_phase) > 0
        # Correct cumulative unwrapping
        unwrapped = copy(plucker_phase)
        shift = 0.0
        for i in 2:length(unwrapped)
            delta = plucker_phase[i] - plucker_phase[i-1]
            shift += (delta > π) ? -2π : (delta < -π) ? 2π : 0
            unwrapped[i] = plucker_phase[i] + shift
        end
        
        lines!(ax1, 1:length(unwrapped), unwrapped, color=:blue, linewidth=2)
        winding = (unwrapped[end] - unwrapped[1]) / (2π)
        text!(ax1, 10.0, maximum(unwrapped)-0.5, text="Winding = $(round(winding, digits=2))")
    end

    # Panel B: Dehn Twist Error
    ax2 = Axis(fig[1, 2], title="B: Dehn Twist Error (Ghost Signals)")
    if !isempty(dehn_error) && any(!isnan, dehn_error)
        valid_idx = findall(!isnan, dehn_error)
        scatter!(ax2, valid_idx, dehn_error[valid_idx], color=:black, markersize=6)
        hlines!(ax2, [2√2], color=:red, linestyle=:dash, linewidth=2)
        
        ghost_idx = findall(x -> !isnan(x) && abs(x - 2√2) < 0.5, dehn_error)
        if !isempty(ghost_idx)
            scatter!(ax2, ghost_idx, dehn_error[ghost_idx], color=:red, marker=:star5, markersize=12)
        end
    end

    # Panel C: Pole Radius
    ax3 = Axis(fig[2, 1], title="C: Twisting Topology (Pole Radius)")

    radius_data = nothing
    if tracker !== nothing
        if hasproperty(tracker, :radius)
            radius_data = tracker.radius
        elseif hasproperty(tracker, :bridge) && hasproperty(tracker.bridge, :pole_radius)
            radius_data = tracker.bridge.pole_radius
        end
    end

    if radius_data !== nothing && !isempty(radius_data)
        radius = Float64.(radius_data)
        lines!(ax3, 1:length(radius), radius, color=:purple, linewidth=2)
        xs = vcat(1:length(radius), length(radius):-1:1)
        ys = vcat(radius, zeros(length(radius)))
        poly!(ax3, Point2f.(xs, ys), color=(:purple, 0.3))
    else
        text!(ax3, 0.5, 0.5, text="No radius data available", color=:red, align=(:center, :center))
    end

    # Panel D: Perversity (Twisting Measure) - Only at blowup events
    ax4 = Axis(fig[2, 2], title="D: Perversity (Twisting Measure)")

    if tracker !== nothing
        # Extract perversity data based on tracker type
        event_indices = nothing
        event_perversities = nothing
        
        if hasproperty(tracker, :perversity) && hasproperty(tracker, :event_times)
            event_indices = findall(x -> x > 0, tracker.perversity)
            event_perversities = tracker.perversity[event_indices]
        elseif hasproperty(tracker, :event_times) && hasproperty(tracker, :event_perversities)
            event_indices = tracker.event_times
            event_perversities = tracker.event_perversities
        elseif hasproperty(tracker, :perversity) && !hasproperty(tracker, :event_times)
            nonzero_idx = findall(x -> x > 0, tracker.perversity)
            if !isempty(nonzero_idx)
                event_indices = nonzero_idx
                event_perversities = tracker.perversity[nonzero_idx]
            end
        end
        
        # Plot if we have event data
        if event_indices !== nothing && !isempty(event_indices)
            perv_values = Float64.(event_perversities)
            
            # Create bar plot at event indices
            barplot!(ax4, event_indices, perv_values, color=:orange, alpha=0.7)
            
            # Add value labels on top of bars
            for (i, idx) in enumerate(event_indices)
                text!(ax4, idx, perv_values[i] + 0.05, 
                    text=string(round(perv_values[i], digits=1)), 
                    fontsize=8, color=:black, align=(:center, :bottom))
            end
            
            # Reference line at 1.0
            hlines!(ax4, [1.0], color=:red, linestyle=:dash, linewidth=2)
            
            # Simple annotation using the max event index (no limits needed)
            n_events = length(event_indices)
            y_max = maximum(perv_values)
            # Just use the last event index for positioning
            text!(ax4, event_indices[end] * 0.7, y_max * 0.85, 
                text="($n_events event$(n_events==1 ? "" : "s"))", 
                color=:gray, fontsize=10)
        else
            text!(ax4, 0.5, 0.5, text="No perversity data (no blowup events)", 
                color=:red, align=(:center, :center))
        end
    else
        text!(ax4, 0.5, 0.5, text="No tracker data available", 
            color=:red, align=(:center, :center))
    end

    # Label axes
    ax4.xlabel = "Snapshot Index"
    ax4.ylabel = "Perversity"

    # Main title
    Label(fig[0, :], "Twisting Topology & Monodromy Analysis", fontsize=18, font=:bold)

    output_path = joinpath(folder, "twisting_topology.png")
    save(output_path, fig)
    println("  ✓ Saved to $output_path")

    return fig
end



# ----------------------------------------------------------------------
# Build prime ideal activity matrix (snapshots × ideals)
# Returns a matrix (rows = snapshots, columns = ideals) of total_support.
# Also returns list of ideal identifiers (e.g., first prime path string).
# ----------------------------------------------------------------------
function build_prime_ideal_matrix(snapshot_files, folder)
    # Optimised: skip snapshots whose prime ideal fingerprint is
    # identical to the previous snapshot (stable zone deduplication).
    # Only reads each unique configuration once.

    all_ideals  = Set{String}()
    ideal_data  = []          # Vector of (ideals_list | nothing=copy_prev)
    n_snap      = length(snapshot_files)
    last_fp     = ""
    last_ideals = []
    n_skipped   = 0

    println("  Building prime ideal matrix ($n_snap snapshots)...")

    for (i, f) in enumerate(snapshot_files)
        snap = load_cached(i)
        sev  = snapshot_severity(snap)

        # Fingerprint using prime_higher_ideals count + top path
        ideals = get(snap, :prime_higher_ideals, [])
        fp_parts = [string(length(ideals))]
        if !isempty(ideals) && haskey(ideals[1], :path)
            push!(fp_parts, join(ideals[1][:path], "→")[1:min(40,end)])
        end
        fp = join(fp_parts, "|")

        # Skip if identical to previous and not high severity
        if fp == last_fp && sev < SEVERITY_THRESHOLD && i > 1 && i % FORCE_INTERVAL != 0
            push!(ideal_data, nothing)   # sentinel = copy previous row
            n_skipped += 1
        else
            for ideal in ideals
                haskey(ideal, :path) || continue
                push!(all_ideals, join(ideal[:path], "→"))
            end
            push!(ideal_data, ideals)
            last_ideals = ideals
            last_fp     = fp
        end

        i % 100 == 0 && @printf("    %d/%d  skipped=%d\n", i, n_snap, n_skipped)
    end

    ideal_list = sort!(collect(all_ideals))
    n_ideals   = length(ideal_list)
    mat        = zeros(Float64, n_snap, n_ideals)

    println("  Unique ideal configurations: $(n_ideals)  skipped rows: $(n_skipped)")

    last_row = zeros(Float64, n_ideals)
    for (i, ideals) in enumerate(ideal_data)
        if ideals === nothing
            # Copy previous row (identical configuration)
            mat[i, :] = last_row
        else
            for ideal in ideals
                haskey(ideal, :path) || continue
                path_str = join(ideal[:path], "→")
                col = findfirst(==(path_str), ideal_list)
                if col !== nothing
                    v = get(ideal, :total_support, 0.0)
                    mat[i, col] += v isa Number ? Float64(v) : 0.0
                end
            end
            last_row = mat[i, :]
        end
    end

    println("  Prime ideal matrix: $(size(mat))  ($n_skipped rows deduplicated)")
    return mat, ideal_list
end

# ----------------------------------------------------------------------
# Extract tracker data directly from A∞ snapshots
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Create proper TrackerResult from extracted data
# ----------------------------------------------------------------------
function create_tracker_result_from_snapshots(snapshot_files, folder, bridge)
    """
    Create a proper TrackerResult object from snapshot data.
    This matches the exact structure expected by SingularityTrackerV2.jl.
    """
    
    # First, check if we have blowup events in the snapshots
    event_times = Int[]
    event_regions = Symbol[]
    event_perversities = Int[]
    pre_radius = Float64[]
    post_radius = Float64[]
    gains = Float64[]
    face_changes = Int[]
    
    for i in eachindex(snapshot_files)
        snap = load_cached(i)
        
        # Check if this is a blowup event (has prime_higher_ideals)
        if haskey(snap, :prime_higher_ideals) && !isempty(snap[:prime_higher_ideals])
            push!(event_times, i)
            
            # Infer region from the snapshot
            region = infer_region_from_snapshot(snap)
            push!(event_regions, region)
            
            # Get max perversity from prime_higher_ideals
            max_perv = 0
            for ideal in snap[:prime_higher_ideals]
                perv = get(ideal, :perversity, 0)
                if perv isa Number
                    max_perv = max(max_perv, Int(round(perv)))
                end
            end
            push!(event_perversities, max_perv)
            
            # Pre and post radius (from bridge data if available)
            pre = i > 1 ? bridge.pole_radius[i-1] : bridge.pole_radius[i]
            post = i < length(snapshot_files) ? bridge.pole_radius[i+1] : bridge.pole_radius[i]
            push!(pre_radius, pre)
            push!(post_radius, post)
            push!(gains, pre - post)
            
            # Face change (flip detection) - FIXED: bridge.flip is a Vector
            fc = 0
            if i > 1 && i <= length(bridge.flip)
                fc = bridge.flip[i]
            end
            push!(face_changes, fc)
        end
    end
    
    # If no blowup events found, create minimal event data
    if isempty(event_times)
        println("  No blowup events detected in snapshots")
        println("  Creating minimal TrackerResult with empty events")
        
        # Use all snapshots as "events" for plotting purposes
        for i in eachindex(snapshot_files)
            push!(event_times, i)
            push!(event_regions, :Unknown)
            push!(event_perversities, 0)
            push!(pre_radius, bridge.pole_radius[max(1, i-1)])
            push!(post_radius, bridge.pole_radius[min(end, i+1)])
            push!(gains, 0.0)
            push!(face_changes, 0)
        end
    end
    
    # Create and return TrackerResult
    return TrackerResult(
        bridge,
        event_times,
        event_regions,
        event_perversities,
        pre_radius,
        post_radius,
        gains,
        face_changes
    )
end

# Helper function to infer region from snapshot
function infer_region_from_snapshot(snap)
    REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :LSX, :sAMY]
    
    # Try explicit region field
    if haskey(snap, :region)
        r = Symbol(String(snap[:region]))
        r in REGIONS && return r
    end
    
    # Scan prime_higher_ideals for region indicators
    if haskey(snap, :prime_higher_ideals)
        for ideal in snap[:prime_higher_ideals]
            # Check closure symbols
            if haskey(ideal, :closure)
                for sym in ideal[:closure]
                    sym_str = String(sym)
                    for r in REGIONS
                        if occursin(String(r), sym_str)
                            return r
                        end
                    end
                end
            end
            # Check path symbols
            if haskey(ideal, :path)
                for sym in ideal[:path]
                    sym_str = String(sym)
                    for r in REGIONS
                        if occursin(String(r), sym_str)
                            return r
                        end
                    end
                end
            end
        end
    end
    
    # Scan m6 keys as fallback
    if haskey(snap, :m6)
        for (k, v) in pairs(snap[:m6])
            k_str = String(k)
            for r in REGIONS
                if occursin(String(r), k_str) && (v isa Number && !iszero(v))
                    return r
                end
            end
        end
    end
    
    return :Unknown
end
# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------
function main()
    # Command line arguments
    folder = length(ARGS) >= 1 ? ARGS[1] : "."
    max_files = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0
    
    # Track if we're using a temp directory for cleanup
    using_temp_dir = false
    test_dir = ""

    println("="^80)
    println("FULL DERIVED (2,1)-STACK PIPELINE")
    println("Folder: $folder")
    if max_files > 0
        println("Test mode: limiting to first $max_files JSON files.")
    else
        println("Full mode: processing all JSON files.")
    end
    println("="^80)

    # 1. List and filter only ainf_export_*.json
    # List all matching files
    all_files = filter(readdir(folder)) do f
        occursin(r"ainf_export_(?:[A-Za-z0-9]+_)?\d+(?:\.\d+)\.json", basename(f))
    end

    # Sort by timestamp (extracts number from filename)
    sort!(all_files, by = f -> parse_timestamp(f))

    println("Found $(length(all_files)) A∞ snapshot files.")

    # Handle test mode with file limiting
    if max_files > 0 && length(all_files) > max_files
        println("Test mode: limiting to first $max_files files.")
        # Create temp folder
        test_dir = joinpath(folder, "test_subset_$(max_files)")
        isdir(test_dir) || mkdir(test_dir)
        using_temp_dir = true
        
        # Clean previous content
        for f in readdir(test_dir)
            if endswith(f, ".json")
                rm(joinpath(test_dir, f), force=true)
            end
        end
        
        # Copy selected files
        selected_files = all_files[1:max_files]
        for f in selected_files
            cp(joinpath(folder, f), joinpath(test_dir, f), force=true)
        end
        
        # Update folder to test_dir and re-list files
        folder = test_dir
        all_files = filter(f -> occursin(r"ainf_export_\d+(?:\.\d+)?\.json", basename(f)), readdir(folder))
        sort!(all_files, by = f -> parse_timestamp(f))
    elseif max_files > 0 && length(all_files) > 0
        # If we have fewer files than max_files, just use what we have
        all_files = all_files[1:min(max_files, length(all_files))]
    end
    
    println("Processing $(length(all_files)) A∞ snapshot files.\n")

    # 2. Load ALL snapshots into memory once — eliminates 4 redundant file passes
    println("--- Loading snapshot cache ---")
    precomp_fps, precomp_sevs = init_snapshot_cache!(all_files, folder)
    println()

    # Run bridge (navigator + spectral)
    println("--- Running IharaAssociahedronBridgeV2 ---")
    bridge = run_bridge!(folder)
    println("Unified zeta saved to unified_zeta.json\n")

    # 3. Run singularity tracker (or extract from snapshots)

    println("--- Running SingularityTrackerV2 ---")

    # First try to run the normal tracker
    tracker = nothing
    try
        tracker = run_tracker!(folder)
        if tracker !== nothing && !isempty(tracker.event_times)
            println("  ✓ Successfully loaded tracker with $(length(tracker.event_times)) events")
            event_table(tracker)
        else
            println("  Tracker has no events, will extract from snapshots")
            tracker = nothing
        end
    catch e
        println("  Could not run normal tracker: $e")
        tracker = nothing
    end

    # If normal tracker failed or has no data, create from snapshots
    if tracker === nothing || (tracker isa TrackerResult && isempty(tracker.event_times))
        println("  Creating tracker from snapshot data...")
        tracker = create_tracker_result_from_snapshots(all_files, folder, bridge)
        println("  Created tracker with $(length(tracker.event_times)) events")
        
        # Try to display event table (may fail for extracted data)
        try
            event_table(tracker)
        catch e
            println("  Note: event_table display not available: $e")
        end
    end

    # Generate plots (ONCE!)
    try
        save("tracker_plots.png", plot_tracker(tracker))
        save("bridge_pole_radius.png", plot_pole_radius(bridge))
        println("  ✓ Saved tracker plots")
    catch e
        println("  Could not save tracker plots: $e")
    end

    # DEBUG: Check tracker contents
    println("\n=== TRACKER DEBUG ===")
    println("tracker type: ", typeof(tracker))
    if tracker !== nothing
        for field in fieldnames(typeof(tracker))
            if hasproperty(tracker, field)
                val = getproperty(tracker, field)
                println("  .$field: ", val === nothing ? "nothing" : 
                        (isa(val, AbstractVector) ? "$(length(val)) elements" : "$val"))
            end
        end
    end
    println("===================\n")

    # 4. Run Schober navigator (categorical chambers/walls)
    println("\n--- Running SchoberNavigatorV2 ---")

    # Stability summary (already computed by init_snapshot_cache!)
    n_unique_fps = length(unique(precomp_fps))
    pct_stable   = round(100.0*(1 - n_unique_fps/max(length(precomp_fps),1)), digits=1)
    @printf("  Fingerprint summary: %d unique / %d total (%.1f%% stable)\n",
            n_unique_fps, length(precomp_fps), pct_stable)

    full_paths = [joinpath(folder, f) for f in all_files]
    schober_state = run_schober_path(full_paths)
    summarize_path(schober_state)
    write_chamber_table(schober_state, "schober_chambers.tsv")
    write_wall_table(schober_state, "schober_walls.tsv")

    # 5. Run Rees‑Grassmann bridge (crisis generators, charts, flips)
    println("\n--- Running ReesGrassmannBridge ---")
    ReesGrassmannBridge.analyze_fibered_rees(; snapshot_folder=folder, use_real_metrics=true)

    # 6. Compute missing data for combined chart
    println("\n--- Computing extra data for combined chart ---")
    # Plücker phase
    plucker_phase = compute_plucker_phase()
    if plucker_phase !== nothing
        open("plucker_phase.json", "w") do f
            JSON3.write(f, Dict("phase" => plucker_phase))
        end
        println("Saved Plücker phase to plucker_phase.json")
    else
        @warn "Plücker phase not saved (plucker_trajectory.json missing)."
    end

    # Dehn twist error (monodromy coherence)
    dehn_error, monodromy_matrices = compute_dehn_error(all_files, folder;
        precomputed_fps=precomp_fps, precomputed_sevs=precomp_sevs)

    # Save as CSV (handles NaN values properly)
    dehn_error_df = DataFrame(
        snapshot_index = 1:length(dehn_error),
        dehn_error = dehn_error,  # NaN will become `missing` in DataFrame
        timestamp = [parse_timestamp(f) for f in all_files],
        filename = all_files
    )
    CSV.write("dehn_error.csv", dehn_error_df)
    println("Saved Dehn twist error series to dehn_error.csv")

    # Also save JSON version (convert NaN to nothing)
    error_for_json = [isnan(x) ? nothing : x for x in dehn_error]
    open("dehn_error.json", "w") do f
        JSON3.write(f, Dict("error" => error_for_json))
    end
    println("Saved JSON version to dehn_error.json (NaN → null)")

    # Print summary
    valid_errors = dehn_error[.!isnan.(dehn_error)]
    if !isempty(valid_errors)
        println("  Valid errors: $(length(valid_errors))/$(length(dehn_error))")
        println("  Max error: $(maximum(valid_errors))")
        println("  Min error: $(minimum(valid_errors))")
    end


    # Prime ideal activity matrix
    prime_mat, ideal_labels = build_prime_ideal_matrix(all_files, folder)
    # Save as CSV and JSON
    prime_df = DataFrame(prime_mat, Symbol.(ideal_labels))
    CSV.write("prime_ideal_activity.csv", prime_df)
    open("prime_ideal_activity.json", "w") do f
        JSON3.write(f, Dict("labels" => ideal_labels, "matrix" => prime_mat))
    end
    println("Saved prime ideal activity matrix (size $(size(prime_mat))) to prime_ideal_activity.csv / .json")

    # 7. Read tubing flips and chart flips from existing outputs
    println("\n--- Loading flip data from outputs ---")

    # Load chamber/wall flips from Schober navigator
    if isfile("schober_chambers.tsv")
        chamber_data = CSV.read("schober_chambers.tsv", DataFrame)
        println("  ✓ Loaded chamber flip data: $(nrow(chamber_data)) entries")
    end

    if isfile("schober_walls.tsv")
        wall_data = CSV.read("schober_walls.tsv", DataFrame)
        println("  ✓ Loaded wall flip data: $(nrow(wall_data)) entries")
    end

    # Load Rees-Grassmann flip data
    rees_output = joinpath(folder, "rees_grassmann_flips.json")
    if isfile(rees_output)
        flip_data = JSON3.read(read(rees_output, String))
        println("  ✓ Loaded Rees-Grassmann flips: $(length(flip_data)) events")
    else
        @warn "Rees-Grassmann flips not found at $rees_output"
    end

    # Optionally combine and analyze flip statistics
    # Compare chamber transitions with Rees chart transitions
    flip_summary = Dict(
        "total_chamber_transitions" => isfile("schober_chambers.tsv") ? nrow(chamber_data) : 0,
        "total_wall_crossings" => isfile("schober_walls.tsv") ? nrow(wall_data) : 0,
        "rees_flips_available" => isfile(rees_output)
    )
    
    # ===== Plot twisting topology =====
    println("\n--- Generating twisting topology visualization ---")

    # Make sure we have plucker_phase
    if !@isdefined(plucker_phase) || plucker_phase === nothing
        plucker_phase = compute_plucker_phase()
    end

    # Call with CORRECT number of arguments (4, not 5)
    plot_twisting_topology(plucker_phase, dehn_error, tracker, folder)

    open("flip_summary.json", "w") do f
        JSON3.write(f, flip_summary)
    end
    println("  ✓ Saved flip summary to flip_summary.json")

    println("\n"^2)
    println("ALL DONE. Output files:")
    println("  tracker_plots.png, bridge_pole_radius.png")
    println("  schober_chambers.tsv, schober_walls.tsv")
    println("  rees_grassmann_analysis.png")
    println("  plucker_phase.json, dehn_error.json, prime_ideal_activity.csv/.json")
    println("  unified_zeta.json, plucker_zeta_dense.json (already present)")
    println("="^80)



    # Check dehn_error values
    df = CSV.read("dehn_error.csv", DataFrame)
    println("Dehn error values:")
    println(df.dehn_error)

    # Check if any reached 2√2 (2.828)
    println("\nAny ghost signals? ", any(x -> !ismissing(x) && abs(x - 2√2) < 0.5, df.dehn_error))

    # Check max error
    println("Max error: ", maximum(skipmissing(df.dehn_error)))
    
    # Return info for cleanup
    return (using_temp_dir, test_dir)
end

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    using_temp_dir, test_dir = main()
    if using_temp_dir && isdir(test_dir)
        #rm(test_dir, recursive=true)
        println("Don't forget to Remove temporary test directory: $test_dir, after seeing .png files")
    end
end
