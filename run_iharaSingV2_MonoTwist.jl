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

# Include modules (adjust paths if needed)
include("OnlineAssociahedronNavigatorV3.jl")
include("IharaAssociahedronBridgeV2.jl")
include("SingularityTrackerV2.jl")
include("SchobarNavigatorV2.jl")
include("ReesGrassmannBridge.jl")

using .OnlineAssociahedronNavigatorV3
using .IharaAssociahedronBridgeV2
using .SingularityTracker
using .SchoberNavigatorV2
using .ReesGrassmannBridge

const REGIONS = [:CA1sp, :BLA, :HY, :HPF, :sAMY, :LA]
load_json(file) = JSON3.read(read(file,String))

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
function compute_dehn_error(snapshot_files, folder)
    """
    Compute cumulative monodromy error (Dehn twist error).
    For Gr(2,4), perfect coherence gives error = 0.
    Ghost signal appears when error → 3.
    """
    # Gr(2,4) embedding requires 4 dimensions
    N = 4  # Fixed for Grassmannian, not length(REGIONS)
    M_total = Matrix{Float64}(I, N, N)
    error_series = Float64[]
    monodromy_series = []
    
    for (i, f) in enumerate(snapshot_files)
        filepath = joinpath(folder, f)
        snap = load_json(filepath)

        # Diagnostic: print first snapshot keys
        if i == 1
            println("Sample snapshot keys: ", keys(snap))
            println("Has prime_paths? ", haskey(snap, :prime_paths))
            println("Has m4? ", haskey(snap, :m4))
            println("prime_paths length: ", length(get(snap, :prime_paths, [])))
        end
        
        try
            M = monodromy(snap)
            
            # Cumulative product (order: later snapshots multiply on left)
            M_total = M * M_total
            
            # Coherence error: distance from identity
            err = norm(M_total - I)
            push!(error_series, err)
            push!(monodromy_series, M_total)
            
            # Detect ghost signal (error ≈ 3.0)
            if err > 2.5 && err < 3.5
                @info "Ghost signal detected at snapshot $i: error = $err"
                @info "   Monodromy eigenvalues: $(eigvals(M_total))"
                @info "   Coherence breakdown at t = $(parse_timestamp(f))"
            end
        catch e
            # Brief error message - no snapshot dumping!
            @warn "Failed to compute monodromy for $f: $(typeof(e).name.name)"
            # Optional: minimal debug info
            if i == 1  # Only for first file
                println("  Minimal debug: has prime_paths? ", haskey(snap, :prime_paths))
            end
            push!(error_series, NaN)
            push!(monodromy_series, nothing)
        end
    end
    
    return error_series, monodromy_series
end

# ----------------------------------------------------------------------
# Build prime ideal activity matrix (snapshots × ideals)
# Returns a matrix (rows = snapshots, columns = ideals) of total_support.
# Also returns list of ideal identifiers (e.g., first prime path string).
# ----------------------------------------------------------------------
function build_prime_ideal_matrix(snapshot_files, folder)
    # First pass: collect all unique prime ideal identifiers (e.g., first path element)
    all_ideals = Set{String}()
    ideal_data = Vector{Dict}[]   # store per snapshot list of ideals
    for f in snapshot_files
        snap = load_json(joinpath(folder, f))
        ideals = get(snap, :prime_higher_ideals, [])
        push!(ideal_data, ideals)
        for ideal in ideals
            # Use the first symbol of the closure or the path as identifier
            path_str = join(ideal[:path], "→")  # Use Symbol, not String
            push!(all_ideals, path_str)
        end
    end
    ideal_list = collect(all_ideals)
    sort!(ideal_list)
    n_ideals = length(ideal_list)
    n_snap = length(snapshot_files)
    # Build matrix: rows = snapshots, columns = ideals
    mat = zeros(Float64, n_snap, n_ideals)
    for (i, ideals) in enumerate(ideal_data)
        for ideal in ideals
            path_str = join(ideal["path"], "→")
            col = findfirst(==(path_str), ideal_list)
            if col !== nothing
                mat[i, col] += ideal["total_support"]
            end
        end
    end
    return mat, ideal_list
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
    all_files = filter(f -> occursin(r"ainf_export_\d+(?:\.\d+)?\.json", basename(f)), readdir(folder))
    sort!(all_files, by = f -> parse_timestamp(f))
    
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

    # 2. Run bridge (navigator + spectral)
    println("--- Running IharaAssociahedronBridgeV2 ---")
    bridge = run_bridge!(folder)
    println("Unified zeta saved to unified_zeta.json\n")

    # 3. Run singularity tracker
    println("--- Running SingularityTrackerV2 ---")
    tracker = run_tracker!(folder)
    event_table(tracker)
    save("tracker_plots.png", plot_tracker(tracker))
    save("bridge_pole_radius.png", plot_pole_radius(bridge))

    # 4. Run Schober navigator (categorical chambers/walls)
    println("\n--- Running SchoberNavigatorV2 ---")
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
    dehn_error, monodromy_matrices = compute_dehn_error(all_files, folder)

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
    
    # Return info for cleanup
    return (using_temp_dir, test_dir)
end

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    using_temp_dir, test_dir = main()
    if using_temp_dir && isdir(test_dir)
        rm(test_dir, recursive=true)
        println("Removed temporary test directory: $test_dir")
    end
end

