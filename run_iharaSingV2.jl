#=
Your IharaAssociahedronBridge.jl and SingularityTracker.jl are ambitious and creative – 
they attempt to link the algebraic obstruction data (prime paths, cup product, Gerstenhaber) 
to graph spectral theory (Ihara zeta proxy) and event detection.
=#
using JSON3
using CairoMakie
using LinearAlgebra
include("IharaAssociahedronBridgeV2.jl")
include("SingularityTrackerV2.jl")
include("SchobarNavigatorV2.jl")
using .SingularityTracker
include("ReesBlowupHybrid.jl")
using .ReesBlowupHybrid
using .SchoberNavigatorV2
using .IharaAssociahedronBridgeV2: effective_adjacency

# ------------------------------------------------------------
# Helper: load prime zeta (Plücker zeta) from JSON
# ------------------------------------------------------------

load_json(file) = JSON3.read(read(file, String))

function load_prime_zeta(file="prime_zeta.json")
    if !isfile(file)
        return nothing, Float64[]
    end
    data = JSON3.read(read(file, String))
    # data["values"] is list of {"real":..., "imag":...}
    zeta_vals = [Complex(entry["real"], entry["imag"]) for entry in data["values"]]
    # The transition times are already saved separately; we assume they are the same as in transition_times.json
    return zeta_vals, Float64.(data["transition_times"])
end

function parse_timestamp(fname)
    m = match(r"([0-9]+\.[0-9]+)", basename(fname))
    return m === nothing ? 0.0 : parse(Float64, m.captures[1])
end

# ------------------------------------------------------------
# Load transition times (as before)
# ------------------------------------------------------------
function load_transition_times(file="transition_times.json")
    if !isfile(file)
        return Float64[]
    end
    data = JSON3.read(read(file, String))
    return Float64.(data)
end

# ------------------------------------------------------------
# Find file indices corresponding to transition times
# ------------------------------------------------------------
function nearest_file_index(times, filenames)
    ts = Float64[]
    for f in filenames
        m = match(r"([0-9]+\.[0-9]+)\.json", basename(f))
        if m !== nothing
            push!(ts, parse(Float64, m.captures[1]))
        else
            push!(ts, NaN)
        end
    end
    indices = Int[]
    for t in times
        idx = argmin(abs.(ts .- t))
        push!(indices, idx)
    end
    return indices
end

# ----------------------------------------------------------------------
# Correlation plot: Ihara pole radius vs |Plücker ζ|
# ----------------------------------------------------------------------
function plot_pole_plucker_correlation(pole_radius, plucker_mags, transition_indices)
    fig = Figure(size=(1000, 500))

    # Left: time series with two y‑axes
    ax1 = Axis(fig[1,1], title="Ihara Pole Radius and |Plücker ζ|", xlabel="File index")
    lines!(ax1, 1:length(pole_radius), pole_radius, color=:blue, label="Ihara pole radius")
    ax2 = Axis(fig[1,2], ylabel="|Plücker ζ|")
    scatter!(ax2, transition_indices, plucker_mags, color=:red, markersize=8, label="|Plücker ζ|")
    axislegend(ax1)
    axislegend(ax2)

    # Right: scatter plot
    ax3 = Axis(fig[1,3], title="Correlation", xlabel="Ihara pole radius", ylabel="|Plücker ζ|")
    # Only plot where we have both values
    valid = (1:length(pole_radius)) .∈ Ref(transition_indices)
    radii = pole_radius[valid]
    mags = plucker_mags
    scatter!(ax3, radii, mags, color=:purple, markersize=6)
    # optional: add linear fit
    if length(radii) > 1
        coef = polyfit(radii, mags, 1)
        x_vals = range(minimum(radii), maximum(radii), length=100)
        lines!(ax3, x_vals, coef[1] .+ coef[2] * x_vals, color=:gray, linestyle=:dash)
    end

    save("pole_plucker_correlation.png", fig)
    println("Saved correlation plot to pole_plucker_correlation.png")
end

function plot_correlation(pole_radius, plucker_mags, transition_indices)
    fig = Figure(size=(900,500))
    ax1 = Axis(fig[1,1], title="Ihara Pole Radius vs Plücker Zeta Magnitude",
               xlabel="File index", ylabel="Pole radius")
    lines!(ax1, 1:length(pole_radius), pole_radius, color=:blue, label="Ihara pole radius")
    ax2 = Axis(fig[1,2], ylabel="|Plücker ζ|")
    scatter!(ax2, transition_indices, plucker_mags, color=:red, markersize=6, label="|Plücker ζ|")
    axislegend(ax1)
    axislegend(ax2)
    save("pole_plucker_correlation.png", fig)
end

# ------------------------------------------------------------
# Plot Ihara pole radius with Plücker zeta magnitude as second axis
# ------------------------------------------------------------
function plot_pole_radius_with_plucker_zeta(B, pole_radius, plucker_zeta_mags, transition_indices)
    fig = Figure(size=(900,500))
    ax1 = Axis(fig[1,1], title="Ihara Pole Radius with Plücker Zeta Magnitude")
    lines!(ax1, 1:length(pole_radius), pole_radius, linewidth=2, label="Ihara pole radius")
    
    # Second axis for Plücker zeta magnitude
    ax2 = Axis(fig[1,2], title="Plücker Zeta Magnitude", ylabel="|ζ|")
    if !isempty(plucker_zeta_mags)
        # Align with transition indices (x = file index)
        x_vals = transition_indices
        y_vals = plucker_zeta_mags
        scatter!(ax2, x_vals, y_vals, color=:blue, markersize=6, label="|Plücker ζ|")
        # Optionally add lines connecting them
        # lines!(ax2, x_vals, y_vals, color=:blue, linestyle=:dash)
    end
    axislegend(ax1)
    axislegend(ax2)
    fig
end

# ------------------------------------------------------------
# Main driver
# ------------------------------------------------------------
# ------------------------------------------------------------
# Create a test subdirectory with a limited number of files
# ------------------------------------------------------------
folder = "./"
all_original_json = filter(f -> occursin(r"ainf_export_(?:\w+_)?\d+(?:\.\d+)?\.json", basename(f)), readdir(folder))
sort!(all_original_json)

max_files = 3   # test limit
test_files = all_original_json[1:min(max_files, end)]

test_dir = joinpath(folder, "test_subset")
isdir(test_dir) || mkdir(test_dir)

# Remove any existing JSON files in test_dir (to avoid leftovers)
for f in readdir(test_dir)
    if endswith(lowercase(f), ".json")
        rm(joinpath(test_dir, f))
    end
end

# Copy the selected files
for f in test_files
    src = joinpath(folder, f)
    dst = joinpath(test_dir, f)
    if isfile(src)
        cp(src, dst, force=true)
    else
        @warn "Source file $src does not exist – skipping"
    end
end

# Now set folder to test_dir and reread the (now clean) file list
# folder = test_dir
all_files = filter(f -> occursin(r"ainf_export_(?:\w+_)?\d+(?:\.\d+)?\.json", basename(f)), readdir(folder))
sort!(all_files)

println("Using test directory with $(length(all_files)) files:")
for f in all_files
    println("  $f")
end

T = run_tracker!(folder)                # runs navigator and bridge from SingularityTrackerV2
B = T.bridge

# Load transition times and prime zeta
transition_times = load_transition_times()
prime_zeta_vals, plucker_times = load_prime_zeta()

if !isempty(transition_times) && !isempty(prime_zeta_vals)
    files = filter(f -> occursin(r"ainf_export_(?:\w+_)?\d+(?:\.\d+)?\.json", basename(f)), readdir(folder))
    sort!(files)
    indices = nearest_file_index(transition_times, files)
    
    # Compute magnitude of Plücker zeta values
    plucker_mags = abs.(prime_zeta_vals)
    
    # Create the combined plot
    fig = plot_pole_radius_with_plucker_zeta(B, B.pole_radius, plucker_mags, indices)
    save("pole_radius_with_plucker_zeta.png", fig)
    println("Saved combined plot to pole_radius_with_plucker_zeta.png")
else
    # Fallback: just plot the standard pole radius
    save("bridge_pole_radius.png", IharaAssociahedronBridgeV2.plot_pole_radius(B))
end

# Also produce standard tracker outputs
event_table(T)
save("tracker_plots.png", plot_tracker(T))
save("bridge_pole_radius.png", IharaAssociahedronBridgeV2.plot_pole_radius(B))
println("Done. Saved: tracker_plots.png, bridge_pole_radius.png")

########################################
# REES ALGEBRA BLOW UP
########################################

B_rees = run_blowup!(folder)
print_events(B_rees)
save("rees_generators.png", plot_generators(B_rees))
save("rees_divisor.png", plot_divisor(B_rees))

# ------------------------------------------------------------------
# Load Plücker zeta and correlate
# ------------------------------------------------------------------
zeta_vals, plucker_times = load_prime_zeta()
if !isempty(zeta_vals) && length(B.pole_radius) == length(all_files)
    plucker_mags = abs.(zeta_vals)
    # Map transition times to file indices (align if times are timestamps)
    indices = nearest_file_index(plucker_times, all_files)
    # Clip to valid range
    indices = filter(i -> 1 ≤ i ≤ length(B.pole_radius), indices)
    plucker_mags_aligned = plucker_mags[1:length(indices)]
    if !isempty(indices)
        plot_pole_plucker_correlation(B.pole_radius, plucker_mags_aligned, indices)
    else
        @warn "No alignment between Plücker transition times and file indices."
    end
else
    @warn "Plücker zeta data missing or length mismatch; skipping correlation plot."
end

# ------------------------------------------------------------------
# Optional: run SchoberNavigator for categorical chambers
# ------------------------------------------------------------------
println("\n--- Running SchoberNavigator (post‑processing) ---")
schober_state = run_schober_path(all_files)
summarize_path(schober_state)
write_chamber_table(schober_state, "schober_chambers.tsv")
write_wall_table(schober_state, "schober_walls.tsv")

println("\nAll done! Output files: ")
println("  tracker_plots.png, bridge_pole_radius.png, pole_plucker_correlation.png")
println("  chambers.tsv, walls.tsv, schober_chambers.tsv, schober_walls.tsv")

# ------------------------------------------------------------------
# Compute unified sheaf zeta from bridge’s effective adjacency
# ------------------------------------------------------------------
println("\n--- Computing unified sheaf zeta (determinant) ---")
unified_zeta_mag = Float64[]
unified_zeta_log = Float64[]
# We need to recompute A for each snapshot (or read from bridge? bridge has only radius)
# Instead, we recompute effective_adjacency for each file (same as bridge did)
for f in all_files
    fullpath = joinpath(folder, f)
    if !isfile(fullpath)
        @warn "File $fullpath not found – skipping"
        continue
    end
    if !endswith(f, ".json")
        continue
    end
    snap = load_json(joinpath(folder, f))
    A = effective_adjacency(snap)
    # Choose parameter t = 0.9 / ρ(A) to stay inside convergence
    evals = eigvals(Matrix(A))
    ρ = maximum(abs.(evals))
    t = ρ > 0 ? 0.9 / ρ : 0.5
    # Compute determinant
    n = size(A,1)
    M = I - t * A
    det_val = det(M)
    mag = abs(1.0 / det_val)
    logmag = -log(abs(det_val))
    push!(unified_zeta_mag, mag)
    push!(unified_zeta_log, logmag)
end
# Save unified zeta time series
unified_data = Dict(
    "times" => [parse_timestamp(f) for f in all_files],
    "log_magnitude" => unified_zeta_log,
    "magnitude" => unified_zeta_mag
)
open("unified_zeta.json", "w") do io
    JSON3.write(io, unified_data)
end
println("Unified zeta saved to unified_zeta.json")


# ------------------------------------------------------------------
# Correlation plot of unified zeta vs Plücker zeta (if available)
# ------------------------------------------------------------------
if isfile("prime_zeta.json")
    pz_data = JSON3.read(read("prime_zeta.json", String))
    pz_mags = [abs(entry["real"] + im*entry["imag"]) for entry in pz_data["values"]]
    pz_times = Float64.(pz_data["transition_times"])
    # Align times with unified zeta (by nearest timestamp)
    unified_times = unified_data["times"]
    aligned_mags = Float64[]
    for pt in pz_times
        idx = argmin(abs.(unified_times .- pt))
        push!(aligned_mags, unified_zeta_mag[idx])
    end
    # Scatter plot
    fig = Figure(size=(600,600))
    ax = Axis(fig[1,1], title="Unified Zeta vs Plücker Zeta", xlabel="|Unified ζ|", ylabel="|Plücker ζ|")
    scatter!(ax, aligned_mags, pz_mags, color=:purple, markersize=5)
    save("zeta_correlation.png", fig)
    println("Correlation plot saved to zeta_correlation.png")
else
    println("prime_zeta.json not found – skipping correlation plot.")
end

println("\n"^2)
println("ALL DONE. Output files:")
println("  tracker_plots.png, bridge_pole_radius.png, schober_chambers.tsv, schober_walls.tsv")
println("  unified_zeta.json, zeta_correlation.png (if prime_zeta.json existed)")
println("="^80)


