using JSON3, LinearAlgebra, Statistics, CairoMakie

# ----------------------------------------------------------------------
# Load unified zeta magnitudes (from run_iharaSingV2.jl)
# ----------------------------------------------------------------------
function load_unified_zeta(file="unified_zeta.json")
    if !isfile(file)
        error("Unified zeta file not found: $file")
    end
    data = JSON3.read(read(file, String))
    if haskey(data, "magnitude")
        mag = Float64.(data["magnitude"])
    elseif haskey(data, "log_magnitude")
        mag = exp10.(Float64.(data["log_magnitude"]))
    else
        error("Unified zeta JSON missing both 'magnitude' and 'log_magnitude'")
    end
    return mag
end

# ----------------------------------------------------------------------
# Load dense Plücker zeta with snapshot indices
# ----------------------------------------------------------------------
function load_dense_plucker(file="plucker_zeta_dense.json")
    if !isfile(file)
        error("Dense Plücker zeta file not found: $file")
    end
    data = JSON3.read(read(file, String))
    if !haskey(data, "snapshot_indices")
        error("Dense Plücker JSON missing 'snapshot_indices'")
    end
    mags = Float64.(data["magnitudes"])
    indices = Int.(data["snapshot_indices"])
    return mags, indices
end

# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------
function main()
    # Load data
    unified_mag = load_unified_zeta()
    dense_mags, snap_indices = load_dense_plucker()

    println("Number of unified zeta points: ", length(unified_mag))
    println("Number of snapshot indices:   ", length(snap_indices))

    if length(unified_mag) != length(snap_indices)
        error("Length mismatch: unified_zeta ($(length(unified_mag))) vs snapshot_indices ($(length(snap_indices)))")
    end

    # Extract Plücker magnitude at the exact snapshot steps
    plucker_at_snapshots = Float64[]
    for idx in snap_indices
        if 1 ≤ idx ≤ length(dense_mags)
            push!(plucker_at_snapshots, dense_mags[idx])
        else
            error("Snapshot index $idx out of range (1..$(length(dense_mags)))")
        end
    end

    # Compute correlation on log‑log scale
    x = log10.(max.(unified_mag, 1e-12))
    y = log10.(max.(plucker_at_snapshots, 1e-12))
    valid = isfinite.(x) .& isfinite.(y)

    if sum(valid) < 2
        error("Not enough valid points for correlation (need at least 2).")
    end

    corr_val = cor(x[valid], y[valid])

    # Linear regression for log‑log data
    X = hcat(ones(sum(valid)), x[valid])
    coeff = X \ y[valid]          # coeff[1] = intercept, coeff[2] = slope
    slope = coeff[2]
    intercept = coeff[1]

    println("\n========== CORRELATION SUMMARY ==========")
    println("Number of aligned points: $(sum(valid))")
    println("Pearson correlation (log‑log): $(round(corr_val, digits=4))")
    println("Slope (power‑law exponent): $(round(slope, digits=4))")
    println("Intercept: $(round(intercept, digits=4))")
    println("==========================================\n")

    # ------------------------------------------------------------------
    # 1. Scatter plot with linear fit
    # ------------------------------------------------------------------
    fig1 = Figure(size=(800, 600))
    ax1 = Axis(fig1[1,1],
               title="Aligned Unified Zeta vs Plücker Zeta",
               xlabel="log10(unified zeta magnitude)",
               ylabel="log10(Plücker ζ magnitude)")
    scatter!(ax1, x[valid], y[valid], color=:purple, markersize=6)

    # Linear fit line
    x_line = range(minimum(x[valid]), maximum(x[valid]), length=100)
    y_line = intercept .+ slope .* x_line
    lines!(ax1, x_line, y_line, color=:gray, linestyle=:dash)

    save("aligned_zeta_correlation.png", fig1)
    println("Saved correlation plot to aligned_zeta_correlation.png")

    # ------------------------------------------------------------------
    # 2. Time‑series plot (snapshot index vs magnitudes, log y‑axis)
    # ------------------------------------------------------------------
    fig2 = Figure(size=(1000, 500))
    ax2 = Axis(fig2[1,1],
               title="Unified Zeta vs Plücker Zeta over Snapshots",
               xlabel="Snapshot index",
               ylabel="Magnitude (log scale)",
               yscale=log10)

    # Unified zeta (blue line)
    lines!(ax2, 1:length(unified_mag), max.(unified_mag, 1e-12),
           color=:blue, linewidth=2, label="Unified ζ")

    # Plücker zeta at snapshots (red line connecting points)
    x_vals = 1:length(plucker_at_snapshots)
    y_vals = max.(plucker_at_snapshots, 1e-12)
    lines!(ax2, x_vals, y_vals, color=:red, linewidth=1.5, label="Plücker ζ (aligned)")
    # Also keep scatter markers for clarity
    scatter!(ax2, x_vals, y_vals, color=:red, markersize=4)

    axislegend(ax2)
    save("aligned_timeseries.png", fig2)
    println("Saved time‑series plot to aligned_timeseries.png")
end

# ----------------------------------------------------------------------
# Run the analysis
# ----------------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
