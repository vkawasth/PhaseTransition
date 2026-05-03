module ReesGrassmannBridge

using JSON3, LinearAlgebra, Statistics, Plots

# ----------------------------------------------------------------------
# Load unified zeta from JSON (produced by run_iharaSingV2.jl)
# ----------------------------------------------------------------------
function load_unified_zeta(file="unified_zeta.json")
    data = JSON3.read(read(file, String))
    return Float64.(data["magnitude"])
end

# ----------------------------------------------------------------------
# Load dense Plücker zeta and snapshot indices
# ----------------------------------------------------------------------
function load_dense_plucker_with_indices(file="plucker_zeta_dense.json")
    data = JSON3.read(read(file, String))
    mags = Float64.(data["magnitudes"])
    if !haskey(data, "snapshot_indices")
        error("Missing 'snapshot_indices' in $file. Run Python with corrected saving.")
    end
    indices = Int.(data["snapshot_indices"])
    return mags, indices
end

# ----------------------------------------------------------------------
# Load best tubings (optional)
# ----------------------------------------------------------------------
function load_best_tubings(file="best_tubings.json")
    if !isfile(file)
        @warn "$file not found; using dummy tubing IDs."
        return [string(i) for i in 1:100]
    end
    data = JSON3.read(read(file, String))
    return data["tubings"]
end

# ----------------------------------------------------------------------
# Extract metrics from a single A∞ snapshot JSON
# Returns: (m6_total, cup_total, gerstenhaber_total)
# ----------------------------------------------------------------------
function extract_metrics_from_snapshot(file)
    snap = JSON3.read(read(file, String))

    # Total m₆ mass
    m6 = 0.0
    if haskey(snap, "m6")
        for (_, coeff_dict) in snap["m6"]
            for (_, coeff) in coeff_dict
                m6 += abs(Float64(coeff))
            end
        end
    end

    # Total cup product magnitude
    cup_total = 0.0
    if haskey(snap, "cup_product")
        for entry in snap["cup_product"]
            cup_total += abs(Float64(entry["coeff"]))
        end
    end

    # Total Gerstenhaber bracket magnitude
    gersten_total = 0.0
    if haskey(snap, "gerstenhaber")
        for entry in snap["gerstenhaber"]
            gersten_total += abs(Float64(entry["coeff"]))
        end
    end

    return m6, cup_total, gersten_total
end

# ----------------------------------------------------------------------
# Load all metrics from A∞ snapshots (aligned with snapshot order)
# ----------------------------------------------------------------------
function load_metrics_from_snapshots(snapshot_files)
    m6_vals = Float64[]
    cup_vals = Float64[]
    gersten_vals = Float64[]
    for f in snapshot_files
        m6, cup, gersten = extract_metrics_from_snapshot(f)
        push!(m6_vals, m6)
        push!(cup_vals, cup)
        push!(gersten_vals, gersten)
    end
    return m6_vals, cup_vals, gersten_vals
end

# ----------------------------------------------------------------------
# Crisis ideal (7 generators: m6, zeta_I, zeta_W, cup, gersten, prime_support, entropy)
# prime_support and entropy are placeholders; you can replace later.
# ----------------------------------------------------------------------
function compute_ideal(m6, zeta_I, zeta_W, cup, gersten, prime_support, entropy)
    return (m6=m6, zeta_I=zeta_I, zeta_W=zeta_W,
            cup=cup, gersten=gersten,
            prime_support=prime_support, entropy=entropy)
end

function active_chart(ideal)
    # Index 1..7 (order as above)
    vals = [ideal.m6, ideal.zeta_I, ideal.zeta_W,
            ideal.cup, ideal.gersten,
            ideal.prime_support, ideal.entropy]
    return argmax(vals)
end

function mismatch(zeta_I, zeta_W)
    return abs(zeta_I - zeta_W)
end

# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------
function analyze_fibered_rees(;
        norm_type="minmax",
        snapshot_folder=".",
        use_real_metrics=true
    )
    # 1. Unified zeta (Ihara proxy)
    unified_mag = load_unified_zeta()
    println("Loaded unified zeta: $(length(unified_mag)) points")

    # 2. Plücker zeta and snapshot indices
    dense_mags, snap_indices = load_dense_plucker_with_indices()
    println("Loaded dense Plücker: $(length(dense_mags)) steps, $(length(snap_indices)) indices")

    plucker_at_snap = [dense_mags[i] for i in snap_indices]
    if length(plucker_at_snap) != length(unified_mag)
        error("Length mismatch: unified zeta ($(length(unified_mag))) vs snapshot_indices ($(length(snap_indices)))")
    end
    println("Aligned Plücker magnitudes: $(length(plucker_at_snap)) points")

    # 3. Best tubings (optional)
    best_tubings = load_best_tubings()
    n = length(unified_mag)
    if length(best_tubings) > n
        best_tubings = best_tubings[1:n]
    elseif length(best_tubings) < n
        best_tubings = vcat(best_tubings, ["dummy" for _ in 1:(n - length(best_tubings))])
    end

    # 4. Real metrics from A∞ snapshots
    if use_real_metrics
        files = filter(f -> occursin(r"ainf_export_(?:\w+_)?\d+(?:\.\d+)?\.json", f), readdir(snapshot_folder))
        sort!(files)
        if length(files) < n
            @warn "Only $(length(files)) snapshot files found, but $n unified points. Using placeholders."
            use_real_metrics = false
        else
            files = files[1:n]
            m6_vals, cup_vals, gersten_vals = load_metrics_from_snapshots([joinpath(snapshot_folder, f) for f in files])
            println("Loaded real metrics from $(length(files)) snapshots.")
        end
    end

    if !use_real_metrics
        @warn "Using placeholders for m6, cup, gersten (set to unified_mag)."
        m6_vals = unified_mag
        cup_vals = unified_mag
        gersten_vals = unified_mag
    end

    # Placeholders for prime_support and entropy (you can later read from prime_higher_ideals)
    prime_support = ones(n)
    entropy = ones(n)

    # 5. Normalize each generator to [0,1] (min‑max scaling)
    function normalize(arr, method)
        if method == "minmax"
            mn, mx = minimum(arr), maximum(arr)
            return (arr .- mn) / (mx - mn + 1e-12)
        elseif method == "zscore"
            μ, σ = mean(arr), std(arr)
            return (arr .- μ) / (σ + 1e-12)
        else
            return arr
        end
    end

    m6_norm = normalize(m6_vals, norm_type)
    zeta_I_norm = normalize(unified_mag, norm_type)
    zeta_W_norm = normalize(plucker_at_snap, norm_type)
    cup_norm = normalize(cup_vals, norm_type)
    gersten_norm = normalize(gersten_vals, norm_type)
    prime_norm = normalize(prime_support, norm_type)
    entropy_norm = normalize(entropy, norm_type)

    # 6. Build ideals (7 generators)
    ideals = []
    for i in 1:n
        push!(ideals, (m6=m6_norm[i], zeta_I=zeta_I_norm[i], zeta_W=zeta_W_norm[i],
                       cup=cup_norm[i], gersten=gersten_norm[i],
                       prime_support=prime_norm[i], entropy=entropy_norm[i]))
    end

    # 7. Chart indices and flips
    charts = [active_chart(ideal) for ideal in ideals]
    chart_flips = [i for i in 2:n if charts[i] != charts[i-1]]
    tubing_flips = [i for i in 2:n if best_tubings[i] != best_tubings[i-1]]
    coincident = intersect(chart_flips, tubing_flips)
    mismatches = [mismatch(ideal.zeta_I, ideal.zeta_W) for ideal in ideals]

    # 8. Plots
    p1 = plot(1:n, [m6_norm, cup_norm, gersten_norm, zeta_W_norm],
              labels=["m₆" "Cup" "Gerstenhaber" "Plücker ζ"],
              title="Crisis Generators (normalized)", xlabel="Snapshot index")
    p2 = plot(1:n, mismatches, label="Dual‑zeta mismatch", title="Mismatch", color=:orange)
    p3 = plot(1:n, charts, label="Active chart", title="Rees chart index",
              yticks=1:7, color=:green)
    p4 = plot(1:n, 1:n, color=:blue, label="", legend=false)
    scatter!(p4, tubing_flips, tubing_flips, label="Tubing flip", color=:red)
    scatter!(p4, chart_flips, chart_flips, label="Chart flip", color=:green)

    plot(p1, p2, p3, p4, layout=(2,2), size=(1000,800))
    savefig("rees_grassmann_analysis.png")

    println("\nAnalysis complete. Plots saved to rees_grassmann_analysis.png")
    println("Chart flips: ", length(chart_flips))
    println("Tubing flips: ", length(tubing_flips))
    println("Coincident flips: ", length(coincident))
end

end # module
