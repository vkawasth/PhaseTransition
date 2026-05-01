#=
Your IharaAssociahedronBridge.jl and SingularityTracker.jl are ambitious and creative – 
they attempt to link the algebraic obstruction data (prime paths, cup product, Gerstenhaber) 
to graph spectral theory (Ihara zeta proxy) and event detection.
=#
using CairoMakie
include("IharaAssociahedronBridgeV2.jl")
include("SingularityTrackerV2.jl")
using .SingularityTracker
include("ReesBlowupHybrid.jl")
using .ReesBlowupHybrid

# ------------------------------------------------------------
# Helper: load prime zeta (Plücker zeta) from JSON
# ------------------------------------------------------------
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
folder = "./"
T = run_tracker!(folder)                # runs navigator and bridge
B = T.bridge

# Load transition times and prime zeta
transition_times = load_transition_times()
prime_zeta_vals, plucker_times = load_prime_zeta()

if !isempty(transition_times) && !isempty(prime_zeta_vals)
    files = filter(f -> endswith(lowercase(f), ".json"), readdir(folder))
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
    save("bridge_pole_radius.png", plot_pole_radius(B))
end

# Also produce standard tracker outputs
event_table(T)
save("tracker_plots.png", plot_tracker(T))
save("bridge_pole_radius.png", plot_pole_radius(B))
println("Done. Saved: tracker_plots.png, bridge_pole_radius.png")

########################################
# REES ALGEBRA BLOW UP
########################################

B_rees = run_blowup!(folder)
print_events(B_rees)
save("rees_generators.png", plot_generators(B_rees))
save("rees_divisor.png", plot_divisor(B_rees))



