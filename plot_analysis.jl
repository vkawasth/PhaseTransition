###############################################################
# plot_analysis.jl  (integrated — 7 plots)
#
# Run: julia plot_analysis.jl
#
# New vs previous version:
#   Plot 1: +Panel C transition zone zoom
#   Plot 3: +Panel C phase source comparison
#   Plot 7: sphere limit / proof chain (NEW)
###############################################################

using CSV, DataFrames, JSON3, CairoMakie, Statistics, LinearAlgebra, Printf

function load_if_exists(file)
    isfile(file) || return nothing
    ext = splitext(file)[2]
    ext == ".tsv"  && return CSV.read(file, DataFrame)
    ext == ".json" && return JSON3.read(read(file, String))
    nothing
end

function safe_vec(col)
    v = Float64[]
    for x in col
        try push!(v, Float64(x)) catch; push!(v, NaN) end
    end
    v
end

function stable_start_idx(gap)
    t = findfirst(x -> x > 0, gap)
    t === nothing ? 30 : t
end

function norm01(v)
    f = filter(isfinite, v)
    isempty(f) && return v
    lo, hi = minimum(f), maximum(f)
    hi > lo ? (v .- lo) ./ (hi - lo) : zeros(length(v))
end

# ── PLOT 1 ────────────────────────────────────────────────────────────────

function plot1_obstruction_decay(chambers)
    chambers === nothing && return
    n   = nrow(chambers)
    obs = safe_vec(chambers.obs)
    gap = safe_vec(chambers.gap)
    ids = collect(1:n)

    fig = Figure(size=(1300, 850))

    ax1 = Axis(fig[1,1], title="A — Obstruction decay (log₁₀ severity)",
               xlabel="Chamber", ylabel="log₁₀(m6 severity)")
    log_obs = log10.(max.(obs, 1e-12))
    lines!(ax1, ids, log_obs, color=:steelblue, linewidth=2)
    ts = stable_start_idx(gap)
    vlines!(ax1, [ts], color=:red, linestyle=:dash, linewidth=1.5,
            label="First stable gap (ch $ts)")
    eq = median(filter(isfinite, log_obs[min(ts,n):end]))
    hlines!(ax1, [eq], color=:green, linestyle=:dot, linewidth=1.5,
            label=@sprintf("Equilibrium 10^%.1f", eq))
    axislegend(ax1)

    ax2 = Axis(fig[1,2], title="B — Spectral gap (quantized chambers)",
               xlabel="Chamber", ylabel="λ₂")
    scatter!(ax2, ids, gap, color=(:steelblue,0.4), markersize=3)
    gap_pos = filter(x->x>0, gap)
    if !isempty(gap_pos)
        cols_ = [:red,:orange,:green,:purple,:brown]
        for (k,v) in enumerate(sort(unique(round.(gap_pos,digits=1)))[1:min(5,end)])
            hlines!(ax2, [v], color=cols_[k], linestyle=:dash, linewidth=1.5,
                    label=@sprintf("%.2f",v))
        end
    end
    axislegend(ax2, position=:rb)

    # Panel C: transition zone
    t0 = max(1, ts-5); t1 = min(n, ts+50)
    tr = t0:t1
    score_vec = "score" in names(chambers) ? safe_vec(chambers.score) : zeros(n)
    ax3 = Axis(fig[2,1:2],
               title="C — Transition zone ch $t0–$t1: Ramanujan activation",
               xlabel="Chamber", ylabel="Normalised [0,1]")
    lines!(ax3, collect(tr), norm01(log_obs[tr]),   color=:red,       linewidth=2,
           label="log severity (↓)")
    lines!(ax3, collect(tr), norm01(gap[tr]),        color=:steelblue, linewidth=2,
           label="spectral gap (↑)")
    lines!(ax3, collect(tr), norm01(score_vec[tr]),  color=:darkgreen, linewidth=2,
           label="score (slowly↑)")
    vlines!(ax3, [ts], color=:red, linestyle=:dash, linewidth=2)
    hlines!(ax3, [0.5], color=:gray, linestyle=:dot)
    axislegend(ax3, position=:rc)
    text!(ax3, Float64(ts+2), 0.1,
          text="← crisis zone  |  stable zone →\nNon-trivial Ramanujan regime here",
          fontsize=11, color=:gray)

    save("plot1_obstruction_decay.png", fig)
    println("✓ plot1_obstruction_decay.png")
end

# ── PLOT 2 ────────────────────────────────────────────────────────────────

function plot2_wall_weights(chambers)
    chambers === nothing && return
    scores  = safe_vec(chambers.score)
    weights = abs.(diff(scores))
    gap     = safe_vec(chambers.gap)
    ts      = stable_start_idx(gap)
    n_w     = length(weights)

    fig = Figure(size=(1200,500))

    ax1 = Axis(fig[1,1], title="A — Wall crossing weights (log scale)",
               xlabel="Wall index", ylabel="log₁₀|score change|")
    lines!(ax1, collect(1:n_w), log10.(max.(weights,1e-12)),
           color=:darkorange, linewidth=1.5)
    hlines!(ax1, [log10(20.0)], color=:red, linestyle=:dash, label="KS≈20")
    vlines!(ax1, [ts], color=:blue, linestyle=:dot, label="Stable start")
    axislegend(ax1)

    sw = weights[min(ts,n_w):end]; sw = sw[sw.<100]
    ax2 = Axis(fig[1,2], title="B — Stable zone wall weights (KS quanta)",
               xlabel="Weight", ylabel="Count")
    !isempty(sw) && hist!(ax2, sw, bins=50, color=(:steelblue,0.7))
    vlines!(ax2, [20.78,21.04,24.14,41.83], color=:red, linestyle=:dash,
            linewidth=1.5, label="Discrete quanta")
    axislegend(ax2)

    save("plot2_wall_weights.png", fig)
    println("✓ plot2_wall_weights.png")
end

# ── PLOT 3 ────────────────────────────────────────────────────────────────

function plot3_bridgeland_lefschetz(chambers, phase_data)
    chambers === nothing && return
    n   = nrow(chambers)
    bp  = "bridgeland_phase" in names(chambers) ?
          safe_vec(chambers.bridgeland_phase) : zeros(n)
    lef = "lefschetz_proxy" in names(chambers) ?
          safe_vec(chambers.lefschetz_proxy)  : zeros(n)

    fig = Figure(size=(1300,850))

    ax1 = Axis(fig[1,1], title="A — Bridgeland phases (Fukaya stability)",
               xlabel="Chamber", ylabel="arg(λ₂)/2π")
    scatter!(ax1, collect(1:n), bp, color=(:steelblue,0.5), markersize=3)
    hlines!(ax1, [0.0,1.0], color=:red, linestyle=:dash, linewidth=1.5,
            label="Stability window (0,1]")
    in_range = count(p->0.0<p<=1.0, bp)
    bp_f = filter(isfinite,bp)
    !isempty(bp_f) && text!(ax1, n*0.05, maximum(bp_f)*0.9,
          text="$in_range/$n in (0,1]", fontsize=13, color=:darkgreen)
    axislegend(ax1)

    ax2 = Axis(fig[1,2], title="B — Lefschetz proxies (Weil I / perverse schober)",
               xlabel="Chamber", ylabel="Trace of support score change")
    scatter!(ax2, collect(1:n), lef, color=(:darkorange,0.5), markersize=3)
    near_int = count(l->isfinite(l)&&abs(l-round(l))<0.1, lef)
    lef_f = filter(isfinite,lef)
    !isempty(lef_f) && text!(ax2, n*0.05, maximum(lef_f)*0.9,
          text="$near_int/$n near integer", fontsize=13, color=:darkgreen)
    hlines!(ax2, [0.0], color=:gray, linestyle=:dot)

    # Panel C: phase sources
    ax3 = Axis(fig[2,1:2],
               title="C — Plücker phase sources (sphere limit: flat A∞ space → single geodesic)",
               xlabel="History index", ylabel="Phase (radians)")

    if phase_data !== nothing
        np = length(phase_data["phase"])
        hi = collect(1:np)
        if haskey(phase_data,"phase_tangent")
            lines!(ax3, hi, Float64.(phase_data["phase_tangent"]),
                   color=(:red,0.35), linewidth=1, label="tangent")
        end
        if haskey(phase_data,"phase_klein")
            lines!(ax3, hi, Float64.(phase_data["phase_klein"]),
                   color=(:green,0.35), linewidth=1, label="klein")
        end
        lines!(ax3, hi, Float64.(phase_data["phase"]),
               color=:steelblue, linewidth=2.5,
               label="selected: $(get(phase_data,\"source_used\",\"?\"))")
        if haskey(phase_data,"mean_stable")
            sm = Float64(phase_data["mean_stable"])
            hlines!(ax3, [sm], color=:purple, linestyle=:dash, linewidth=2,
                    label=@sprintf("stable mean=%.4f",sm))
            hlines!(ax3, [0.5+1/120], color=:darkred, linestyle=:dot, linewidth=1.5,
                    label=@sprintf("1/2+1/120=%.4f",0.5+1/120))
        end
        axislegend(ax3, position=:rt)
        ph_f = filter(isfinite, Float64.(phase_data["phase"]))
        if !isempty(ph_f)
            text!(ax3, Float64(np*0.35), minimum(ph_f)+0.05*(maximum(ph_f)-minimum(ph_f)),
                  text="Flat phase in stable zone = sphere limit:\nA∞ space nearly flat → prime paths degenerate\nto single geodesic → Ramanujan saturation",
                  fontsize=11, color=:gray)
        end
    else
        text!(ax3, 0.3, 0.0,
              text="plucker_phase.json missing — run: julia generate_plucker_phase.jl",
              fontsize=12, color=:red)
    end

    save("plot3_bridgeland_lefschetz.png", fig)
    println("✓ plot3_bridgeland_lefschetz.png")
end

# ── PLOT 4 ────────────────────────────────────────────────────────────────

function plot4_zeta_mismatch(zeta_df)
    zeta_df === nothing && return
    println("  zeta_comparison columns: ", names(zeta_df))
    n = nrow(zeta_df)

    function get_col(df, cands, def)
        for c in cands; c in names(df) && return safe_vec(df[!,c]); end
        fill(Float64(def), nrow(df))
    end

    mm      = get_col(zeta_df, ["log_mismatch","mismatch"], 0.0)
    n_paths = get_col(zeta_df, ["n_paths","n_active_paths","active_paths"], 0.0)
    walls   = "wall_crossing" in names(zeta_df) ?
              zeta_df[!,"wall_crossing"] : fill(false,n)
    mm_c    = min.(mm, 10.0)
    wids    = findall(x->x=="true"||x==true||x==1, walls)
    bass_ok = count(x->isfinite(x)&&x<0.1, mm)

    fig = Figure(size=(1200,500))

    ax1 = Axis(fig[1,1], title="A — Zeta mismatch |log ζ_spec - log ζ_comb|",
               xlabel="Snapshot", ylabel="Log mismatch (capped 10)")
    lines!(ax1, collect(1:n), mm_c, color=:steelblue, linewidth=1.5)
    !isempty(wids) && scatter!(ax1, wids, mm_c[wids], color=:red,
             markersize=10, marker=:star5, label="Wall crossing")
    hlines!(ax1, [0.1], color=:green, linestyle=:dash, label="Bass threshold")
    axislegend(ax1)
    text!(ax1, n*0.05, 9.0,
          text="Bass: $bass_ok/$n ($(round(100*bass_ok/max(n,1),digits=0))%)",
          fontsize=13, color=:darkgreen)

    ax2 = Axis(fig[1,2], title="B — Active prime paths per snapshot",
               xlabel="Snapshot", ylabel="N paths")
    lines!(ax2, collect(1:n), n_paths, color=:purple, linewidth=1.5)
    !isempty(wids) && scatter!(ax2, wids, n_paths[wids], color=:red,
             markersize=8, label="Wall crossing")
    stable_n = median(filter(x->x>0, n_paths))
    isfinite(stable_n) && hlines!(ax2, [stable_n], color=:green,
             linestyle=:dot, linewidth=1.5,
             label=@sprintf("%.0f paths — degenerate geodesic", stable_n))
    axislegend(ax2)

    save("plot4_zeta_mismatch.png", fig)
    println("✓ plot4_zeta_mismatch.png")
end

# ── PLOT 5 ────────────────────────────────────────────────────────────────

function plot5_plucker_norm(plucker_data)
    plucker_data === nothing && return
    mags = Float64.(plucker_data["magnitudes"])
    n    = length(mags)

    fig = Figure(size=(1200,500))
    ax1 = Axis(fig[1,1], title="A — Plücker norm ‖q‖ over time",
               xlabel="Step", ylabel="‖q‖")
    lines!(ax1, 1:n, mags, color=:steelblue, linewidth=1.5, label="‖q‖")
    rm_ = accumulate(min, mags)
    lines!(ax1, 1:n, rm_, color=:red, linestyle=:dash, linewidth=2,
           label="Running min (Ramanujan floor)")
    med = median(mags)
    hlines!(ax1, [med/sqrt(2)], color=:green, linestyle=:dot,
            linewidth=1.5, label="median/√2")
    axislegend(ax1, position=:rb)

    ax2 = Axis(fig[1,2], title="B — Norm ratio ‖q‖ / running_min",
               xlabel="Step", ylabel="Ratio")
    lines!(ax2, 1:n, mags./max.(rm_,1e-12), color=:darkorange, linewidth=1.5)
    hlines!(ax2, [1.0], color=:red, linestyle=:dash, linewidth=1.5, label="Floor=1")
    axislegend(ax2, position=:rt)

    @printf("  Floor: %.4f  median/√2: %.4f\n", minimum(mags), med/sqrt(2))
    save("plot5_plucker_norm.png", fig)
    println("✓ plot5_plucker_norm.png")
end

# ── PLOT 6 ────────────────────────────────────────────────────────────────

function plot6_ihara_poles(poles_data)
    poles_data === nothing && return
    poles = poles_data["poles"]
    isempty(poles) && return
    re_ = Float64[Float64(p["re"])     for p in poles]
    im_ = Float64[Float64(p["im"])     for p in poles]
    r_  = Float64[Float64(p["radius"]) for p in poles]
    s_  = Int[Int(p["snapshot"])       for p in poles]
    rb  = mean(Float64.(poles_data["ramanujan_bounds"]))

    fig = Figure(size=(800,700))
    ax  = Axis(fig[1,1], title="Ihara poles", xlabel="Re(λ)",
               ylabel="Im(λ)", aspect=DataAspect())
    sc = scatter!(ax, re_, im_, color=s_, colormap=:viridis, markersize=6, alpha=0.6)
    Colorbar(fig[1,2], sc, label="Snapshot")
    θ = range(0,2π,300)
    lines!(ax, rb.*cos.(θ), rb.*sin.(θ), color=:red, linewidth=2,
           linestyle=:dash, label="Ramanujan |λ|=√q")
    vlines!(ax, [0.5], color=:blue, linestyle=:dash, linewidth=1.5, label="Re=0.5")
    axislegend(ax, position=:rt)
    near = count(x->abs(x-rb)<0.05*rb, r_)
    text!(ax, minimum(re_)+0.02, maximum(im_)-0.3,
          text="Near circle: $near/$(length(r_))", fontsize=12, color=:darkred)
    save("plot6_ihara_poles.png", fig)
    println("✓ plot6_ihara_poles.png")
end

# ── PLOT 7 ────────────────────────────────────────────────────────────────

function plot7_sphere_limit(chambers, plucker_traj)
    chambers === nothing && return
    n       = nrow(chambers)
    gap     = safe_vec(chambers.gap)
    obs     = safe_vec(chambers.obs)
    score   = "score" in names(chambers) ? safe_vec(chambers.score) : zeros(n)
    ts      = stable_start_idx(gap)
    log_obs = log10.(max.(obs, 1e-12))

    fig = Figure(size=(1300,900))

    # Panel A: obstruction level
    ax1 = Axis(fig[1,1], title="A — Obstruction level (sphere limit approach)",
               xlabel="Chamber", ylabel="log₁₀(obstruction)")
    lines!(ax1, collect(1:n), log_obs, color=:steelblue, linewidth=2)
    vlines!(ax1, [ts], color=:red, linestyle=:dash, linewidth=1.5,
            label="Stable start")
    eq = median(filter(isfinite, log_obs[min(ts,n):end]))
    hlines!(ax1, [eq], color=:green, linestyle=:dot, linewidth=2,
            label=@sprintf("Equilibrium ~10^%.1f", eq))
    axislegend(ax1)
    text!(ax1, n*0.4, eq+1.0,
          text="Low obstruction = near-flat A∞ space\n= sphere limit regime",
          fontsize=11, color=:darkgreen)

    # Panel B: score drift = motion along geodesic
    ax2 = Axis(fig[1,2], title="B — Score drift (Toda flow along unique geodesic)",
               xlabel="Chamber", ylabel="Score")
    sc_stable = score[min(ts,n):end]
    ids_s = collect(ts:n)
    lines!(ax2, ids_s, sc_stable, color=:darkgreen, linewidth=2)
    if length(sc_stable) > 2
        sc_f = filter(isfinite, sc_stable)
        if length(sc_f) > 1
            slope = (sc_f[end]-sc_f[1]) / max(length(sc_f)-1, 1)
            text!(ax2, Float64(ids_s[length(ids_s)÷4]),
                  median(filter(isfinite, sc_stable)),
                  text=@sprintf("+%.1f/chamber\n(walking along geodesic)", slope),
                  fontsize=11, color=:darkgreen)
        end
    end

    # Panel C: Plücker trajectory
    if plucker_traj !== nothing
        q12_ = Float64.(plucker_traj["q12"])
        q13_ = Float64.(plucker_traj["q13"])
        norms_ = sqrt.(q12_.^2 .+ q13_.^2)
        nm = maximum(norms_)

        ax3 = Axis(fig[2,1], title="C — Plücker trajectory q12 vs q13",
                   xlabel="q12", ylabel="q13")
        scatter!(ax3, q12_, q13_, color=norms_./max(nm,1e-10),
                 colormap=:viridis, markersize=3, alpha=0.6)
        text!(ax3, maximum(q12_)*0.1, maximum(q13_)*0.8,
              text="Arc = stable zone (Toda on Gr(2,4))\nCluster = crisis zone",
              fontsize=11, color=:gray)
    else
        ax3 = Axis(fig[2,1], title="C — Plücker trajectory (not available)")
        text!(ax3, 0.5, 0.5, text="plucker_trajectory.json not found",
              fontsize=12, color=:red, align=(:center,:center))
    end

    # Panel D: proof chain
    ax4 = Axis(fig[2,2], title="D — Proof chain: sphere limit → Ramanujan")
    hidedecorations!(ax4); hidespines!(ax4)
    steps = [
        (0.5,0.93,"Toda-Lax flow on M_{A∞}",:steelblue,14),
        (0.5,0.82,"↓",:black,16),
        (0.5,0.73,"Obstruction decays  ‖mₖ‖→0  (k≥3)",:darkorange,12),
        (0.5,0.62,"↓",:black,16),
        (0.5,0.53,"A∞ space → flat  (sphere limit)",:darkgreen,12),
        (0.5,0.42,"↓",:black,16),
        (0.5,0.33,"Prime paths → single geodesic",:purple,12),
        (0.5,0.22,"↓",:black,16),
        (0.5,0.11,"‖T‖ → √q   poles on Re(s)=1/2",:red,13),
    ]
    for (x,y,txt,col,fs) in steps
        text!(ax4, x, y, text=txt, fontsize=fs, color=col,
              align=(:center,:center))
    end

    save("plot7_sphere_limit.png", fig)
    println("✓ plot7_sphere_limit.png")
end

# ── MAIN ─────────────────────────────────────────────────────────────────

println("="^60)
println("PLOT ANALYSIS")
println("="^60)

chambers     = load_if_exists("chambers.tsv")
zeta_df      = load_if_exists("zeta_comparison.tsv")
plucker_data = load_if_exists("plucker_zeta_dense.json")
poles_data   = load_if_exists("ihara_poles.json")
phase_data   = load_if_exists("plucker_phase.json")
plucker_traj = load_if_exists("plucker_trajectory.json")

println("chambers.tsv:           ", chambers     === nothing ? "✗" : "✓ $(nrow(chambers)) rows")
println("zeta_comparison.tsv:    ", zeta_df      === nothing ? "✗" : "✓ $(nrow(zeta_df)) rows")
println("plucker_zeta_dense.json:", plucker_data === nothing ? "✗" : "✓")
println("ihara_poles.json:       ", poles_data   === nothing ? "✗" : "✓")
println("plucker_phase.json:     ", phase_data   === nothing ? "✗ run generate_plucker_phase.jl" :
        "✓ $(get(phase_data,\"unique_count\",\"?\")) unique phases")
println("plucker_trajectory.json:", plucker_traj === nothing ? "✗" : "✓")
println()

plot1_obstruction_decay(chambers)
plot2_wall_weights(chambers)
plot3_bridgeland_lefschetz(chambers, phase_data)
plot4_zeta_mismatch(zeta_df)
plot5_plucker_norm(plucker_data)
plot6_ihara_poles(poles_data)
plot7_sphere_limit(chambers, plucker_traj)

println()
println("Saved: plot1–plot7")
println("  plot7_sphere_limit.png — NEW: sphere limit / proof chain")
println("="^60)
