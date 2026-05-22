using Plots, LinearAlgebra, DelimitedFiles, Statistics
gr()
ENV["GKSwstype"] = "100"

include("load_chambers.jl")
d = load_chambers("chambers.tsv")

ram_radius = sqrt(5.0)   # sqrt(q_max) for Q_7P; use sqrt(4) for Q_7L
lam1, lam2 = 1.7898, -0.8021
n_snaps    = d.n
m6         = abs.(d.obs)
walls      = findall(d.walls .> 0)

# Dynamic "Ihara-like" poles from complex spectral gap
# gap_complex = gap_re + i*gap_im encodes the dynamic eigenvalue
poles_re = d.gap_re
poles_im = d.gap_im
pole_mag = sqrt.(poles_re.^2 .+ poles_im.^2)
stray    = [pole_mag[i] > ram_radius ? 1 : 0 for i in 1:n_snaps]
n_stray  = stray

# ── Panel A: Pole trajectory ───────────────────────────────────────────────
θ = LinRange(0, 2π, 300)
pA = scatter(poles_re, poles_im,
    marker_z=1:n_snaps, seriescolor=:plasma,
    ms=2, markerstrokewidth=0, alpha=0.5,
    xlabel="Re(λ)", ylabel="Im(λ)",
    title="A — Ihara pole trajectory\n(purple=early, yellow=late)",
    aspect_ratio=:equal, colorbar_title="Snapshot",
    xlims=(-ram_radius*3, ram_radius*4), ylims=(-ram_radius*2, ram_radius*2),
    legend=false)
plot!(pA, ram_radius.*cos.(θ), ram_radius.*sin.(θ),
    color=:red, lw=2, ls=:dash)
scatter!(pA, [lam1, lam2], [0.0, 0.0],
    color=:yellow, ms=8, marker=:star5, markerstrokecolor=:black)
# Mark stray poles
stray_idx = findall(pole_mag .> ram_radius)
if !isempty(stray_idx)
    scatter!(pA, poles_re[stray_idx], poles_im[stray_idx],
        color=:white, ms=4, markerstrokecolor=:red, markerstrokewidth=1.5)
end

# ── Panel B: Stray pole count ──────────────────────────────────────────────
pB = bar(1:n_snaps, n_stray,
    xlabel="Snapshot", ylabel="Poles outside Ramanujan circle",
    title="B — Spectral shattering events",
    color=ifelse.(n_stray .> 0, :red, :steelblue), lw=0, label="")
vline!(pB, walls, color=:black, alpha=0.5, lw=0.8, label="")

# ── Panel C: Pole radius over time ────────────────────────────────────────
pC = plot(1:n_snaps, pole_mag,
    xlabel="Snapshot", ylabel="|pole|",
    title="C — Pole radius dynamics",
    label="|λ(t)|", color=:steelblue, lw=1.5)
hline!(pC, [ram_radius], color=:red, lw=2, ls=:dash, label="√q Ramanujan")
vline!(pC, walls, color=:black, alpha=0.3, lw=0.8, label="")

# ── Panel D: Zoom on worst event ──────────────────────────────────────────
crisis_snap = isempty(stray_idx) ? argmax(pole_mag) : stray_idx[argmax(pole_mag[stray_idx])]
win = max(1,crisis_snap-8):min(n_snaps,crisis_snap+8)
n_w = length(win)
cols_d = [RGB((i-1)/(n_w-1+1e-8), 1-(i-1)/(n_w-1+1e-8), 0.3) for i in 1:n_w]
pD = scatter(title="D — Zoom: shattering at snapshot $crisis_snap",
    xlabel="Re(λ)", ylabel="Im(λ)", aspect_ratio=:equal,
    xlims=(-ram_radius*2, ram_radius*2.5), ylims=(-ram_radius*1.5,ram_radius*1.5))
for (ci, i) in enumerate(win)
    scatter!(pD, [poles_re[i]], [poles_im[i]], color=cols_d[ci],
             ms=5, markerstrokewidth=0, label="")
end
plot!(pD, ram_radius.*cos.(θ), ram_radius.*sin.(θ), color=:red, lw=2, ls=:dash, label="")

fig = plot(pA, pB, pC, pD, layout=(2,2), size=(1300,1000),
    plot_title="Plot 9: Ihara Pole Dynamics — Spectral Shattering")
savefig(fig, "plot9_ihara_pole_dynamics.png")
println("Saved: plot9_ihara_pole_dynamics.png")
println("Stray pole events: $(sum(n_stray)) / $n_snaps snapshots")
println("Max |pole|: $(round(maximum(pole_mag), sigdigits=4)) vs Ramanujan √q=$(round(ram_radius,sigdigits=4))")
