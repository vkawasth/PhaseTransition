"""
plot11_shattering_diagnostic.jl
Layout (2 rows × 3 cols, right col spans both rows via separate plots):
  Row 1, Col 1:  A — Local Basis Stability
  Row 1, Col 2:  B — Ihara Pole Shattering (full)
  Row 2, Col 1:  C — Broken Affinity Matrix
  Row 2, Col 2:  B zoom — Ramanujan region
  Right side:    nothing (C is wide enough)
"""

using Plots, LinearAlgebra, DelimitedFiles, Statistics
gr()
ENV["GKSwstype"] = "100"

include("load_chambers.jl")
d = load_chambers("chambers.tsv")

n_snaps     = d.n
t           = 1:n_snaps
m6          = abs.(d.obs)
walls       = findall(d.walls .> 0)
crisis_snap = argmax(m6)
prolate_gap = d.gap
poles_re    = d.gap_re
poles_im    = d.gap_im
pole_mag    = sqrt.(poles_re.^2 .+ poles_im.^2)
ram_radius  = sqrt(4.0)
θ           = LinRange(0, 2π, 200)
stray_idx   = findall(pole_mag .> ram_radius)

# ── Affinity matrix at crisis snapshot ───────────────────────────────────────
C_mat = hcat(abs.(d.K), abs.(d.gap), abs.(d.obs), abs.(d.score),
             d.support, d.perv, abs.(d.bphase))
for j in 1:size(C_mat,2)
    mx = maximum(abs.(C_mat[:,j]))
    C_mat[:,j] = mx > 1e-10 ? C_mat[:,j]./mx : zeros(n_snaps)
end
c_row    = vec(C_mat[crisis_snap,:])
n_feats  = length(c_row)
σ        = 0.3
W_crisis = [exp(-(c_row[i]-c_row[j])^2/(2σ^2)) for i in 1:n_feats, j in 1:n_feats]
region_names = ["K(t)","gap","m₀(obs)","score","support","perv","B-phase"]

# ── Build each panel independently ───────────────────────────────────────────

# Panel A — top left
pA = plot(t, prolate_gap,
    xlabel="Snapshot", ylabel="Prolate Spectral Gap",
    title="A — Local Basis Stability",
    color=:darkgreen, lw=1.5, label="Gap size",
    legend=:bottomright, legendfontsize=7,
    xlims=(1, n_snaps))
vline!(pA, [crisis_snap], color=:purple, lw=2, ls=:dash, label="m₀ Peak")
vline!(pA, walls, color=:red, alpha=0.25, lw=0.8, label="Walls")

# Panel B — top right (full pole scatter, no aspect_ratio to avoid size fight)
pB = scatter(poles_re, poles_im,
    marker_z=t, seriescolor=:plasma, ms=2.5,
    markerstrokewidth=0, alpha=0.5,
    xlabel="Re(λ)", ylabel="Im(λ)",
    title="B — Ihara Pole Shattering",
    colorbar=false, label="")
plot!(pB, ram_radius.*cos.(θ), ram_radius.*sin.(θ),
    color=:red, lw=1.5, ls=:dash, label="Ramanujan")
if !isempty(stray_idx)
    scatter!(pB, poles_re[stray_idx], poles_im[stray_idx],
        color=:cyan, marker=:xcross, ms=5,
        label="Shattered (m₀≠0)")
end

# Panel C — bottom left (affinity matrix)
pC = heatmap(W_crisis,
    xticks=(1:n_feats, region_names),
    yticks=(1:n_feats, region_names),
    c=:RdYlGn, clims=(0,1),
    xrotation=45, yflip=true,
    title="C — Broken Affinity (snap $crisis_snap)",
    titlefontsize=9)

# Panel B zoom — bottom right
pBz = scatter(poles_re, poles_im,
    marker_z=t, seriescolor=:plasma, ms=4,
    markerstrokewidth=0, alpha=0.7,
    xlabel="Re(λ)", ylabel="Im(λ)",
    title="B zoom — Ramanujan region",
    xlims=(-ram_radius*4, ram_radius*4),
    ylims=(-ram_radius*4, ram_radius*4),
    colorbar=false, label="",
    aspect_ratio=:equal)
plot!(pBz, ram_radius.*cos.(θ), ram_radius.*sin.(θ),
    color=:red, lw=2, ls=:dash, label="√q bound")
if !isempty(stray_idx)
    scatter!(pBz, poles_re[stray_idx], poles_im[stray_idx],
        color=:cyan, marker=:xcross, ms=6, label="Shattered",
        legend=:topright, legendfontsize=7)
end

# ── Assemble: simple 2×2 grid ─────────────────────────────────────────────────
fig = plot(pA, pB, pC, pBz,
    layout=(2, 2),
    size=(1400, 900),
    left_margin=10Plots.mm,
    bottom_margin=10Plots.mm,
    top_margin=4Plots.mm,
    right_margin=4Plots.mm)

savefig(fig, "plot11_shattering_diagnostic.png")
println("Saved: plot11_shattering_diagnostic.png")
println("Crisis snapshot: $crisis_snap  |  Stray poles: $(length(stray_idx))")
