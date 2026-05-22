using Plots, LinearAlgebra, DelimitedFiles, Statistics
gr()
ENV["GKSwstype"] = "100"

include("load_chambers.jl")
d = load_chambers("chambers.tsv")

n_snaps = d.n
m6      = abs.(d.obs)
walls   = findall(d.walls .> 0)

# ── Region features: use available per-snapshot signals as proxy ───────────
# Each column = one "region signal" proxy derived from the simulation output
# For Q7P (7 regions): combine available signals into a 7-channel matrix
# Channels: K, gap, obs, score, support, perv, bphase
C = hcat(abs.(d.K), abs.(d.gap), abs.(d.obs), abs.(d.score),
         d.support, d.perv, abs.(d.bphase))
# Normalise each channel
for j in 1:size(C,2)
    mx = maximum(abs.(C[:,j]))
    C[:,j] = mx > 1e-10 ? C[:,j] ./ mx : zeros(n_snaps)
end

region_names  = ["K(t)", "gap", "obs(m6)", "score", "support", "perv", "B-phase"]
true_clusters = [2, 1, 1, 2, 2, 2, 1]   # 1=algebraic, 2=geometric
σ = 0.3

function affinity_matrix(c_row, σ)
    n = length(c_row)
    W = zeros(n, n)
    for i in 1:n, j in 1:n
        W[i,j] = exp(-((c_row[i]-c_row[j])^2)/(2σ^2))
    end
    W
end

function fiedler_vector(W)
    D = Diagonal(vec(sum(W, dims=2)))
    L = Symmetric(D - W)
    vals, vecs = eigen(L)
    vecs[:,2], vals[2]
end

# Select snapshots
good_candidates = findall((m6 .< quantile(m6, 0.3)) .& (d.walls .== 0))
good_snap  = isempty(good_candidates) ? 50 : good_candidates[end÷2]
poor_snap  = isempty(walls) ? argmax(m6) : walls[argmax(m6[walls])]
recov_snap = min(poor_snap + 20, n_snaps)

println("Good: $good_snap  Poor: $poor_snap  Recovery: $recov_snap")

function make_row(snap)
    W  = affinity_matrix(vec(C[snap,:]), σ)
    Wr, fv, ord = let
        fv, _ = fiedler_vector(W)
        ord = sortperm(fv)
        W[ord,ord], fv[ord], ord
    end
    W, Wr, fv, ord
end

Wg,Wgr,fvg,og = make_row(good_snap)
Wp,Wpr,fvp,op = make_row(poor_snap)

clim = (0.0, 1.0)

hg1 = heatmap(Wg,  title="Input (stable, snap $good_snap)",
    xticks=(1:7,region_names), yticks=(1:7,region_names),
    c=:RdYlGn, clims=clim, xrotation=45, yflip=true, colorbar=false, titlefontsize=8)
hg2 = heatmap(Wgr, title="Reordered (clean blocks)",
    xticks=(1:7,region_names[og]), yticks=(1:7,region_names[og]),
    c=:RdYlGn, clims=clim, xrotation=45, yflip=true, colorbar=false, titlefontsize=8)
fg  = scatter(1:7, fvg,
    color=ifelse.(true_clusters[og].==1,:steelblue,:tomato),
    ms=6, title="Fiedler (clear step)", xlabel="", ylabel="Fiedler value",
    xticks=(1:7,region_names[og]), xrotation=45, legend=false, titlefontsize=8)
hline!(fg,[0.0],color=:black,lw=1,ls=:dash)

hp1 = heatmap(Wp,  title="Input (wall crossing, snap $poor_snap)",
    xticks=(1:7,region_names), yticks=(1:7,region_names),
    c=:RdYlGn, clims=clim, xrotation=45, yflip=true, colorbar=false, titlefontsize=8)
hp2 = heatmap(Wpr, title="Reordered (diffuse blocks)",
    xticks=(1:7,region_names[op]), yticks=(1:7,region_names[op]),
    c=:RdYlGn, clims=clim, xrotation=45, yflip=true, colorbar=false, titlefontsize=8)
fp  = scatter(1:7, fvp,
    color=ifelse.(true_clusters[op].==1,:steelblue,:tomato),
    ms=6, title="Fiedler (noisy)", xlabel="", ylabel="Fiedler value",
    xticks=(1:7,region_names[op]), xrotation=45, legend=false, titlefontsize=8)
hline!(fp,[0.0],color=:black,lw=1,ls=:dash)

# Fiedler gap time series
fgap = [fiedler_vector(affinity_matrix(vec(C[i,:]),σ))[2] for i in 1:n_snaps]
pg = plot(1:n_snaps, fgap,
    xlabel="Snapshot", ylabel="Fiedler λ₂",
    title="Fiedler gap over time — cluster boundary strength",
    label="λ₂(L)", color=:steelblue, lw=1.5)
vline!(pg, walls, color=:red, alpha=0.4, lw=1, label="wall crossings")
vline!(pg, [good_snap],  color=:green,  lw=2, ls=:dash, label="good")
vline!(pg, [poor_snap],  color=:red,    lw=2, ls=:dash, label="poor")

top_row = plot(hg1, hg2, fg, layout=(1,3),
    plot_title="Good similarity (stable)", plot_titlefontsize=9)
mid_row = plot(hp1, hp2, fp, layout=(1,3),
    plot_title="Poor similarity (wall crossing)", plot_titlefontsize=9)

fig = plot(top_row, mid_row, pg,
    layout=grid(3,1,heights=[0.35,0.35,0.30]),
    size=(1300,1200))
savefig(fig, "plot10_affinity_reorganisation.png")
println("Saved: plot10_affinity_reorganisation.png")
