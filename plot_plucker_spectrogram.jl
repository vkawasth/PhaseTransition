using Plots, DelimitedFiles, Statistics, LinearAlgebra
gr()
ENV["GKSwstype"] = "100"

include("load_chambers.jl")
d = load_chambers("chambers.tsv")

n_snaps = d.n
q12, q13, q14 = d.q12, d.q13, d.q14
q23, q24, q34 = d.q23, d.q24, d.q34
K      = d.K
m6     = abs.(d.obs)
walls  = findall(d.walls .> 0)

# ── Schur polynomial modes ─────────────────────────────────────────────────
s0  = ones(n_snaps)
s1  = q12 .+ q34
s20 = q12.^2 .+ q13.*q24 .+ q34.^2
s11 = q13.*q24 .- q14.*q23
s21 = (q12 .+ q34) .* s11
s22 = K.^2

modes      = hcat(s0, s1, s20, s11, s21, s22)
mode_names = ["s_∅", "s_(1)", "s_(2)", "s_(1,1)", "s_(2,1)", "s_(2,2)=K²"]
n_modes    = 6

# ── Normalise ──────────────────────────────────────────────────────────────
conc = zeros(n_snaps, n_modes)
for j in 1:n_modes
    mx = maximum(abs.(modes[:,j]))
    conc[:,j] = mx > 1e-10 ? abs.(modes[:,j]) ./ mx : zeros(n_snaps)
end

gap_series = [let s = sort(conc[i,:], rev=true); length(s)>1 ? s[1]-s[2] : 0.0; end
              for i in 1:n_snaps]

# ── Panel A: Spectrogram ───────────────────────────────────────────────────
pA = heatmap(1:n_snaps, 1:n_modes, conc',
    xlabel="Snapshot", ylabel="Schur mode",
    yticks=(1:n_modes, mode_names),
    title="Gr(2,4) Prolate Spectrogram",
    color=:viridis, clims=(0,1))
vline!(pA, walls, color=:red, alpha=0.5, lw=1, label="wall crossings")

# ── Panel B: Eigenvalue gap ────────────────────────────────────────────────
pB = plot(1:n_snaps, gap_series,
    xlabel="Snapshot", ylabel="Prolate gap",
    title="Eigenvalue gap (Schubert stratum indicator)",
    label="gap(t)", color=:blue, lw=1.5)
vline!(pB, walls, color=:red, alpha=0.4, lw=1, label="wall crossings")

# ── Panel C: Mode activity — which Schur modes are most active ──────────────
# Since K≈0 throughout (sphere limit), K² is near machine zero and uninformative.
# Instead: show the TIME-AVERAGED concentration of each mode (bar chart)
# and the ACTIVE MODE COUNT over time (how many modes have conc > 0.1)
mean_conc  = [mean(conc[:,j]) for j in 1:n_modes]
active_cnt = [sum(conc[i,:] .> 0.1) for i in 1:n_snaps]

# Left sub-panel: mean mode concentrations (which modes dominate overall)
pC1 = bar(0:n_modes-1, mean_conc,
    xlabel="Schur degree |μ|", ylabel="Mean concentration",
    title="C — Mean Schur mode activity",
    xticks=(0:n_modes-1, ["∅","(1)","(2)","(1,1)","(2,1)","(2,2)"]),
    color=[:gold,:green,:teal,:blue,:purple,:red],
    legend=false, xrotation=30)

# ── Panel D: Three-phase fan ───────────────────────────────────────────────
stable_snap  = walls[end] + min(50, n_snaps - walls[end] - 1)
crisis_snap  = argmax(m6)
recover_snap = min(crisis_snap + 30, n_snaps)

pD = plot(title="Schur mode profile: three phases",
    xlabel="Schur degree", ylabel="Concentration", legend=:topright)
for (snap, lab, col) in [(stable_snap,"Stable",:blue),
                          (crisis_snap,"Crisis",:red),
                          (recover_snap,"Recovery",:green)]
    plot!(pD, 0:n_modes-1, conc[snap,:], label=lab, color=col, lw=2,
          marker=:circle, ms=4)
end

fig = plot(pA, pB, pC1, pD, layout=(2,2), size=(1300,900))
savefig(fig, "plucker_spectrogram.png")
println("Saved: plucker_spectrogram.png")
println("Snapshots: $n_snaps  |  Wall crossings: $(length(walls))")
println("K range: [$(round(minimum(abs.(K)),sigdigits=3)), $(round(maximum(abs.(K)),sigdigits=3))]")
