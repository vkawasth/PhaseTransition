"""
plot_master_timeline.jl
Master timeline: the full sequence of events in one figure.

Panels stacked vertically, all sharing the same x-axis (snapshot).
Each panel shows one "layer" of the event progression:

  1. Obstruction (m6 norm)     — crisis onset and resolution
  2. Prolate gap               — eigenbasis stability / collapse / re-emergence  
  3. Complex pole radius       — spectral shattering / reassembly
  4. Bridgeland phase          — Schubert stratum occupancy
  5. Klein constraint K(t)     — sphere limit approach
  6. Fiedler gap               — cluster boundary strength (prime path ideal)
  7. Score / Rees activity     — blowup / recovery events

Vertical event markers (shaded bands):
  Red   = wall crossing (Rees blowup triggered)
  Green = recovery (new eigenbasis stabilised)
  Blue  = prime path ideal transition (Schubert stratum change)
"""

using Plots, DelimitedFiles, Statistics, LinearAlgebra
gr()
ENV["GKSwstype"] = "100"

include("load_chambers.jl")
d = load_chambers("chambers.tsv")

n   = d.n
t   = 1:n
m6  = log10.(abs.(d.obs) .+ 1)       # log10 obstruction
K   = abs.(d.K)
bp  = d.bphase
sc  = d.score
gap = d.gap
gap_re = d.gap_re
gap_im = d.gap_im
pole_r = sqrt.(gap_re.^2 .+ gap_im.^2)

# ── Prolate gap (from spectrogram computation) ─────────────────────────────
q12 = d.q12; q13 = d.q13; q14 = d.q14
q23 = d.q23; q24 = d.q24; q34 = d.q34
s0  = ones(n)
s1  = q12 .+ q34
s20 = q12.^2 .+ q13.*q24 .+ q34.^2
s11 = q13.*q24 .- q14.*q23
s21 = (q12 .+ q34) .* s11
s22 = K.^2
modes = hcat(s0, s1, s20, s11, s21, s22)
conc  = zeros(n, 6)
for j in 1:6
    mx = maximum(abs.(modes[:,j]))
    conc[:,j] = mx > 1e-10 ? abs.(modes[:,j]) ./ mx : zeros(n)
end
prolate_gap = [let s = sort(conc[i,:], rev=true); length(s)>1 ? s[1]-s[2] : 0.0; end for i in 1:n]

# ── Fiedler gap proxy (use score derivative as affinity gap proxy) ─────────
score_norm = abs.(sc) ./ (maximum(abs.(sc)) .+ 1e-10)
fiedler_px = abs.(diff([score_norm; score_norm[end]]))  # gradient of score

# ── Event detection ────────────────────────────────────────────────────────
wall_snaps    = findall(d.walls .> 0)
thresh_m6     = quantile(m6, 0.92)
crisis_snaps  = findall(m6 .> thresh_m6)
# Recovery: after each crisis, first snapshot where m6 drops below median
m6_med        = median(m6)
recovery_snaps = let
    in_crisis = false
    result = Int[]
    for i in 1:n
        if m6[i] > thresh_m6; in_crisis = true; end
        if in_crisis && m6[i] < m6_med
            push!(result, i)
            in_crisis = false
        end
    end
    result
end

# ── 7-panel layout ─────────────────────────────────────────────────────────
heights = [0.20, 0.14, 0.14, 0.12, 0.12, 0.14, 0.14]
lo      = grid(7, 1, heights=heights)
shared_xlim = (1, n)
wall_alpha  = 0.12

function add_events!(p, walls, recoveries; ylims=nothing)
    for w in walls
        vline!(p, [w], color=:red, alpha=wall_alpha, lw=1.5, label="")
    end
    for r in recoveries
        vline!(p, [r], color=:green, alpha=wall_alpha, lw=1.0, label="")
    end
end

# Panel 1: Obstruction / m6 norm ────────────────────────────────────────────
p1 = plot(t, m6,
    ylabel="log₁₀|m₆|", label="", color=:crimson, lw=1.2,
    title="Master timeline: A∞ crisis → blowup → recovery → new eigenbasis",
    titlefontsize=9, xlims=shared_xlim, xformatter=_->"",
    ytickfontsize=7, yguidefontsize=7, bottom_margin=-2Plots.mm)
hline!(p1, [thresh_m6], color=:darkred, ls=:dash, lw=0.8, label="")
add_events!(p1, wall_snaps, recovery_snaps)
annotate!(p1, n*0.02, maximum(m6)*0.88, Plots.text("m₆ obstruction", 6, :crimson, :left))

# Panel 2: Prolate eigenbasis gap ───────────────────────────────────────────
p2 = plot(t, prolate_gap,
    ylabel="Prolate\ngap", label="", color=:royalblue, lw=1.0,
    xlims=shared_xlim, xformatter=_->"", ylims=(0,1.05),
    ytickfontsize=7, yguidefontsize=7, bottom_margin=-2Plots.mm)
add_events!(p2, wall_snaps, recovery_snaps)
# Annotate eigenbasis events
if !isempty(recovery_snaps)
    r1 = recovery_snaps[1]
    annotate!(p2, min(r1+20,n), 0.85, Plots.text("new eigenbasis", 5, :green, :left))
end
annotate!(p2, n*0.02, 0.88, Plots.text("Schur mode gap", 6, :royalblue, :left))

# Panel 3: Spectral shattering (pole radius) ────────────────────────────────
ram = sqrt(5.0)
p3 = plot(t, log10.(pole_r .+ 0.1),
    ylabel="log|λ|", label="", color=:darkorange, lw=0.9,
    xlims=shared_xlim, xformatter=_->"",
    ytickfontsize=7, yguidefontsize=7, bottom_margin=-2Plots.mm)
hline!(p3, [log10(ram)], color=:red, ls=:dash, lw=1.2, label="")
add_events!(p3, wall_snaps, recovery_snaps)
annotate!(p3, n*0.02, maximum(pole_r)*0.88, Plots.text("pole radius (shattering)", 6, :darkorange, :left))
annotate!(p3, n*0.75, log10(ram)+0.08, Plots.text("√q Ramanujan", 5, :red, :right))

# Panel 4: Bridgeland phase ─────────────────────────────────────────────────
p4 = plot(t, bp,
    ylabel="B-phase", label="", color=:purple, lw=0.9,
    xlims=shared_xlim, xformatter=_->"",
    ytickfontsize=7, yguidefontsize=7, bottom_margin=-2Plots.mm)
add_events!(p4, wall_snaps, recovery_snaps)
annotate!(p4, n*0.02, maximum(bp)*0.88, Plots.text("Bridgeland stability", 6, :purple, :left))

# Panel 5: Lefschetz proxy (perverse schober stalk occupancy) ─────────────────
p5 = plot(t, d.lefsch,
    ylabel="Lefschetz
proxy", label="", color=:teal, lw=0.9,
    xlims=shared_xlim, xformatter=_->"",
    ytickfontsize=7, yguidefontsize=7, bottom_margin=-2Plots.mm)
add_events!(p5, wall_snaps, recovery_snaps)
annotate!(p5, n*0.02, maximum(d.lefsch)*0.88, Plots.text("Lefschetz / schober stalk", 6, :teal, :left))

# Panel 6: Fiedler / prime path ideal proxy ─────────────────────────────────
p6 = plot(t, fiedler_px,
    ylabel="Cluster\nboundary", label="", color=:darkgreen, lw=0.9,
    xlims=shared_xlim, xformatter=_->"",
    ytickfontsize=7, yguidefontsize=7, bottom_margin=-2Plots.mm)
add_events!(p6, wall_snaps, recovery_snaps)
annotate!(p6, n*0.02, maximum(fiedler_px)*0.88,
    Plots.text("prime path ideal / μ-κ boundary", 6, :darkgreen, :left))

# Panel 7: Score / Rees activity ────────────────────────────────────────────
sc_norm = sc ./ (maximum(abs.(sc)) .+ 1e-10)
score_smooth = [mean(sc_norm[max(1,i-3):min(n,i+3)]) for i in 1:n]
p7 = plot(t, score_smooth,
    ylabel="Score\n(Rees)", label="", color=:brown, lw=0.9,
    xlabel="Snapshot", xlims=shared_xlim,
    ytickfontsize=7, yguidefontsize=7, xtickfontsize=7, xguidefontsize=7)
add_events!(p7, wall_snaps, recovery_snaps)
annotate!(p7, n*0.02, maximum(score_smooth)*0.88,
    Plots.text("transport score / blowup activity", 6, :brown, :left))

# ── Event legend strip (separate panel would be too small; use plot title) ──
fig = plot(p1, p2, p3, p4, p5, p6, p7,
    layout=lo, size=(1300, 1100),
    left_margin=12Plots.mm, right_margin=4Plots.mm)

savefig(fig, "plot_master_timeline.png")
println("Saved: plot_master_timeline.png")
println()
println("Event summary:")
println("  Wall crossings (Rees blowup): $(length(wall_snaps))")
println("  Crisis events (m6 > 92nd pct): $(length(crisis_snaps))")
println("  Recovery events:              $(length(recovery_snaps))")
