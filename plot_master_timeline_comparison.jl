"""
plot_master_timeline_comparison.jl
Side-by-side master timeline: Phase 1 (left) vs Phase 2 (right).

Usage:
  julia plot_master_timeline_comparison.jl

Expects two chambers.tsv files:
  chambers_phase1.tsv   — Q7L Phase 1 (m0=0, flat A∞)
  chambers_phase2.tsv   — Q7L Phase 2 (m0=5.0, curved A∞)

If only one file exists, pass it as chambers.tsv and the script
will synthesise a flat Phase 1 from the first 801 snapshots.
"""

using Plots, DelimitedFiles, Statistics, LinearAlgebra
gr()
ENV["GKSwstype"] = "100"

include("load_chambers.jl")

# ── Load both phases ──────────────────────────────────────────────────────────
file1 = isfile("chambers_phase1.tsv") ? "chambers_phase1.tsv" : "chambers.tsv"
file2 = isfile("chambers_phase2.tsv") ? "chambers_phase2.tsv" : "chambers.tsv"

println("Phase 1 file: $file1")
println("Phase 2 file: $file2")

d1 = load_chambers(file1)
d2 = load_chambers(file2)

# ── Compute derived signals for one phase ─────────────────────────────────────
function compute_signals(d)
    n   = d.n
    m6  = log10.(abs.(d.obs) .+ 1)
    K   = abs.(d.K)

    # Prolate gap
    q12=d.q12; q13=d.q13; q14=d.q14
    q23=d.q23; q24=d.q24; q34=d.q34
    modes = hcat(ones(n), q12.+q34,
                 q12.^2 .+q13.*q24.+q34.^2,
                 q13.*q24.-q14.*q23,
                 (q12.+q34).*(q13.*q24.-q14.*q23),
                 K.^2)
    conc = zeros(n,6)
    for j in 1:6
        mx = maximum(abs.(modes[:,j]))
        conc[:,j] = mx>1e-10 ? abs.(modes[:,j])./mx : zeros(n)
    end
    pgap = [let s=sort(conc[i,:],rev=true); length(s)>1 ? s[1]-s[2] : 0.0; end
            for i in 1:n]

    pole_r = sqrt.(d.gap_re.^2 .+ d.gap_im.^2)
    bp     = d.bphase
    lefsch = d.lefsch
    perv_n = d.perv ./ (maximum(abs.(d.perv)).+1e-10)
    sc_norm= let mx=maximum(abs.(d.score)).+1e-10
        [mean(d.score[max(1,i-3):min(n,i+3)] ./mx) for i in 1:n]
    end

    thresh_m6 = quantile(m6, 0.92)
    m6_med    = median(m6)
    wall_snaps = findall(d.walls .> 0)
    recovery_snaps = let
        in_crisis=false; result=Int[]
        for i in 1:n
            if m6[i]>thresh_m6; in_crisis=true; end
            if in_crisis && m6[i]<m6_med; push!(result,i); in_crisis=false; end
        end
        result
    end

    (n=n, m6=m6, pgap=pgap, pole_r=pole_r, bp=bp,
     lefsch=lefsch, perv_n=perv_n, score=sc_norm,
     walls=wall_snaps, recoveries=recovery_snaps,
     thresh_m6=thresh_m6)
end

s1 = compute_signals(d1)
s2 = compute_signals(d2)

ram = sqrt(5.0)

# ── Build one column of 7 panels ──────────────────────────────────────────────
function make_column(s, title_str, show_ylabel, show_xlabel)
    n  = s.n
    t  = 1:n
    xl = (1, n)
    wa = 0.10   # wall band alpha

    function add_ev!(p)
        for w in s.walls;      vline!(p,[w],color=:red,  alpha=wa,lw=1.2,label=""); end
        for r in s.recoveries; vline!(p,[r],color=:green,alpha=wa,lw=0.9,label=""); end
    end

    fmt = show_xlabel ? Plots.Formatter(identity) : (v -> "")
    yl(s) = show_ylabel ? s : ""

    # 1. m6 obstruction
    p1 = plot(t, s.m6, color=:crimson, lw=1.2, label="",
        title=title_str, titlefontsize=8,
        ylabel=yl("log₁₀|m₆|"), xlims=xl,
        xformatter=_->"", ytickfontsize=6, yguidefontsize=6,
        bottom_margin=-2Plots.mm, top_margin=2Plots.mm)
    hline!(p1,[s.thresh_m6],color=:darkred,ls=:dash,lw=0.7,label="")
    add_ev!(p1)

    # 2. Prolate eigenbasis gap
    p2 = plot(t, s.pgap, color=:royalblue, lw=0.9, label="",
        ylabel=yl("Prolate\ngap"), xlims=xl, ylims=(0,1.05),
        xformatter=_->"", ytickfontsize=6, yguidefontsize=6,
        bottom_margin=-2Plots.mm)
    add_ev!(p2)

    # 3. Pole radius (log)
    lr = log10.(s.pole_r .+ 0.1)
    p3 = plot(t, lr, color=:darkorange, lw=0.9, label="",
        ylabel=yl("log|λ|"), xlims=xl,
        xformatter=_->"", ytickfontsize=6, yguidefontsize=6,
        bottom_margin=-2Plots.mm)
    hline!(p3,[log10(ram)],color=:red,ls=:dash,lw=1.0,label="")
    add_ev!(p3)

    # 4. Bridgeland phase
    p4 = plot(t, s.bp, color=:purple, lw=0.8, label="",
        ylabel=yl("B-phase"), xlims=xl,
        xformatter=_->"", ytickfontsize=6, yguidefontsize=6,
        bottom_margin=-2Plots.mm)
    add_ev!(p4)

    # 5. Lefschetz proxy
    p5 = plot(t, s.lefsch, color=:teal, lw=0.8, label="",
        ylabel=yl("Lefschetz"), xlims=xl,
        xformatter=_->"", ytickfontsize=6, yguidefontsize=6,
        bottom_margin=-2Plots.mm)
    add_ev!(p5)

    # 6. Perverse sum
    p6 = plot(t, s.perv_n, color=:darkgreen, lw=0.8, label="",
        ylabel=yl("Perv sum"), xlims=xl,
        xformatter=_->"", ytickfontsize=6, yguidefontsize=6,
        bottom_margin=-2Plots.mm)
    add_ev!(p6)

    # 7. Normalised score
    p7 = plot(t, s.score, color=:brown, lw=0.8, label="",
        ylabel=yl("Score"), xlims=xl,
        xlabel=show_xlabel ? "Snapshot" : "",
        ytickfontsize=6, yguidefontsize=6,
        xtickfontsize=6, xguidefontsize=6)
    add_ev!(p7)

    p1, p2, p3, p4, p5, p6, p7
end

col1 = make_column(s1, "Phase 1  (m₀=0, flat A∞, $(length(s1.walls)) wall crossings)", true,  false)
col2 = make_column(s2, "Phase 2  (m₀=5.0, curved A∞, $(length(s2.walls)) wall crossings)", false, true)

# ── Interleave: p1_left, p1_right, p2_left, p2_right … ───────────────────────
panels = vec(hcat(collect(col1), collect(col2))')  # 14 panels, row-major

heights = repeat([0.165, 0.115, 0.115, 0.10, 0.10, 0.115, 0.115], inner=1)
# 7 rows × 2 cols: each row same height
row_h   = [0.18, 0.13, 0.13, 0.11, 0.11, 0.13, 0.11]
lo      = grid(7, 2, heights=row_h)

fig = plot(panels..., layout=lo, size=(1500, 1200),
    plot_title="Q7L: Phase 1 vs Phase 2 — A∞ obstruction dynamics",
    plot_titlefontsize=10,
    left_margin=10Plots.mm, right_margin=2Plots.mm,
    top_margin=1Plots.mm)

savefig(fig, "plot_master_timeline_comparison.png")
println("\nSaved: plot_master_timeline_comparison.png")
println("Phase 1: $(s1.n) snapshots, $(length(s1.walls)) wall crossings, $(length(s1.recoveries)) recovery events")
println("Phase 2: $(s2.n) snapshots, $(length(s2.walls)) wall crossings, $(length(s2.recoveries)) recovery events")
