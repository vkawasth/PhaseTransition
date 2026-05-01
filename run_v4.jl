#=
**********************************************************************************************************
The results between V3 (prime‑path monodromy) and V4 (Gerstenhaber monodromy) may differ 
moderately, but not drastically in most cases.

    Prime paths (V3) capture 6‑ary obstructions – they encode how the highest multiplication 
    (m6) interacts with regions.

    Gerstenhaber monodromy (V4) captures infinitesimal deformations (HH¹) – the Lie algebra of 
    derivations. This is a first‑order, linearised version of the obstruction landscape.

Because both are derived from the same algebra, they often point to similar “active” regions. 
However, there can be discrepancies:

    A small derivation weight may sum to a large projector on a region, while prime paths might 
    not involve that region.

    Conversely, a prime path could have huge weight but involve regions that are not significantly 
    acted upon by any derivation.

In your data (from earlier logs), prime paths and prime ideals both identified sAMY, HPF, HY, BLA, LA 
as active. Gerstenhaber will likely reinforce these, but the dominant region might sometimes differ 
(e.g., sAMY vs HY). The transport (region probability distribution) will evolve differently because 
the matrices are different.

Verdict: Expect qualitatively similar trends, but the ranking and the exact sequence of dominant regions 
may change. The differences are scientifically interesting – they highlight whether the obstruction is 
dominated by higher‑arity interactions (m6) or by infinitesimal symmetries (HH¹).

You can run both V3 and V4 on the same JSON files and compare the dominant region sequences. That’s the 
only way to see the actual difference for your specific connectome data.
**********************************************************************************************************
=#
using CairoMakie
using Graphs, GraphPlot, Plots   # for the static flip graph

include("OnlineAssociahedronNavigatorV4.jl")
using .OnlineAssociahedronNavigatorV4
include("AssociahedronFromAinf.jl")
using .AssociahedronFromAinf

# Associahedra can't be created even for graph of 6 nodes
"""
The enumeration of all subsets of tubes (2^57 ≈ 1.4e17) is impossible, so the above code will not run. 
We need a more efficient method to generate maximal tubings. Since your graph is only 6 vertices, 
the number of tubes is small (~57), but enumerating all subsets of tubes is still impossible. 
Instead, we can use the existing maximal_tubings from GraphAssociahedron (which you already have) 
– that function is already fast enough for N=6. So the best is to reuse that function by 
including the GraphAssociahedron module.
"""

function plot_associahedron()
    # Build the support graph (same as in navigator)
    G = Dict(
        :CA1sp => [:HPF, :sAMY],
        :BLA   => [:HPF, :LA, :sAMY],
        :HY    => [:sAMY],
        :HPF   => [:CA1sp, :BLA, :sAMY],
        :sAMY  => [:CA1sp, :BLA, :HY, :HPF, :LA],
        :LA    => [:BLA, :sAMY]
    )
    # symmetrize
    for v in keys(G)
        for u in G[v]
            if !(v in G[u])
                push!(G[u], v)
            end
        end
    end

    tubings = maximal_tubings(G)
    # Build flip graph
    n = length(tubings)
    g = SimpleGraph(n)
    for i in 1:n
        for j in i+1:n
            diff_i = setdiff(tubings[i], tubings[j])
            diff_j = setdiff(tubings[j], tubings[i])
            if length(diff_i) == 1 && length(diff_j) == 1
                add_edge!(g, i, j)
            end
        end 
    end         
            
    layout = spring_layout(g)
    graphplot(g, layout,
              nodecolor="lightblue",
              nodesize=0.2,
              edgewidth=0.5,
              title="Associahedron Flip Graph")
    savefig("associahedron_flip_graph.png")
    println("Saved associahedron flip graph to associahedron_flip_graph.png")
end


folder = "./"

# 57! = 2^57 ≈ 1.4e17, impossible on laptop -- use beam search
#G = build_support_graph()
#ALL_TUBINGS = maximal_tubings(G)
#S = run_folder_allcomb!(folder, ALL_TUBINGS)

S = run_folder!(folder)

save("scores.png", plot_scores(S))
save("transport.png", plot_transport(S))
save("tube_participation.png", plot_tubes(S))

plot_associahedron()
