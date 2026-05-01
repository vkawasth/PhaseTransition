using CairoMakie
using Graphs, GraphPlot, Plots   # for the static flip graph

include("OnlineAssociahedronNavigatorV3.jl")
using .OnlineAssociahedronNavigatorV3
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
# 57! time.
#plot_associahedron()
