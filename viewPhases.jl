using Plots

function plot_phase_transition(time_points, R_history, dopamine_level)
    p1 = plot(time_points, R_history,
              title="Theta-Alpha Dominance Phase Transition",
              xlabel="Time (s)", ylabel="R = (E_θ - E_α)/(E_θ + E_α)",
              label="Dominance Parameter",
              linewidth=2)
    
    # Add threshold lines
    hline!([0.5], linestyle=:dash, color=:red, label="Theta Dominance Threshold")
    hline!([-0.5], linestyle=:dash, color=:blue, label="Alpha Dominance Threshold")
    
    # Shade regions
    fillrange = fill(0.5, length(time_points))
    plot!(time_points, fillrange, fillrange=-0.5,
          fillalpha=0.2, fillcolor=:gray, label="Transition Region")
    
    # Add dopamine level
    p2 = plot(time_points, dopamine_level .* ones(length(time_points)),
              title="Dopamine Level",
              xlabel="Time (s)", ylabel="Dopamine (arb. units)",
              label="Dopamine", color=:green, linewidth=2)
    
    plot(p1, p2, layout=(2,1), size=(800,600))
end
