function detect_multi_scale_transition(tower::MetaplecticTower,
                                        region::Vector{Int})
    transitions = Dict{Int,Float64}()  # level → transition time
    
    for k in 1:length(tower.levels)
        sheaf = tower.sheaves[k]
        
        # Compute dominance in region at level k
        R_k = map_coarse_region(region, k, tower.projection_maps)
        
        dominance_history = Float64[]
        for t in time_points
            # Get dopamine level at time t
            D = get_dopamine_at_time(t)
            
            # Compute order parameter R(t) at level k
            R = compute_dominance(sheaf, R_k, D)
            push!(dominance_history, R)
        end
        
        # Detect transition at this level
        t_c_k = find_crossing_time(dominance_history, threshold=0.5)
        transitions[k] = t_c_k
    end
    
    # Fit scaling law: t_c^(k) = t_c^(0) + Δ·k^γ
    levels = collect(1:length(tower.levels))
    times = [transitions[k] for k in levels]
    
    # Nonlinear fit
    model(t, p) = p[1] + p[2] * t.^p[3]  # t_c^(0) + Δ·k^γ
    params = curve_fit(model, levels, times, [times[1], 1.0, 1.0])
    
    return transitions, params.param
end
