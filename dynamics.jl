function dynamics!(dx, x, params, t)
    # Extract state variables
    pursuer_pos, pursuer_vel, evader_pos, evader_vel, area_pursuer, area_evader, capture_distance = params
    dp = norm(pursuer_pos - evader_pos)  # Distance between pursuer and evader

    # Compute control inputs
    if dp <= area_pursuer/2 + area_evader/2 + capture_distance
        # Evader is captured
        dx[1:4] .= [0.0, 0.0, 0.0, 0.0]
        dx[5:8] .= [0.0, 0.0, 0.0, 0.0]
    else
        # Evader is not captured
        ev_dir = evader_pos - pursuer_pos
        ev_dir /= norm(ev_dir)
        dx[1:2] .= pursuer_vel
        dx[3:4] .= area_pursuer/2 * ev_dir
        dx[5:6] .= evader_vel
        dx[7:8] .= area_evader/2 * ev_dir
    end
    
    return nothing
end

