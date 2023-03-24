# """
# This is an experiment in machine assisted coding.
# I have prompted GTP3.5 as follows:
# ----
# using the SciML ecosystem, I wish to build a stochastic dynamical systems mode of the homicidal chauffeur problem. specifically please use sperate components representing the pursuer model and the evader model.  it should have the following logical elements

# - the environment is a 2 dimensional cartesian plane
# - the pursuer has sensors that allows it to directly observe its own position in the 2 dimensional cartesian plane
# - the pursuer has sensors that allow it a noisy observation of the relative position and velocity of the evader 
# - the pursuer estimates the(absolute) position and velocity of the evader
# - the pursuer has laws of motion which allow it to modify its angular velocity but up to a maximum value, and to modify its acceleration along its heading up to a particular maximum speed (magnitude of its vector velocity); given these dynamics its probably appropriate for the pursuers local coordinate system to be polar coordinates
# - pursuer must use its observations of itself and of the evader to attempt to mimimize the magnitude of the vector distance between the position of the evader and the pursuer.
# - the pursuer wins if this distance is converging to 0
# - conversely, the evader has sensors that allows it to directly observe its own position in the 2 dimensional cartesian plane
# - the evader has sensors that allow it a noisy observation of the relative position and velocity of the  pursuer 
# - the evader estimates the (absolute) position and velocity of the pursuer
# - the evader has laws of motion which allow it to modify its acceleration up to a maximum acceleration magnitude along ANY heading up to a particular maximum speed (magnitude of its vector velocity); given these dynamics its probably appropriate for the pursuers local coordinate system to be Cartesian Coordinates
# - evader must use its observations of itself and of the pursuerto attempt to maximize the magnitude of the vector distance between the position of the evader and the pursuer.
# - the evader wins if this distance is diverging

# Please create a Julia script for this stochastic dynamical system, please put the parameter choices at the top. the parameters include the number of time steps in the simulation, the stepsize, the pursuers max speed and max angular velocity, the evaders max acceleration and max speed. Additionally the variance of the obervation noise for both the pursuer and the evader.
# ----
# The following code was proposed by GTP3.5
# """

using DynamicalSystems
using DifferentialEquations
using OrdinaryDiffEq
using LinearAlgebra #added by me

# Parameters
num_steps = 1000  # number of time steps in the simulation
dt = 0.1  # step size
pursuer_max_speed = 5.0  # maximum speed of the pursuer
pursuer_max_ang_velocity = 1.0  # maximum angular velocity of the pursuer
evader_max_accel = 2.0  # maximum acceleration of the evader
evader_max_speed = 10.0  # maximum speed of the evader
pursuer_obs_noise = 0.1  # variance of the observation noise for the pursuer
evader_obs_noise = 0.1  # variance of the observation noise for the evader

# Define the pursuer model
function pursuer_dynamics!(pDx, x, p, t)
    # Unpack state variables
    px, py, pvx, pvy, ph = x
    # Unpack parameters
    max_speed, max_ang_vel, evader_pos, evader_vel, obs_noise = p
    
    # Calculate distance and bearing to evader
    dx = evader_pos[1] - px
    dy = evader_pos[2] - py
    r = hypot(dx, dy)
    theta = atan(dy, dx) - ph
    
    # Apply observation noise
    r += randn() * obs_noise
    theta += randn() * obs_noise
    
    # Estimate evader position and velocity
    evader_vx_est = evader_vel[1] + randn() * obs_noise
    evader_vy_est = evader_vel[2] + randn() * obs_noise
    evader_px_est = evader_pos[1] + evader_vx_est * r / pursuer_max_speed
    evader_py_est = evader_pos[2] + evader_vy_est * r / pursuer_max_speed
    
    # Control law to minimize distance to evader
    desired_heading = atan(evader_py_est - py, evader_px_est - px)
    ang_error = wrap_to_pi(desired_heading - ph)
    ang_vel = clamp(ang_error, -max_ang_vel, max_ang_vel)
    accel = clamp(max_speed - pvx, -max_speed / dt, max_speed / dt)

    # Update state variables
    pDx[1] = pvx * cos(ph) - pvy * sin(ph)
    pDx[2] = pvx * sin(ph) + pvy * cos(ph)
    pDx[3] = accel * cos(ph)
    pDx[4] = accel * sin(ph)
    pDx[5] = ang_vel

end

# """
# the results from GTP3.5 were cut off here, so I have reprompted it as follows

# ----
# the other seems to have to big of a request. let's break it up into pieces. return the code evader model `function evader_dynamics!`

# """

# Define the evader model
function evader_dynamics!(eDx, x, p, t)
    # Unpack state variables
    ex, ey, evx, evy = x
    # Unpack parameters
    max_accel, max_speed, pursuer_pos, pursuer_vel, obs_noise = p
    
    # Calculate distance and bearing to pursuer
    dx = pursuer_pos[1] - ex
    dy = pursuer_pos[2] - ey
    r = hypot(dx, dy)
    theta = atan(dy, dx)
    
    # Apply observation noise
    r += randn() * obs_noise
    theta += randn() * obs_noise
    
    # Estimate pursuer position and velocity
    pursuer_vx_est = pursuer_vel[1] + randn() * obs_noise
    pursuer_vy_est = pursuer_vel[2] + randn() * obs_noise
    pursuer_px_est = pursuer_pos[1] + pursuer_vx_est * r / evader_max_speed
    pursuer_py_est = pursuer_pos[2] + pursuer_vy_est * r / evader_max_speed
    
    # Control law to maximize distance to pursuer
    desired_accel = max_accel .* [cos(theta), sin(theta)] #reworked to handle scalar times vector
    dax = desired_accel[1]
    day = desired_accel[2]
    accel_error = [dax-evx, day-evy]
    accel = norm(accel_error) <= max_accel ? accel_error : normalize(accel_error) * max_accel
    speed_error = max_speed - hypot(evx, evy)
    speed = clamp(speed_error, -max_accel * dt, max_accel * dt) + hypot(evx, evy)
    
    # Update state variables
    eDx[1] = evx
    eDx[2] = evy
    eDx[3] = accel[1]
    eDx[4] = accel[2]

end

# Helper function to wrap an angle to the range [-pi, pi]
# this was missing from the GPT3 recommendation so i added it
function wrap_to_pi(angle)
    return atan(sin(angle), cos(angle))
end


# Initial conditions
pursuer_x0 = [0.0, 0.0, 0.0, 0.0, 0.0]  # x position, y position, x velocity, y velocity, heading
evader_x0 = [10.0, 10.0, 0.0, 0.0]  # x position, y position, x velocity, y velocity
timespan = (0.0, num_steps * dt)

# Define the full system
function homicidal_chauffeur!(Dx, x, p, t)
    # Unpack state variables
    px, py, pvx, pvy, ph = x[1:5]
    ex, ey, evx, evy = x[6:9]
    
    pDx = Dx[1:5]
    eDx = Dx[6:9]

    # Compute pursuer dynamics
    pursuer_pos = [px, py]
    pursuer_vel = [pvx, pvy]
    evader_pos = [ex, ey]
    evader_vel = [evx, evy]
    pursuer_params = (pursuer_max_speed, pursuer_max_ang_velocity, evader_pos, evader_vel, pursuer_obs_noise)
    pursuer_dynamics!(pDx, x[1:5], pursuer_params, t)
    if mod(t, 10) == 0
        println("Pursuer Dx: $pDx")
    end
    
    # Compute evader dynamics
    pursuer_pos = [px, py]
    pursuer_vel = [pvx, pvy]
    evader_pos = [ex, ey]
    evader_vel = [evx, evy]
    evader_params = (evader_max_accel, evader_max_speed, pursuer_pos, pursuer_vel, evader_obs_noise)
    evader_dynamics!(eDx, x[6:9], evader_params, t)
    if mod(t, 10) == 0
        println("Evader Dx: $eDx")
    end

    Dx[1:5] = pDx
    Dx[6:9] = eDx
end



# Define the full system as a continuous-time dynamical system
diffeq = (alg = Vern9(), abstol = 1e-9, reltol = 1e-9)
full_system = ContinuousDynamicalSystem(homicidal_chauffeur!, vcat(pursuer_x0, evader_x0),timespan;diffeq)

# Get the dimension of the state vector
dim = dimension(full_system)
println("Dimension of the state vector: $dim")

# Get the initial state of the system
init_state = initial_state(full_system)
println("Initial state of the system: $init_state")

# Simulate the system
#sol = solve(full_system) #solver not working so i used euler

# Define the simulation loop (simple Euler integration)
function euler()
    mystate = vcat(pursuer_x0, evader_x0)
    Dx = zeros(length(mystate))
    sol = zeros(length(mystate), num_steps)
    for i in 1:num_steps
        # Update the state using the dynamics function
        t = (i - 1) * dt
        homicidal_chauffeur!(Dx, mystate, [], t)
        
        if mod(t, 10) == 0
            println("Dx at time $t: $Dx")
        end

        mystate += Dx .* dt
      
        # Store the state in the solution array
        sol[:, i] = mystate
    
        # Print the state at every 10th time step
        if mod(t, 10) == 0
            println("State at time $t: $mystate")
            println("Dx at time $t: $Dx")
        end
    end
    
    return sol
end

sol = euler()

# """
# please provide code for the plot

#     position and velocity of both the pursuer and evader on the cartesian plane with absolute coordinates, position will be indicated by the position of an acute triangle, the direction of motion indicated by the direction the triangle points
# """

using Plots

# Extract pursuer and evader positions and velocities from solution
pursuer_pos = sol[1:2, :]
pursuer_vel = sol[3:4, :]
evader_pos = sol[6:7, :]
evader_vel = sol[8:9, :]

# Compute the error between pursuer and evader positions
err_mat = pursuer_pos - evader_pos

error = zeros(num_steps)
for i in 1:num_steps
    error[i] = norm(err_mat[:, i])
end

# Define time vector
t = collect(range(0, stop=num_steps*dt, length=num_steps))

# Plot pursuer and evader positions and velocities
plot1 = plot(title="Pursuer and Evader Positions and Velocities", xlabel="X", ylabel="Y", aspect_ratio=:equal)
quiver!(plot1, pursuer_pos[1, :], pursuer_pos[2, :], quiver=pursuer_vel, color=:red, label="Pursuer Velocity")
quiver!(plot1, evader_pos[1, :], evader_pos[2, :], quiver=evader_vel, color=:blue, label="Evader Velocity")
plot!(plot1, pursuer_pos[1, :], pursuer_pos[2, :], color=:red, label="Pursuer Position")
plot!(plot1, evader_pos[1, :], evader_pos[2, :], color=:blue, label="Evader Position")
plot!(plot1, legend=:bottomright)

# Plot time series of positions and error
plot2 = plot(title="Pursuer and Evader Position Time Series", xlabel="Time", ylabel="Position")
plot!(plot2, t, pursuer_pos[1, :], label="Pursuer X", color=:red)
plot!(plot2, t, evader_pos[1, :], label="Evader X", color=:blue)
plot!(plot2, t, pursuer_pos[2, :], label="Pursuer Y", color=:darkred)
plot!(plot2, t, evader_pos[2, :], label="Evader Y", color=:darkblue)
plot!(plot2, legend=:bottomright)

plot3 = plot(title="Error Time Series", xlabel="Time", ylabel="Error")
plot!(plot3, t, error, label="Error", color=:green)
plot!(plot3, legend=:bottomright)

# Display plots
plot(plot1, plot2, plot3, layout=(3, 1), size=(800, 800))


# using Plots

# # Extract pursuer and evader positions and velocities from solution
# pursuer_pos = sol[1:2, :]
# pursuer_vel = sol[3:4, :]
# evader_pos = sol[6:7, :]
# evader_vel = sol[8:9, :]

# # Plot pursuer and evader positions and velocities
# p1 = plot(pursuer_pos[1, :], pursuer_pos[2, :], color=:red, label="Pursuer Position")
# p2 = plot(pursuer_vel[1, :], pursuer_vel[2, :], color=:darkred, label="Pursuer Velocity")
# p3 = plot(evader_pos[1, :], evader_pos[2, :], color=:blue, label="Evader Position")
# p4 = plot(evader_vel[1, :], evader_vel[2, :], color=:darkblue, label="Evader Velocity")

# # Plot position and direction of motion for pursuer
# quiver!(p1, pursuer_pos[1, :], pursuer_pos[2, :], quiver=pursuer_vel, color=:red, legend=:none)
# plot!(p1, size=(600, 600), aspect_ratio=:equal, xlabel="X", ylabel="Y", title="Pursuer Position and Velocity")

# # Plot position and direction of motion for evader
# quiver!(p3, evader_pos[1, :], evader_pos[2, :], quiver=evader_vel, color=:blue, legend=:none)
# plot!(p3, size=(600, 600), aspect_ratio=:equal, xlabel="X", ylabel="Y", title="Evader Position and Velocity")

# # Show plots
# plot(p1, p2, p3, p4, layout=(2, 2))

# # Extract pursuer and evader positions and velocities from solution
# pursuer_pos = sol[1:2, :]
# pursuer_vel = sol[3:4, :]
# evader_pos = sol[6:7, :]
# evader_vel = sol[8:9, :]

# # Plot pursuer positions and velocities
# sp1 = plot(pursuer_pos[1, :], pursuer_pos[2, :], color=:red, label="Pursuer Position")
# sp2 = plot(pursuer_vel[1, :], pursuer_vel[2, :], color=:darkred, label="Pursuer Velocity")

# # Plot evader positions and velocities on same plot as pursuer
# plot!(sp1, evader_pos[1, :], evader_pos[2, :], color=:blue, label="Evader Position")
# quiver!(sp1, evader_pos[1, :], evader_pos[2, :], quiver=evader_vel, color=:darkblue, legend=:none)

# # Set axis labels and title
# xlabel!("X")
# ylabel!("Y")
# title!("Pursuer and Evader Position and Velocity")

# # Add a legend to the plot
# plot!(sp1, legend=:bottomright)

# # Show the plot
# display(sp1)

# using Plots

# # Extract pursuer and evader positions and velocities from solution
# pursuer_pos = sol[1:2, :]
# pursuer_vel = sol[3:4, :]
# evader_pos = sol[6:7, :]
# evader_vel = sol[8:9, :]

# # Plot pursuer and evader positions and velocities
# p1 = plot(pursuer_pos[1, :], pursuer_pos[2, :], color=:red, label="Pursuer Position")
# p2 = plot(evader_pos[1, :], evader_pos[2, :], color=:blue, label="Evader Position")

# # Plot position and direction of motion for pursuer
# quiver!(p1, pursuer_pos[1, :], pursuer_pos[2, :], quiver=pursuer_vel, color=:red, legend=:none)
# plot!(p1, size=(600, 600), aspect_ratio=:equal, xlabel="X", ylabel="Y", title="Pursuer and Evader Positions")

# # Plot position and direction of motion for evader
# quiver!(p2, evader_pos[1, :], evader_pos[2, :], quiver=evader_vel, color=:blue, legend=:none)
# plot!(p2, size=(600, 600), aspect_ratio=:equal, xlabel="X", ylabel="Y", title="")

# # Show plots
# plot(p1, p2, layout=(2, 1), legend=:topleft)

# # Compute the error between pursuer and evader positions
# error = vecnorm(pursuer_pos - evader_pos, dims=1)

# # Plot time series of positions and error
# t = collect(range(0, stop=num_steps*dt, length=num_steps))

# p3 = plot(t, pursuer_pos[1, :], label="Pursuer X", color=:red, ylabel="Position", xlabel="Time")
# plot!(p3, t, evader_pos[1, :], label="Evader X", color=:blue)
# p4 = plot(t, pursuer_pos[2, :], label="Pursuer Y", color=:red, ylabel="Position", xlabel="Time")
# plot!(p4, t, evader_pos[2, :], label="Evader Y", color=:blue)

# p5 = plot(t, error, label="Error", color=:green, ylabel="Error", xlabel="Time")

# plot(p3, p4, p5, layout=(3, 1), legend=:bottomright)
