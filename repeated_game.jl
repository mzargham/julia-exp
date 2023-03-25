#provide a refactor of the following code that allows me 
#to run experiments where change the initial conditions 
# and the parameters and run experiments.
# What i want is to get a data set where 
# the initial conditions and parameter values are the features 
#and the outcome is labeled as -1 if the evader manages to intersect 
#the pursuer and +1 if the evader causes the distance between them to diverge. 
#The label 0 is reserved for the case where the two agents orbit each other 
#without intersecting. in order to make this work it is also important 
#to add parameters defining the are of the two agents. 
#Assume the pursuer is a rectangle in the ration 2:1 with the 
#long dimension being the direction of motion (like a car); 
#the evader is a circle. the area of each of the agents needs
# to be added as a parameter

using DifferentialEquations, LinearAlgebra
include("dynamics.jl")


"""
    dynamics_model(u, p, t)

Solves the differential equations for the pursuit-evasion problem.

# Arguments
- `u`: State vector containing the pursuer's and evader's positions and velocities.
- `p`: Model parameters.
- `t`: Current time.

# Returns
- `du`: Time derivative of `u`.
"""
function dynamics_model(du, u, p, t)
    # Extract positions and velocities
    pursuer_pos = u[1:2]
    pursuer_vel = u[3:4]
    evader_pos = u[5:6]

    # Compute direction of motion for pursuer
    dir_pursuer = normalize(pursuer_vel)

    # Compute direction from pursuer to evader
    dir_evader = normalize(evader_pos - pursuer_pos)

    # Compute distance between pursuer and evader
    dist = norm(evader_pos - pursuer_pos)

    # Determine whether evader is inside pursuer's capture range
    inside_capture_range = abs(dot(dir_pursuer, dir_evader)) > sqrt(2)/2 && dist <= p[:capture_range]

    # Determine whether evader is inside pursuer's kill range
    inside_kill_range = dist <= p[:kill_range]

    # Compute accelerations for pursuer and evader
    if inside_capture_range
        # Evader is captured
        du[1:4] = [0; 0; 0; 0]
        du[5:6] = [0; 0]
    elseif inside_kill_range
        # Evader is killed
        du[1:4] = [0; 0; 0; 0]
        du[5:6] = [0; 0]
    else
        # Evader is outside capture and kill ranges
        accel_pursuer = p[:pursuer_speed] * dir_pursuer
        accel_evader = p[:evader_speed] * dir_evader
        du[1:2] = pursuer_vel
        du[3:4] = accel_pursuer
        du[5:6] = accel_evader
    end
end

"""
    simulate_dynamics(ic, params, tspan)

Simulates the pursuit-evasion problem.

# Arguments
- `ic`: Initial conditions for pursuer and evader positions and velocities.
- `params`: Model parameters.
- `tspan`: Time span for simulation.

# Returns
- `sol`: Solution to the differential equations.
"""
function simulate_dynamics(ic, params, tspan)
    prob = ODEProblem(dynamics_model, ic, tspan, params)
    sol = solve(prob, Tsit5(), saveat=0.01)
    return sol
end

"""
    area_rectangle(width, height)

Computes the area of a rectangle with the given width and height.

# Arguments
- `width`: Width of the rectangle.
- `height`: Height of the rectangle.

# Returns
- `area`: Area of the rectangle.
"""
function area_rectangle(width, height)
    return width * height
end

"""
    area_circle(radius)

Computes the area of a circle with the given radius.

# Arguments
- `radius`: Radius of the circle.

# Returns
- `area`: Area of the circle.
"""
function area_circle(radius)
    return π * radius^2
end

using DifferentialEquations

"""
    generate_dataset(num_samples::Int, tspan::Tuple, area_pursuer::Float64, 
                    area_evader::Float64, pursuer_init::Vector, evader_init::Vector,
                    pursuer_max_speed::Float64, evader_max_speed::Float64)

This function generates a labeled dataset of num_samples simulations of the pursuer-evader
system over the time interval tspan. The initial conditions of the pursuer and evader
are given by the vectors pursuer_init and evader_init, respectively. The function returns
a matrix of features and a vector of corresponding labels, where each row of the feature matrix
corresponds to one simulation and has the following columns:
- pursuer initial x-position
- pursuer initial y-position
- pursuer initial x-velocity
- pursuer initial y-velocity
- evader initial x-position
- evader initial y-position
- evader initial x-velocity
- evader initial y-velocity
- area of the pursuer
- area of the evader
- the distance between the pursuer and evader at the end of the simulation
The label of each simulation is -1 if the evader intersects the pursuer during the simulation,
+1 if the evader manages to evade the pursuer, and 0 if the two agents simply orbit each other.

"""
function generate_dataset(num_samples::Int, tspan::Tuple, area_pursuer::Float64, area_evader::Float64,
                           pursuer_init::Vector, evader_init::Vector, pursuer_max_speed::Float64,
                           evader_max_speed::Float64)
    
    features = zeros(num_samples, 11)
    labels = zeros(num_samples)
    
    for i in 1:num_samples
        
        # Generate random initial positions for the pursuer and evader
        pursuer_pos = [pursuer_init[1] + rand() - 0.5, pursuer_init[2] + rand() - 0.5]
        evader_pos = [evader_init[1] + rand() - 0.5, evader_init[2] + rand() - 0.5]

        # Generate random initial velocities for the pursuer and evader
        pursuer_vel = [pursuer_init[3] + rand() - 0.5, pursuer_init[4] + rand() - 0.5]
        evader_vel = [evader_init[3] + rand() - 0.5, evader_init[4] + rand() - 0.5]

        # Solve the system of ODEs for the given initial conditions
        sol = solve(pursuer_evader, (0.0, tspan[2]), [pursuer_pos[1], pursuer_pos[2], pursuer_vel[1], pursuer_vel[2],
                area_pursuer, evader_pos[1], evader_pos[2], evader_vel[1], evader_vel[2], area_evader,
                pursuer_max_speed, evader_max_speed])
        
        # Extract final positions of pursuer and evader from solution
        pursuer_final_pos = sol.u[end][1:2]
        evader_final_pos = sol.u[end][6:7]
        
        # Compute distance between pursuer and evader at final time step
        dist = norm(pursuer_final_pos - evader_final_pos)
        
        # Classify the outcome based on the distance between pursuer and evader
        if dist < (area_pursuer + area_evader)/2
            labels[i] = -1 # evader caught pursuer
        elseif dist > 2 * (area_pursuer + area_evader)
            labels[i] = 1 # evader successfully evaded pursuer
        else
            labels[i] = 0 # pursuer and evader orbiting each other
        end
        
        # Save features for this example
        features[i, 1:2] = pursuer_pos
        features[i, 3:4] = evader_pos
        features[i, 5:6] = pursuer_vel
        features[i, 7:8] = evader_vel
        features[i, 9] = area_pursuer
        features[i, 10] = area_evader
        features[i, 11] = tspan[2]
        
    end
    
    return features, labels
end


function simulate_capture(pursuer_pos0, pursuer_vel0, evader_pos0, evader_vel0, area_pursuer, area_evader, capture_distance, t_final, dt)
    # Initialize state variable
    x = zeros(11)
    x[1:4] .= [pursuer_pos0; pursuer_vel0]
    x[5:8] .= [evader_pos0; evader_vel0]
    params = [pursuer_pos0, pursuer_vel0, evader_pos0, evader_vel0, area_pursuer, area_evader, capture_distance]
    
    # Define ODE problem
    prob = ODEProblem(dynamics!, x, (0.0, t_final), params)
    
    # Solve ODE problem
    sol = solve(prob, Tsit5(), saveat=0:dt:t_final)
    
    # Determine outcome
    dp = norm(sol[1:2,:] .- sol[6:7,:])
    if any(dp .<= capture_distance)
        outcome = -1
    elseif all(dp .> capture_distance)
        outcome = 1
    else
        outcome = 0
    end
    
    return outcome
end


# Test of the above
# pursuer_pos0 = [0.0, 0.0]
# pursuer_vel0 = [1.0, 0.0]
# evader_pos0 = [5.0, 0.0]
# evader_vel0 = [-1.0, 0.0]
# area_pursuer = 2.0
# area_evader = 1.0
# capture_distance = 0.1
# t_final = 20.0
# dt = 0.01

# simulate_capture(pursuer_pos0, pursuer_vel0, evader_pos0, evader_vel0, area_pursuer, area_evader, capture_distance, t_final, dt)

# Define area of pursuer and evader
area_pursuer = 20.0 # m^2
area_evader = area_circle(1.5) # m^2

# Define capture distance
capture_distance = 2.0 # m

# Define the ranges of initial conditions to test
pursuer_pos_range = -5:5
pursuer_vel_range = -2:2
evader_pos_range = -5:5
evader_vel_range = -2:2

# Define parameters for the simulation
area_pursuer = 6.0
area_evader = 1.0
capture_distance = 0.1
t_final = 100.0
dt = 0.1

# Collect simulation outcomes in a matrix
num_pursuer_pos = length(pursuer_pos_range)
num_pursuer_vel = length(pursuer_vel_range)
num_evader_pos = length(evader_pos_range)
num_evader_vel = length(evader_vel_range)

data = zeros(num_pursuer_pos*num_pursuer_vel*num_evader_pos*num_evader_vel, 7)
ind = 0

for i_pursuer_pos in 1:num_pursuer_pos
    pursuer_pos0 = pursuer_pos_range[i_pursuer_pos]
    for i_pursuer_vel in 1:num_pursuer_vel
        pursuer_vel0 = pursuer_vel_range[i_pursuer_vel]
        for i_evader_pos in 1:num_evader_pos
            evader_pos0 = evader_pos_range[i_evader_pos]
            for i_evader_vel in 1:num_evader_vel
                global ind += 1
                evader_vel0 = evader_vel_range[i_evader_vel]
                outcome = simulate_capture([pursuer_pos0; 0.0], [pursuer_vel0; 0.0], [evader_pos0; 0.0], [evader_vel0; 0.0], area_pursuer, area_evader, capture_distance, t_final, dt)
                data[ind,:] = [pursuer_pos0, pursuer_vel0, evader_pos0, evader_vel0, area_pursuer, area_evader, outcome]
            end
        end
    end
end



using CSV, Tables

# Convert data matrix to table
data_table = Tables.table(data, header=["pursuer_pos", "pursuer_vel", "evader_pos", "evader_vel", "area_pursuer", "area_evader", "outcome"])

# Write table to CSV file
CSV.write("simulation_data.csv", data_table)
