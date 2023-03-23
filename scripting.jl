using DifferentialEquations
using Plots

function drift(du, u, p, t)
    du .= -p.λ .* u
end

function diffusion(du, u, p, t)
    du .= p.σ
end

u0 = [1.0]  # Initial condition
tspan = (0.0, 10.0)  # Time interval
p = (λ = 1.0, σ = 0.5)  # Model parameters

sde_problem = SDEProblem(drift, diffusion, u0, tspan, p)

sol = solve(sde_problem, EM(), dt=0.01)

plot(sol, xlabel="Time", ylabel="Value", title="Solution to Stochastic Differential Equation", legend=false)

