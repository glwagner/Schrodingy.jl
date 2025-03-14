using Schrodingy
using Oceananigans
using Oceananigans.Units
using GLMakie

Nt = 4e3
Nx = Ny = 128
Lx = Ly = 1000
grid = RectilinearGrid(size = (Nx, Ny),
                       x = (-Lx/2, Lx/2),
                       y = (-Ly/2, Ly/2),
                       topology = (Periodic, Periodic, Flat))

k₀ = 0.0
ℓ₀ = 0.05
g = 9.81
K₀ = sqrt(k₀^2 + ℓ₀^2)
ω₀ = sqrt(g * K₀)
parameters = (; k₀, ℓ₀, ω₀, g)

model = Surfinger.SchrodingerModel(grid, parameters)

A(x, y) = exp(-(x^2 + y^2) / (Ly^2 / 64))
u(x, y) = exp(-(y - Ly/4)^2 / (Ly^2 / 32))
set!(model.action, A)
set!(model.velocities.u, 1)
Oceananigans.BoundaryConditions.fill_halo_regions!(fields(model))

U = sqrt(5)
Δx = Lx / Nx
Δt = 1e-2 * Δx / U

fig = Figure()
ax0 = Axis(fig[1, 1], aspect=1)
ax1 = Axis(fig[1, 2], aspect=1)
heatmap!(ax0, model.action)

Nt = 400
for n = 1:Nt
    time_step!(model, Δt)
end

heatmap!(ax1, model.action)
display(fig)
