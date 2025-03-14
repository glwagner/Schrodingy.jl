module Schrodingy

using Oceananigans
using Oceananigans: AbstractModel

using Oceananigans.Architectures: architecture
using Oceananigans.Advection: div_Uc
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: AbstractGrid, xnode, ynode, znode
using Oceananigans.Fields: Center, Face, ZeroField
using Oceananigans.Operators
using Oceananigans.TimeSteppers: Clock
using Oceananigans.Utils: launch!

using KernelAbstractions

import Oceananigans.Simulations: prettytime, iteration, reset!, initialize!
import Oceananigans.OutputWriters: default_included_properties
import Oceananigans.Models: fields, prognostic_fields

struct SchrodingerModel{Ar, Gr, Cl, Ad, F, Ve, Au, Ti, Pa} <: AbstractModel{Nothing, Ar}
    architecture :: Ar
    grid :: Gr
    clock :: Cl
    advection :: Ad
    action :: F
    phase :: F
    velocities :: Ve
    propagation_velocities :: Ve
    auxiliaries :: Au
    timestepper :: Ti
    parameters :: Pa
end

function SchrodingerModel(grid, parameters)
    arch = architecture(grid)
    FT = eltype(grid)
    clock = Clock{FT}(time=0)
    advection = WENO()
    action = CenterField(grid)
    phase = CenterField(grid)
    u = XFaceField(grid)
    v = YFaceField(grid)
    w = ZeroField(grid)
    velocities = (; u, v, w)

    uw = XFaceField(grid)
    vw = YFaceField(grid)
    ww = ZeroField(grid)
    propagation_velocities = (; u, v, w)

    auxiliaries = nothing
   
    return SchrodingerModel(arch,
                            grid,
                            clock,
                            advection,
                            action,
                            phase,
                            velocities,
                            propagation_velocities,
                            auxiliaries,
                            nothing,
                            parameters)
end

const SM = SchrodingerModel

Base.summary(::SM) = "SchrodingerModel"
prettytime(m::SM) = prettytime(m.clock.time)
iteration(m::SM) = m.clock.iteration
Base.show(io::IO, m::SM) = print(io, summary(m))
        
reset!(::SM) = nothing
initialize!(::SM) = nothing
default_included_properties(::SM) = tuple(:grid)

# Fallback
function fields(model::SM)
    A = model.action
    θ = model.phase
    return (; A, θ)
end

prognostic_fields(model::SM) = fields(model)

#####
##### Some computations
#####

#####
##### Time step
#####

import Oceananigans.TimeSteppers: time_step!

function time_step!(model::SM, Δt)
    fill_halo_regions!(fields(model))

    grid = model.grid
    arch = architecture(grid)

    launch!(arch, grid, :xy,
            _compute_propagation_velocities!,
            model.propagation_velocities,
            grid,
            model.velocities,
            model.action,
            model.phase,
            model.parameters)

    fill_halo_regions!(model.propagation_velocities)

    launch!(arch, grid, :xy,
            _step_forward!,
            model.action,
            model.phase,
            grid,
            Δt,
            model.advection,
            model.velocities,
            model.parameters)

    return nothing
end

@kernel function _compute_propagation_velocities!(Uw, grid, U, A, θ, p)
    i, j = @index(Global, NTuple)
    k = 1

    (; k₀, ℓ₀, ω₀, g) = p
    uw, vw = Uw
    u, v = U

    cx = 0.0 #ω₀ / 2k₀
    cy = 0.05 #ω₀ / 2ℓ₀

    @inbounds begin
        uw[i, j, 1] = cx + u[i, j, 1]
        vw[i, j, 1] = cy + v[i, j, 1]
    end
end

@kernel function _step_forward!(A, θ, grid, Δt, advection, propagation_velocities, p)
    i, j = @index(Global, NTuple)
    k = 1

    @inbounds begin
        GA = - div_Uc(i, j, k, grid, advection, propagation_velocities, A)
        Gθ = 0 #- div_Uc(i, j, k, grid, advection, propagation_velocities, A)

        A[i, j, k] += Δt * GA
        θ[i, j, k] += Δt * Gθ
    end
end

end # module Schrodingy

