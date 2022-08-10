```julia
using LinearAlgebra
using Random
using Statistics
using Plots

using GomalizingFlow
using GomalizingFlow: PhysicalParams, HyperParams, ScalarPhi4Action
using Flux
using Flux.Zygote
using ProgressMeter
using Flux: unsqueeze
```

```julia
configpath = joinpath(pkgdir(GomalizingFlow), "cfgs", "example3d.toml")
hp = GomalizingFlow.load_hyperparams(configpath);
pp = hp.pp
```

```julia
module My

using GomalizingFlow: PhysicalParams

struct HMC{T <: AbstractFloat, N}
    cfgs::Array{T,N} # configurations
    p::Array{T,N} # momentum
    cfgs_old::Array{T,N} # configurations
    p_old::Array{T,N} # momentum
    F::Array{T,N} # Force
end

function HMC{T}(pp::PhysicalParams; init::Function=zeros) where T<:AbstractFloat    
    lattice_shape = pp.lattice_shape
    HMC{T, length(lattice_shape)}(
        init(lattice_shape...),
        init(lattice_shape...),
        init(lattice_shape...),
        init(lattice_shape...),
        init(lattice_shape...),
    )
end

HMC(pp::PhysicalParams; kwargs...) = HMC{Float64}(pp::PhysicalParams; kwargs...)

Base.eltype(::HMC{T, N}) where {T, N} = T

end # module

using .My
```

```julia
function calc_potential(Φ::My.HMC{T, N}, pp::PhysicalParams) where {T, N}
    Lx, Ly, Lz = pp.lattice_shape
    m²=T(pp.m²)
    λ =T(pp.λ)
    ϕ = Φ.cfgs
    p = Φ.p
    V = zero(T)
    for iz=1:Lz
        for iy=1:Ly
            for ix=1:Lx
                V += m² * ϕ[ix,iy,iz]^2+ λ * ϕ[ix,iy,iz]^4
            end
        end
    end
    return V
end
```

```julia
function calc_kinetic(Φ::My.HMC{T, N},pp::PhysicalParams) where {T, N}
    Lx, Ly, Lz = pp.lattice_shape
    ϕ = Φ.cfgs
    K = zero(T)
    for iz=1:Lz
        for iy=1:Ly
            for ix=1:Lx
                ip=ix+1
                im=ix-1
                if ip>Lx #  1<= ip <= L 
                    ip-=Lx
                end
                if im<1
                    im+=Lx
                end
                K+=ϕ[ix,iy,iz]*(2ϕ[ix,iy,iz] - ϕ[ip,iy,iz]-ϕ[im,iy,iz])
                # - - - - - - - - - - - - - - - - - - - -
                ip=iy+1
                im=iy-1
                if ip>Ly
                    ip-=Ly
                end
                if im<1
                    im+=Ly
                end
                K+=ϕ[ix,iy,iz]*(2ϕ[ix,iy,iz] - ϕ[ix,ip,iz]-ϕ[ix,im,iz])
                # - - - - - - - - - - - - - - - - - - - -
                ip=iz+1
                im=iz-1
                if ip>Lz
                    ip-=Lz
                end
                if im<1
                    im+=Lz
                end
                K+=ϕ[ix,iy,iz]*(2ϕ[ix,iy,iz] - ϕ[ix,iy,ip]-ϕ[ix,iy,im])
            end
        end
    end
    return K
end
```

```julia
function calcgreen(cfgs::AbstractArray, pp::PhysicalParams)
    example_loc = CartesianIndex(repeat([1], ndims(cfgs))...)
    volume = prod(pp.lattice_shape)
    #             sum          vec
    # (Lx, Ly, Lz) -> (1,1,Lz) -> (Lz)
    return sum(
        cfgs[example_loc] * cfgs,
        dims=1:(ndims(cfgs)-1)
    )/volume |> vec
end
```

```julia
Φ = My.HMC(pp, init=rand);
```

```julia
action = GomalizingFlow.ScalarPhi4Action(pp.m², pp.λ)
@time action(unsqueeze(Φ.cfgs, dims=ndims(Φ.cfgs)+1))[begin]
```

```julia
@assert calc_potential(Φ, pp) ≈ sum(action.m² * Φ.cfgs .^2 + action.λ * Φ.cfgs .^ 4)
@assert calc_kinetic(Φ, pp) + calc_potential(Φ, pp) ≈ action(unsqueeze(Φ.cfgs, dims=ndims(Φ.cfgs)+1))[begin]
```

```julia
function ∂potential(action::ScalarPhi4Action{T}, cfgs::AbstractArray{T, N}) where {T, N}
    @. 2action.m² * cfgs + 4*action.λ * cfgs ^ 3
end

function ∂kinetic(cfgs::AbstractArray{T, N}) where {T, N}
    ϕ = cfgs
    F = zero(ϕ)
    sz = ndims(cfgs)
    F .+= sz * 4cfgs
    F .-= sum(2circshift(cfgs, -Flux.onehot(μ, 1:sz)) for μ in 1:sz)
    F .-= sum(2circshift(cfgs, +Flux.onehot(μ, 1:sz)) for μ in 1:sz)
    return F
end
```

```julia
S(cfgs::AbstractArray) = sum(action(unsqueeze(cfgs, dims=ndims(cfgs)+1)))
∂S(cfgs::AbstractArray) = ∂kinetic(cfgs) + ∂potential(action, cfgs)

cfgs2d = rand(3,3)
@assert gradient(S, cfgs2d)[begin] ≈ ∂S(cfgs2d)
cfgs3d = rand(3,3,3)
@assert gradient(S, cfgs3d)[begin] ≈ ∂S(cfgs3d)

function hamiltonian(S::Function, cfgs, p)
    Σ_p² = dot(p, p)
    H = 0.5Σ_p² + S(cfgs)
    return H
end
```

```julia
# Solve Molecular Dynamics
# a.k.a leapfrog algorithm 
function md!(∂S::Function, x, p, Nτ, Δτ)
    @. x += Δτ/2 * p
    for _ in 1:(Nτ-1)
        ∇ = ∂S(x)
        @. p -= Δτ * ∇
        @. x += Δτ * p
    end
    ∇ = ∂S(x)
    @. p -= Δτ * ∇
    @. x += Δτ/2 * p
    return x , p
end

function hmc_update(S::Function, ∂S::Function, x, Nτ, Δτ)
    p = randn(eltype(x), size(x)...)

    x_init = copy(x)
    p_init = copy(p)
    H_init = hamiltonian(S, x_init, p_init)

    x_cand, p_cand = md!(∂S, x, p, Nτ, Δτ)
    H_cand = hamiltonian(S, x_cand, p_cand)

    ΔH = H_cand - H_init
    r = rand()
    accepted, x_next = (r < exp(-ΔH)) ? (true, x_cand) : (false, x_init)
    #@show accepted, exp(-ΔH), mean(x_cand), mean(x_init)
    return (accepted, x_next, ΔH)
end
```

```julia
function runHMC(pp::PhysicalParams; ntrials, Nτ, Δτ)
    T = Float32
    cfgs = rand(T, pp.lattice_shape...)
    action = GomalizingFlow.ScalarPhi4Action{T}(pp.m², pp.λ)
    function S(cfgs::AbstractArray)
        out = sum(action(unsqueeze(cfgs, dims=ndims(cfgs)+1)))
        return out
    end
    ∂S(cfgs::AbstractArray) = ∂kinetic(cfgs) + ∂potential(action, cfgs)
    
    T = eltype(cfgs) # e.g. Float64
    history = (cfgs=typeof(cfgs)[], cond=T[], ΔH=T[], accepted=Bool[], Green=Vector{T}[])
    
    @showprogress for _ in 1:ntrials
        accepted, cfgs, ΔH = hmc_update(S, ∂S, cfgs, Nτ, Δτ)
        cond = mean(cfgs)
        push!(history.accepted, accepted)
        push!(history.cond, cond)
        push!(history.ΔH, ΔH)
        push!(history.cfgs, cfgs)
        push!(history.Green, calcgreen(cfgs, pp))
    end
    return history
end
```

```julia
configpath = joinpath(pkgdir(GomalizingFlow), "cfgs", "example3d.toml")
hp = GomalizingFlow.load_hyperparams(configpath);
pp = hp.pp
@show pp
Nτ = 20
Δτ = 0.01
ntrials = 10 ^ 4
Random.seed!(54321)
history = runHMC(pp; ntrials, Nτ, Δτ);
```

```julia
mean(exp.(-history.ΔH)) ≈ 1f0
```

```julia
plot(history.cond)
```

```julia
plot(history[:accepted], title="$(mean(history.accepted))")
```
