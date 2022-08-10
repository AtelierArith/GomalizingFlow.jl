```julia
using Flux
using Flux.Zygote
using BenchmarkTools
using LinearAlgebra
```

```julia
mutable struct ScalarPhi4Action{T<:AbstractFloat}
    m²::T
    λ::T
end

function calc_action(
    action::ScalarPhi4Action{T},
    cfgs::AbstractArray{T,N},
) where {T<:AbstractFloat,N}
    potential = @. action.m² * cfgs^2 + action.λ * cfgs^4
    sz = length(size(cfgs))
    Nd = sz - 1 # exclude last axis
    k1 = sum(2cfgs .^ 2 for μ in 1:Nd)
    k2 = sum(cfgs .* circshift(cfgs, -Flux.onehot(μ, 1:sz)) for μ in 1:Nd)
    k3 = sum(cfgs .* circshift(cfgs, Flux.onehot(μ, 1:sz)) for μ in 1:Nd)
    action_density = potential .+ k1 .- k2 .- k3
    dropdims(
        sum(action_density, dims=1:Nd),
        dims=Tuple(1:Nd),
    )
end

function calc_action2(
    action::ScalarPhi4Action{T},
    cfgs::AbstractArray{T,N},
) where {T<:AbstractFloat,N}
    potential = @. action.m² * cfgs^2 + action.λ * cfgs^4
    sz = length(size(cfgs))
    Nd = sz - 1 # exclude last axis
    B = size(cfgs, sz) # batch size
    k1 = Nd * 2cfgs .^ 2
    k2 = sum(cfgs .* circshift(cfgs, -Flux.onehot(μ, 1:sz)) for μ in 1:Nd)
    k3 = sum(cfgs .* circshift(cfgs, Flux.onehot(μ, 1:sz)) for μ in 1:Nd)
    action_density = potential .+ k1 .- k2 .- k3
    reshape(
        sum(action_density, dims=1:Nd),
        B
    )
end

function calc_action2_1(
    action::ScalarPhi4Action{T},
    cfgs::AbstractArray{T,N},
) where {T<:AbstractFloat,N}
    cfgs² = cfgs .^ 2
    potential = @. action.m² * cfgs² + action.λ * cfgs² ^ 2
    sz = ndims(cfgs)
    Nd = sz - 1 # exclude last axis
    B = size(cfgs, sz) # batch size
    k1 = Nd * 2cfgs²
    k2 = sum(cfgs .* circshift(cfgs, -Flux.onehot(μ, 1:sz)) for μ in 1:Nd)
    k3 = sum(cfgs .* circshift(cfgs, Flux.onehot(μ, 1:sz)) for μ in 1:Nd)
    action_density = @. potential + k1 - k2 - k3
    reshape(
        sum(action_density, dims=1:Nd),
        B
    )
end

"""
Fast impl but cannot be differentiable
"""
function calc_action3(
    action::ScalarPhi4Action{T},
    cfgs::AbstractArray{T,N},
) where {T<:AbstractFloat,N}
    potential = @. action.m² * cfgs^2 + action.λ * cfgs^4
    sz = length(size(cfgs))
    Nd = sz - 1 # exclude last axis
    B = size(cfgs, sz) # batch size
    k1 = Nd .* 2cfgs .^ 2
    dest = similar(cfgs)
    k2 = zero(cfgs)
    for μ in 1:Nd
        circshift!(dest, cfgs, -Flux.onehot(μ, 1:sz))
        @. k2 += cfgs * dest
    end
    k3 = zero(cfgs)
    for μ in 1:Nd
        circshift!(dest, cfgs, Flux.onehot(μ, 1:sz))
        @. k3 += cfgs * dest
    end
    action_density = potential .+ k1 .- k2 .- k3
    reshape(
        sum(action_density, dims=1:Nd),
        B
    )
end

function (action::ScalarPhi4Action{T})(
    cfgs::AbstractArray{T,N},
) where {T<:AbstractFloat,N}
    calc_action(action, cfgs)
end

```

```julia
T = Float32
action = ScalarPhi4Action{T}(1, 1)

L = 16
cfgs = rand(T, L, L, L, 1) |> cpu;

@assert calc_action(action, cfgs) ≈ calc_action2(action, cfgs) sum(calc_action2(action, cfgs) .- calc_action(action, cfgs))
@assert calc_action2(action, cfgs) ≈ calc_action2_1(action, cfgs) sum(calc_action2_1(action, cfgs) .- calc_action2(action, cfgs))
@assert calc_action2(action, cfgs) ≈ calc_action3(action, cfgs) sum(calc_action3(action, cfgs) .- calc_action2(action, cfgs))

@btime calc_action($action, $cfgs)
@btime calc_action2($action, $cfgs)
@btime calc_action2_1($action, $cfgs)

@btime gradient(sum ∘ Base.Fix1(calc_action, action), $cfgs)
@btime gradient(sum ∘ Base.Fix1(calc_action2, action), $cfgs)
@btime gradient(sum ∘ Base.Fix1(calc_action2_1, action), $cfgs);
```

```julia
T = Float32
action = ScalarPhi4Action{T}(1, 1)

L = 16
cfgs = rand(T, L, L, L, 64) |> cpu;

@assert calc_action(action, cfgs) ≈ calc_action2(action, cfgs) sum(calc_action2(action, cfgs) .- calc_action(action, cfgs))
@assert calc_action2(action, cfgs) ≈ calc_action2_1(action, cfgs) sum(calc_action2_1(action, cfgs) .- calc_action2(action, cfgs))
@assert calc_action2(action, cfgs) ≈ calc_action3(action, cfgs) sum(calc_action3(action, cfgs) .- calc_action2(action, cfgs))

@btime calc_action($action, $cfgs)
@btime calc_action2($action, $cfgs)
@btime calc_action2_1($action, $cfgs)

@btime gradient(sum ∘ Base.Fix1(calc_action, action), $cfgs)
@btime gradient(sum ∘ Base.Fix1(calc_action2, action), $cfgs)
@btime gradient(sum ∘ Base.Fix1(calc_action2_1, action), $cfgs);
```
