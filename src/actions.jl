using ChainRulesCore

mutable struct ScalarPhi4Action{T<:AbstractFloat}
    m²::T
    λ::T
end

function calc_action(
    action::ScalarPhi4Action{T},
    cfgs::AbstractArray{T,N},
) where {T<:AbstractFloat,N}
    cfgs² = cfgs .^ 2
    potential = @. action.m² * cfgs² + action.λ * cfgs²^2
    sz = ndims(cfgs)
    Nd = sz - 1 # exclude last axis
    B = size(cfgs, sz) # batch size
    k1 = Nd * 2cfgs²
    k2 = sum(cfgs .* circshift(cfgs, -Flux.onehot(μ, 1:sz)) for μ in 1:Nd)
    k3 = sum(cfgs .* circshift(cfgs, Flux.onehot(μ, 1:sz)) for μ in 1:Nd)
    action_density = @. potential + k1 - k2 - k3
    reshape(
        sum(action_density, dims=1:Nd),
        B,
    )
end

function (action::ScalarPhi4Action{T})(
    cfgs::AbstractArray{T,N},
) where {T<:AbstractFloat,N}
    calc_action(action, cfgs)
end

function ∂kinetic(cfgs::AbstractArray{T,N}) where {T,N}
    F = zero(cfgs)
    sz = ndims(cfgs)
    Nd = sz - 1 # exclude last axis
    dest = similar(cfgs)
    F .+= Nd .* 4cfgs
    for μ in 1:Nd
        circshift!(dest, cfgs, -Flux.onehot(μ, 1:sz))
        F .-= 2 .* dest
    end
    for μ in 1:Nd
        circshift!(dest, cfgs, Flux.onehot(μ, 1:sz))
        F .-= 2 .* dest
    end
    return F
end

function ∂potential(
    action::ScalarPhi4Action{T},
    cfgs::AbstractArray{T,N},
) where {T,N}
    @. 2action.m² * cfgs + 4 * action.λ * cfgs^3
end

function ∂kinetic(
    ::ScalarPhi4Action{T},
    cfgs::AbstractArray{T,N},
) where {T,N}
    ∂kinetic(cfgs)
end

function ChainRulesCore.rrule(
    a::ScalarPhi4Action{T},
    x::AbstractArray{T,N},
) where {T,N}
    y = a(x)
    function pullback(ȳ)
        sz = ndims(x)
        B = size(x, sz)
        Nd = sz - 1
        ∂a∂x = ∂potential(a, x) .+ ∂kinetic(a, x)
        # do multiplication for each batch
        x̄ = ∂a∂x .* reshape(ȳ, ones(Int, Nd)..., B)
        return NoTangent(), x̄
    end
    return y, pullback
end