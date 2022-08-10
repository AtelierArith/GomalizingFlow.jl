mutable struct ScalarPhi4Action{T<:AbstractFloat}
    m²::T
    λ::T
end

function calc_action(
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

function (action::ScalarPhi4Action{T})(
    cfgs::AbstractArray{T,N},
) where {T<:AbstractFloat,N}
    calc_action(action, cfgs)
end
