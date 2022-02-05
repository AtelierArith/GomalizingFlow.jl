struct ScalarPhi4Action
    m²::Float32
    λ::Float32
end

function calc_action(action::ScalarPhi4Action, cfgs)
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

function (action::ScalarPhi4Action)(cfgs)
    calc_action(action, cfgs)
end
