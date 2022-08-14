mutable struct ReferenceScalarPhi4Action{T<:AbstractFloat}
    m²::T
    λ::T
end

function calc_action(
    action::ReferenceScalarPhi4Action{T},
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

@testset "ScalarPhi4Action" begin
    T = Float32
    L = 16
    Nd = 3
    B = 64
    cfgs = Flux.batch(fill(1, Nd)..., B)
    ref_action = ReferenceScalarPhi4Action{T}(T(0.3), T(2))
    action = GomalizingFlow.ScalarPhi4Action{T}(T(0.3), T(2))

    @test action(cfgs) ≈ ref_action(cfgs)
    J_zygote, = Zygote.jacobian(ref_action, cfgs)
    J, = Zygote.jacobian(action, cfgs)
    @test J_zygote |> size == J |> size
    @test J_zygote ≈ J

    ref_jt = @benchmark Zygote.jacobian(ref_action, $cfgs)
    jt = @benchmark Zygote.jacobian(action, $cfgs)
    @test mean(jt.times) < mean(ref_jt.times)

    ∇_zygote, = Zygote.gradient(sum ∘ action, cfgs)
    ∇, = Zygote.gradient(sum ∘ differentiable_action, cfgs)
    @test ∇_zygote ≈ ∇

    ref_gt = @benchmark Zygote.gradient(sum ∘ action, cfgs)
    gt = @benchmark Zygote.gradient(sum ∘ differentiable_action, cfgs)
    @test mean(gt.times) < mean(ref_gt.times)
end