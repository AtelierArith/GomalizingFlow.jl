```julia
using BenchmarkTools

using GomalizingFlow
using GomalizingFlow: PhysicalParams, HyperParams, ScalarPhi4Action
using Flux
using Flux.Zygote
using Flux: unsqueeze
```

```julia
function calc_kinetic_withloop(cfgs::AbstractArray{T, N},pp::PhysicalParams) where {T, N}
    Lx, Ly, Lz = pp.lattice_shape
    ϕ = cfgs
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
function calc_kinetic(
    action::ScalarPhi4Action{T},
    cfgs::AbstractArray{T,N},
) where {T<:AbstractFloat,N}
    sz = length(size(cfgs))
    Nd = sz - 1 # exclude last axis
    B = size(cfgs, sz) # batch size
    k1 = Nd * 2cfgs .^ 2
    k2 = sum(cfgs .* circshift(cfgs, -Flux.onehot(μ, 1:sz)) for μ in 1:Nd)
    k3 = sum(cfgs .* circshift(cfgs, Flux.onehot(μ, 1:sz)) for μ in 1:Nd)
    K = k1 .- k2 .- k3
    reshape(
        sum(K, dims=1:Nd),
        B
    )
end
```

```
o11 = 11 * 51 + 11 * 21
o12 = 12 * 52 + 12 * 22
o13 = 13 * 53 + 13 * 23
o14 = 14 * 54 + 14 * 24
o15 = 15 * 55 + 15 * 25

o21 = 21 * 11 + 21 * 31
o22 = 22 * 12 + 22 * 32
o23 = 23 * 13 + 23 * 33
o24 = 24 * 14 + 24 * 34
o25 = 25 * 15 + 25 * 35

o31 = 31 * 21 + 31 * 41
o32 = 32 * 22 + 32 * 42
o33 = 33 * 23 + 33 * 43
o34 = 34 * 24 + 34 * 44
o35 = 35 * 25 + 35 * 45
```

```
x x x x x
x x x x x
x x x x x
x x x x x
x x x x x
```

```julia
function ∂kinetic(cfgs::AbstractArray{T, N}, pp::PhysicalParams) where {T, N}
    Lx, Ly, Lz = pp.lattice_shape
    ϕ = cfgs
    F = zero(ϕ)
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
                F[ix,iy,iz] += (2ϕ[ix,iy,iz] - 2ϕ[ip,iy,iz]-2ϕ[im,iy,iz]) + 2ϕ[ix,iy,iz]
                # - - - - - - - - - - - - - - - - - - - -
                ip=iy+1
                im=iy-1
                if ip>Ly
                    ip-=Ly
                end
                if im<1
                    im+=Ly
                end
                F[ix,iy,iz] += (2ϕ[ix,iy,iz] - 2ϕ[ix,ip,iz]-2ϕ[ix,im,iz]) + 2ϕ[ix,iy,iz]
                # - - - - - - - - - - - - - - - - - - - -
                ip=iz+1
                im=iz-1
                if ip>Lz
                    ip-=Lz
                end
                if im<1
                    im+=Lz
                end
                F[ix,iy,iz] += (2ϕ[ix,iy,iz] - 2ϕ[ix,iy,ip]-2ϕ[ix,iy,im]) + 2ϕ[ix,iy,iz]
            end
        end
    end
    return F
end

# Slightly optimised than ∂kinetic
function ∂kinetic1(cfgs::AbstractArray{T, N}, pp::PhysicalParams) where {T, N}
    Lx, Ly, Lz = pp.lattice_shape
    ϕ = cfgs
    F = zero(ϕ)
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
                F[ix,iy,iz] += (4ϕ[ix,iy,iz] - 2ϕ[ip,iy,iz]-2ϕ[im,iy,iz])
                # - - - - - - - - - - - - - - - - - - - -
                ip=iy+1
                im=iy-1
                if ip>Ly
                    ip-=Ly
                end
                if im<1
                    im+=Ly
                end
                F[ix,iy,iz] += (4ϕ[ix,iy,iz] - 2ϕ[ix,ip,iz]-2ϕ[ix,im,iz])
                # - - - - - - - - - - - - - - - - - - - -
                ip=iz+1
                im=iz-1
                if ip>Lz
                    ip-=Lz
                end
                if im<1
                    im+=Lz
                end
                F[ix,iy,iz] += (4ϕ[ix,iy,iz] - 2ϕ[ix,iy,ip]-2ϕ[ix,iy,im])
            end
        end
    end
    return F
end

function ∂kinetic2(cfgs::AbstractArray{T, N}, pp::PhysicalParams) where {T, N}
    ϕ = cfgs
    F = zero(ϕ)
    sz = ndims(cfgs)
    F += sz * 4cfgs
    F .-= sum(2circshift(cfgs, -Flux.onehot(μ, 1:sz)) for μ in 1:sz)
    F .-= sum(2circshift(cfgs, +Flux.onehot(μ, 1:sz)) for μ in 1:sz)
    return F
end
```

```julia
function calc_potential(action::ScalarPhi4Action{T}, cfgs::AbstractArray{T, N}) where {T, N}
    sum(action.m² * cfgs .^ 2 + action.λ * cfgs .^ 4)
end

function calc_potential1(action::ScalarPhi4Action{T}, cfgs::AbstractArray{T, N}) where {T, N}
    cfgs² = cfgs .^2
    sum(action.m² * cfgs² + action.λ * cfgs² .^ 2)
end

function ∂potential(action::ScalarPhi4Action{T}, cfgs::AbstractArray{T, N}) where {T, N}
    @. 2action.m² * cfgs + 4*action.λ * cfgs ^ 3
end
```

```julia
T = Float32
L = 16
pp = PhysicalParams(L, 3, 1, 1)
action = GomalizingFlow.ScalarPhi4Action{Float32}(pp.m², pp.λ)
cfgs = rand(T, L, L, L) |> cpu;
@assert calc_potential(action, cfgs) ≈ sum(action.m² * cfgs .^2 + action.λ * cfgs .^ 4)
@assert calc_potential(action, cfgs) ≈ calc_potential1(action, cfgs)
```

```julia
@btime calc_potential(action, cfgs)
@btime calc_potential1(action, cfgs)
```

```julia
@btime gradient(Base.Fix1(calc_potential, action), $cfgs)[1]
@btime gradient(Base.Fix1(calc_potential1, action), $cfgs)[1]
@btime ∂potential(action, $cfgs);
```

# Kinetic

```julia
@assert sum(calc_kinetic(action, unsqueeze(cfgs, dims=ndims(cfgs)+1))) ≈ calc_kinetic_withloop(cfgs, pp)
```

```julia
∇1 = gradient(sum∘Base.Fix1(calc_kinetic, action),unsqueeze(cfgs, dims=ndims(cfgs)+1))[begin]
∇2 = gradient(Base.Fix2(calc_kinetic_withloop, pp),cfgs)[begin]
∇3 = Base.Fix2(∂kinetic, pp)(cfgs)
∇4 = Base.Fix2(∂kinetic1, pp)(cfgs)
∇5 = Base.Fix2(∂kinetic2, pp)(cfgs)
@assert ∇1 ≈ ∇2
@assert ∇2 ≈ ∇3
@assert ∇3 ≈ ∇4  ∇3 - ∇4
@assert ∇4 ≈ ∇5  ∇4 - ∇5
```

```julia
@btime gradient(sum∘Base.Fix1(calc_kinetic, action),unsqueeze($cfgs, dims=ndims($cfgs)+1))[begin]
@btime gradient(Base.Fix2(calc_kinetic_withloop, pp),$cfgs)[begin]
@btime Base.Fix2(∂kinetic, pp)($cfgs);
@btime Base.Fix2(∂kinetic2, pp)($cfgs);
```
