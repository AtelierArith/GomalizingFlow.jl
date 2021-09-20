---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.12.0
  kernelspec:
    display_name: Julia 1.6.2
    language: julia
    name: julia-1.6
---

# `U(1)`

```julia
using PyCall
using Flux
using EllipsisNotation
```

```julia
L = 8
lattice_shape = (2, L, L)
link_shape = (2, L, L)
```

```julia
py"""
import numpy as np
import torch


class U1GaugeAction:
    def __init__(self, beta):
        self.beta = beta
    def __call__(self, cfgs):
        Nd = cfgs.shape[1]
        action_density = 0
        for mu in range(Nd):
            for nu in range(mu+1,Nd):
                theta = compute_u1_plaq(cfgs, mu, nu)
                action_density += torch.cos(theta)
        return -self.beta * torch.sum(action_density, dim=tuple(range(1,Nd+1)))

def compute_u1_plaq(links, mu, nu):
    return (links[:,mu] + torch.roll(links[:,nu], -1, mu+1)
            - torch.roll(links[:,mu], -1, nu+1) - links[:,nu])

def sample_cfgs(L=$L):
    lattice_shape = (L, L)
    link_shape = (2, L, L)
    rng = np.random.default_rng(12345)
    u1_ex1 = 2*np.pi*rng.random(size=link_shape).astype(np.float64)
    u1_ex2 = 2*np.pi*rng.random(size=link_shape).astype(np.float64)
    cfgs = torch.from_numpy(np.stack((u1_ex1, u1_ex2), axis=0))
    return cfgs
"""
```

```julia
reversedims(inp::AbstractArray{<:Any, N}) where {N} = permutedims(inp, N:-1:1)
```

```julia
pycfgs = py"sample_cfgs"(L)
cfgs = pycfgs.numpy()
@show cfgs |> size # (W, H, Nd, B)
beta = 1
u1_action = py"U1GaugeAction"(beta)
u1_action(pycfgs).numpy()
```

$$
\theta_{\mu\nu} = \theta_\mu(\vec{n}) + \theta_\nu(\vec{n}+\hat{\mu}) - \theta_\mu(\vec{n}+\hat{\nu}) - \theta_\nu(\vec{n})
$$

```julia
"""
cfgs |> size == (Batch, Nd, H=L, W=L)
"""
function compute_u1_plaq(cfgs, μ, ν)
    θ_μ = @view cfgs[:, μ, ..]
    θ_ν = @view cfgs[:, ν, ..]
    θ_ν_shift = circshift(θ_ν, -Flux.onehot(1+μ, 1:ndims(θ_ν)))
    θ_μ_shift = circshift(θ_μ, -Flux.onehot(1+ν, 1:ndims(θ_μ)))
    θ_μν = θ_μ + θ_ν_shift - θ_μ_shift - θ_ν
    return θ_μν
end
```

```julia
struct U1GaugeAction{T<:Real}
    beta::T
end

"""
cfgs |> size == (B, Nd, H=L, W=L)
"""
function (u::U1GaugeAction)(cfgs)
    Nd = size(cfgs, 2)
    action_density=zeros(2,8,8)
    for μ in 1:Nd
        for ν in (μ+1):Nd
            θ = compute_u1_plaq(cfgs, μ, ν)
            action_density += cos.(θ)
        end
    end
    return -u.beta * sum(action_density, dims=(2,3))
end
```

```julia
pycfgs = py"sample_cfgs"(L)
cfgs = pycfgs.numpy()
@show cfgs |> size # (B, Nd, H, W)
β = 1.
u1_action = py"U1GaugeAction"(β)
@assert U1GaugeAction(β)(cfgs) ≈ u1_action(pycfgs).numpy()
```

```julia

```
