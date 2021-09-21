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

# $U(1)$

```julia
using PyCall
using Flux
using Distributions
using EllipsisNotation
```

# Setup constants

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
py_u1_action = py"U1GaugeAction"(beta)
py_u1_action(pycfgs).numpy()
```

# $U(1)$ action


$$
\theta_{\mu\nu} = \theta_\mu(\vec{n}) + \theta_\nu(\vec{n}+\hat{\mu}) - \theta_\mu(\vec{n}+\hat{\nu}) - \theta_\nu(\vec{n})
$$

```julia
"""
links |> size == (Batch, Nd, H=L, W=L)
"""
function compute_u1_plaq(links, μ, ν)
    θ_μ = @view links[:, μ, ..]
    θ_ν = @view links[:, ν, ..]
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

pycfgs = py"sample_cfgs"(L)
cfgs = pycfgs.numpy()
@show cfgs |> size # (B, Nd, H, W)
@assert compute_u1_plaq(cfgs, 1, 2) ≈ py"compute_u1_plaq"(pycfgs, 0, 1).numpy()

β = 1.
u1_action = py"U1GaugeAction"(β)
@assert U1GaugeAction(β)(cfgs) ≈ py_u1_action(pycfgs).numpy()
```

## gauge transform

```julia
"""
links |> size == (B, Nd, H, W)
α |> size == (B, H, W)
"""
function gauge_transform!(links, α)
    Nd = size(links, 2)
    for μ in 1:Nd
        links[:, μ, ..] .= α .+ links[:, μ, ..] .- circshift(α, -Flux.onehot(1+μ, 1:ndims(α)))
    end
    return links
end
```

```julia
function random_gauge_transform(cfgs)
    Nconf = size(cfgs, 1)
    VolShape = size(cfgs)[3:end]
    return gauge_transform!(cfgs, 2π*rand(Nconf, VolShape...))
end
```

```julia
β = 1.
u1_action = U1GaugeAction(β)
cfgs_transformed = random_gauge_transform(copy(cfgs))
@assert ~(cfgs_transformed ≈ cfgs)
@assert u1_action(cfgs) ≈ u1_action(cfgs_transformed)
```

# topologial charge on lattice

$$
Q = \frac{1}{2\pi} \sum_{\vec{n}} \mathrm{args}(P_{01}(\vec{n})) \quad Q \in \mathbb{Z}
$$

```julia
function topo_charge(cfgs)
    μ = 1
    ν = 2
    P₀₁ = compute_u1_plaq(cfgs, μ, ν)
    sum(P₀₁, dims=2:ndims(P₀₁)) / 2π
end
```

```julia
topo_charge(2π*rand(2, 2, 8, 8)) # verysmall value
```

```julia
using Distributions
```

```julia
prior = Uniform(0, 2π)
batch = 17
z = rand(prior, (batch, link_shape...));
```

```julia
function log_prob(z)
    logp = logpdf(prior, z)
    sum(logp, dims=2:ndims(logp))
end
```

```julia
log_prob(z)
```

# Gauge quivariant coupling layers

```julia

```
