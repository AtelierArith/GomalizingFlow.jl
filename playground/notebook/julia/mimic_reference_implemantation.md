---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.3
  kernelspec:
    display_name: Julia 1.6.1
    language: julia
    name: julia-1.6
---

```julia
using PyCall
using PyPlot
using Random
using Distributions
using Flux
```

```julia
batch_size = 2^14
u1 = rand(batch_size)
u2 = rand(batch_size)
z1 = @. sqrt(-2log(u1)) * cos(2π*u2)
z2 = @. sqrt(-2log(u1)) * sin(2π*u2)
fig, ax = plt.subplots()
ax.hist2d(
    z1,
    z2,
    bins=30,
    range=[
        [-3.0, 3.0],
        [-3.0, 3.0],
    ],
)
ax.set_aspect("equal")
```

```julia
py"""
import numpy as np
def init_simple_coupling_layer_params(seed=12345):
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=(8, 1)).astype(np.float32)
    w2 = rng.normal(size=(8, 8)).astype(np.float32)
    w3 = rng.normal(size=(1, 8)).astype(np.float32)
    b1 = -np.ones(8).astype(np.float32)
    b2 = -np.ones(8).astype(np.float32)
    b3 = -np.ones(1).astype(np.float32)
    return w1, w2, w3, b1, b2, b3
"""

struct SimpleCouplingLayer
    scaling
    function SimpleCouplingLayer()
        w1, w2, w3, b1, b2, b3 = py"init_simple_coupling_layer_params"()
        net = Chain(
            Dense(w1, b1, relu),
            Dense(w2, b2, relu),
            Dense(w3, b3, tanh),
        )
        new(net)
    end
end

function forward(coupling_layer::SimpleCouplingLayer, x1, x2)
    s = coupling_layer.scaling(Flux.unsqueeze(x2, 1))
    s = reshape(s, size(s)[end])
    fx1 = @. exp(s) * x1
    fx2 = x2
    logJ = s
    return fx1, fx2, logJ
end

function reverse(coupling_layer::SimpleCouplingLayer, fx1, fx2)
    s = coupling_layer.scaling(Flux.unsqueeze(fx2, 1))
    s = reshape(s, size(s)[end])
    x1 = @. exp(-s) * fx1
    x2 = fx2
    logJ = -s
    return x1, x2, logJ
end


coupling_layer = SimpleCouplingLayer()
```

```julia
np = pyimport("numpy")
```

```julia
py"""
def gen_sample(batch_size, seed=12345):
    rng = np.random.default_rng(seed)
    x1 = 2 * rng.random(size=batch_size).astype(np.float32) - 1
    x2 = 2 * rng.random(size=batch_size).astype(np.float32) - 1
    return x1, x2
"""
```

```julia
batch_size = 1024
rng = np.random.default_rng(12345)
x1 = 2rng.random(size=batch_size, dtype=np.float32) .- 1
x2 = 2rng.random(size=batch_size, dtype=np.float32) .- 1
x1, x2 = py"gen_sample"(batch_size)
gx1, gx2, logJ = forward(coupling_layer, x1, x2)
@assert gx1[1:3] ≈ [-0.25462535, -0.17111894, 0.27769268]
@assert gx2[1:3] ≈ [-0.26831156,  0.02142358, -0.92905575]

rev_x1, rev_x2, rev_logJ = reverse(coupling_layer, gx1, gx2)

fig, ax = plt.subplots(1, 3, figsize=(10, 3), sharex=true, sharey=true)
for a in ax
    a.set_xlim(-1.1, 1.1)
    a.set_ylim(-1.1, 1.1)
end
ax[1].scatter(x1, x2)
ax[2].scatter(gx1, gx2)
ax[3].scatter(rev_x1, rev_x2)
fig.tight_layout()
```

```julia
function apply_flow_to_prior(r, coupling_layers; batch_size)
    z = rand(r, batch_size, 2)
    x1 = z[:, 1]
    x2 = z[:, 2]
    logq = logpdf(r, z)
    for lay in coupling_layers
        x1, x2, logJ = forward(lay, x1, x2)
        logq = logq - logJ
    end
    return x1, x2, logq
end
```

# ScalarPhi4Action

```julia
torch = pyimport("torch")
py"""
import torch
class SimpleNormal:
    def __init__(self, loc, var):
        self.distribution = torch.distributions.normal.Normal(
            loc,
            var,
        )
        self.shape = loc.shape

    def log_prob(self, x):
        logp = self.distribution.log_prob(x)
        return torch.sum(logp, dim=tuple(range(1, logp.ndim)))

    def sample_n(self, batch_size):
        x = self.distribution.sample((batch_size,))
        return x
"""
```

```julia

py"""
import numpy as np
def create_cfgs(L):
    rng = np.random.default_rng(2021)
    lattice_shape=(L, L)
    phi_ex1 = rng.normal(size=(lattice_shape)).astype(np.float32)
    phi_ex2 = rng.normal(size=(lattice_shape)).astype(np.float32)
    cfgs = np.stack((phi_ex1, phi_ex2), axis=0)
    return cfgs
"""

L = 8
m² = -4.0
λ = 8.0
const lattice_shape = (L, L)
```

```julia
struct ScalarPhi4Action
    m²::Float32
    λ::Float32
end 

function calc_action(action::ScalarPhi4Action, cfgs)
    action_density = @. action.m² * cfgs ^ 2 + action.λ * cfgs ^ 4
    Nd = lattice_shape |> length
    term1 = sum(2cfgs .^ 2 for μ in 2:Nd+1)
    term2 = sum(cfgs .* circshift(cfgs, Flux.onehot(μ, 1:(Nd+1))) for μ in 2:(Nd+1))
    term3 = sum(cfgs .* circshift(cfgs, -Flux.onehot(μ, 1:(Nd+1))) for μ in 2:(Nd+1))
    dropdims(
        sum(action_density .+ term1 .- term2 .- term3, dims=2:Nd+1),
        dims=Tuple(2:(Nd+1))
    )
end

function (action::ScalarPhi4Action)(cfgs)
    calc_action(action, cfgs)
    #=
    action_density = @. action.m² * cfgs ^ 2 + action.λ * cfgs ^ 4
    Nd = lattice_shape |> length
    for μ ∈ 2:Nd+1
        action_density .+= 2cfgs .^ 2

        shifts_plus = zeros(Nd+1)
        shifts_plus[μ] = 1 # \vec{n} + \hat{\mu}
        action_density .-= cfgs .* circshift(cfgs, shifts_plus)

        shifts_minus = zeros(Nd+1)
        shifts_minus[μ] = -1 # \vec{n} - \hat{\mu}
        action_density .-= cfgs .* circshift(cfgs, shifts_minus)
    end
    return dropdims(sum(action_density, dims=2:Nd+1), dims=Tuple(2:(Nd+1)))
    =#
end

cfgs = py"create_cfgs"(L)
@assert ScalarPhi4Action(1, 1)(cfgs) ≈ [499.7602, 498.5477]
phi4_action = ScalarPhi4Action(m², λ)
@assert phi4_action(cfgs) ≈ [1598.679, 1545.5698]
```

```julia
torch.manual_seed(12345)

batch_size = 1024

prior = py"SimpleNormal"(
    torch.from_numpy(zeros(Float32, lattice_shape)), 
    torch.from_numpy(ones(Float32, lattice_shape)),
)

z = prior.sample_n(1).detach().numpy()
torch.manual_seed(1)
z = prior.sample_n(1024).detach().numpy()
@assert z[1,1,1] ≈ -1.5255959
@assert z[2, 5, 6] ≈ -0.81384623
@assert z[11, 4, 3] ≈ -0.3155666
S_eff = -prior.log_prob(torch.from_numpy(z)).detach().numpy()
@assert S_eff[1:3] ≈ [91.6709, 88.7396, 97.8660]
@assert S_eff[end-2:end] ≈ [92.81951, 91.93714, 80.611916]
S = phi4_action(z)
@assert S[1:3] ≈ [2103.6904, 1269.7277, 1697.3539]
@assert S[end-2:end] ≈ [1231.3823, 1500.0603, 600.58417]

fit_b = mean(S) - mean(S_eff)

fig, ax = plt.subplots(1,1, dpi=125, figsize=(4,4))
ax.hist2d(S_eff |> vec, S |> vec, bins=20, range=[[-800, 800], [200,1800]])
xs = range(-800, 800, length=4)
ax.plot(xs, xs .+ fit_b, ":", color="w", label="slope 1 fit")
ax.set_xlabel("\$S_{\\mathrm{eff}} \\equiv -\\log~r(z)\$")
ax.set_ylabel("S(z)")
ax.set_aspect("equal")
plt.legend(prop=Dict("size" => 6))
plt.show()
```
