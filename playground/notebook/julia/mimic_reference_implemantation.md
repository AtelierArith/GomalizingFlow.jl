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
