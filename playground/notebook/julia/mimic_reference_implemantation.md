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
struct SimpleCouplingLayer
    scaling
    function SimpleCouplingLayer()
        net = Chain(
            Dense(rand(Normal(1, 2), 8, 1), -ones(8), relu),
            Dense(rand(Normal(1, 2), 8, 8), -ones(8), relu),
            Dense(rand(Normal(1, 2), 1, 8), -ones(1), tanh),
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
batch_szie = 1024
x1 = rand(Uniform(-1, 1), batch_size)
x2 = rand(Uniform(-1, 1), batch_size)

fx1, fx2, logJ = forward(coupling_layer, x1, x2)

rev_x1, rev_x2, rev_logJ = reverse(coupling_layer, fx1, fx2)

fig , ax = plt.subplots(1, 3, figsize=(10,4))

ax[1].scatter(x1, x2)
ax[2].scatter(fx1, fx2)
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
