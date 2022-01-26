---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
  kernelspec:
    display_name: julia 1.6.5
    language: julia
    name: julia-1.6
---

```julia
using LinearAlgebra

using BSON
using LFT
using Flux
using Distributions
using ImageTransformations
using Parameters
using Plots
```

```julia
result_dir = joinpath(pkgdir(LFT), "result")
BSON.@load joinpath(result_dir, "example2d", "trained_model_best_ess.bson") trained_model_best_ess
```

```julia
hp = LFT.load_hyperparams(joinpath(result_dir, "example2d", "config.toml"));
```

```julia
device = hp.dp.device

prior = eval(Meta.parse(hp.tp.prior))

@unpack m², λ, lattice_shape = hp.pp
action = LFT.ScalarPhi4Action(m², λ)
batchsize = 1
```

```julia
model = trained_model_best_ess |> device;
```

```julia
z = rand(prior, lattice_shape..., batchsize)
```

```julia
function infer(model, z_, idx=length(model))
    z = Flux.unsqueeze(z_, ndims(z_) + 1)
    z_device = z |> device
    logq_device = sum(logpdf.(prior, z), dims = (1:ndims(z)-1)) |> device

    x_, logq_ = model[1:idx]((z_device, logq_device))
    dropdims(x_, dims=ndims(x_))
end
```

```julia
p1 = heatmap(rand(10, 10))
p2 = heatmap(rand(10, 10))
plot(p1, p2, layout=(1, 2), size=(1000, 400))
```

```julia
d = MvNormal(Float32[4, 4], Float32[1/10 0;0 1/10])
f(x, y) = pdf(d, [x, y])
x = 0:0.1f0:8
y = 0:0.1f0:8
im = f.(x', y)
p1 = heatmap(im, clim=(0, 0.2))
z = imresize(im, lattice_shape)
p2 = heatmap(z, clim=(0, 0.2))
plot(p1, p2, layout=(1, 2), size=(800,300))
```

```julia
anim = @animate for idx in 1:length(model)
    x = infer(model, z, idx)
    if idx != length(model)
        activation = tanh
    else
        activation = identity
    end

    heatmap(activation.(x), clim=(-1, 1), title="idx=$idx")
end
gif(anim, fps=1)
```