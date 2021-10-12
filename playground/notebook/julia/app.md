---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Julia 1.6.3
    language: julia
    name: julia-1.6
---

# Load trained model

```julia
using BSON
using PyPlot
using IterTools
using Flux: cpu

using LFT
```

```julia
Nd = 2
model_path = joinpath(dirname(dirname(pathof(LFT))), "result" , "example2d", "trained_model.bson")
config_path = joinpath(dirname(dirname(pathof(LFT))), "result" , "example2d", "config.toml")
```

```julia
BSON.@load model_path trained_model
```

```julia
hp = LFT.load_hyperparams(config_path);
```

```julia
ensamble_size = 8192
batchsize=1024
device = cpu
model = trained_model
prior = hp.tp.prior
phi4_action = LFT.ScalarPhi4Action(hp.pp.m², hp.pp.λ)
L = hp.pp.L
lattice_shape = hp.pp.lattice_shape
```

```julia
z = rand(prior, lattice_shape..., batchsize)
logq_device = sum(logpdf(prior, z), dims=(1:ndims(z) - 1)) |> device
z_device = z |> device
x_device, logq_ = model((z_device, logq_device))
x = cpu(x_device)
S_eff = -logq_ |> cpu
S = phi4_action(x)
fit_b = mean(S) - mean(S_eff)
@show fit_b
print("slope 1 linear regression S = -logr + $fit_b")
fig, ax = plt.subplots(1,1, dpi=125, figsize=(4,4))
ax.hist2d(vec(S_eff), vec(S), bins=20)

xs = range(-800, stop=800, length=4)
ax.plot(xs, xs .+ fit_b, ":", color=:w, label="slope 1 fit")
ax.set_xlabel(L"S_{\mathrm{eff}} \equiv -\log ~ r(z)")
ax.set_ylabel(L"S(z)")
ax.set_aspect(:equal)
plt.legend(prop=Dict("size"=> 6))
plt.show()
```

```julia
history = LFT.make_mcmc_ensamble(model, prior, phi4_action, lattice_shape, batchsize=64, nsamples=ensamble_size);
@show mean(history[:accepted]) |> mean
```

```julia
function green(cfgs, offsetX)
    Gc = zero(Float32)
    for posY in IterTools.product((1:l for l in lattice_shape)...)
        phi_y = cfgs[posY..., :]
        shifts = (broadcast(-, offsetX)..., 0)
        phi_y_x = circshift(cfgs, shifts)[posY..., :]
        mean_phi_y = mean(phi_y)
        mean_phi_y_x = mean(phi_y_x)
        Gc += mean(phi_y .* phi_y_x) - mean_phi_y * mean_phi_y_x
    end
    Gc /= prod(lattice_shape)
    return Gc
end
```

```julia
function mfGc(cfgs, t)
    space_shape = size(cfgs)[end-1]
    ret = 0
    for s in IterTools.product((1:l for l in space_shape)...)
        ret += green(cfgs, (s..., t))
    end
    ret /= prod(space_shape)
    return ret
end
```

```julia
cfgs = cat(history[:x][512:2000]..., dims=length(lattice_shape)+1);
```

```julia
plt.plot(0:hp.pp.L, [mfGc(cfgs, t) for t in 0:hp.pp.L])
plt.yscale("log")
```
