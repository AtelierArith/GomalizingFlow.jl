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
using Distributions
using PyPlot
using Flux
```

```julia
L = 8
lattice_shape = (L, L)
```

```julia
function make_checker_mask(shape, parity)
    checker = ones(Int, shape) .- parity
    checker[begin:2:end, begin:2:end] .= parity
    checker[(begin+1):2:end, (begin+1):2:end] .= parity
    return checker
end
```

```julia
make_checker_mask(lattice_shape, 0)
```

```julia
function make_conv_net(
        ; in_channels,out_channels, hidden_sizes, kernel_size, use_final_tanh)
    sizes = vcat(in_channels, hidden_sizes, out_channels)
    @assert isodd(kernel_size)
    pad = kernel_size ÷ 2
    net = []
    for i in 1:(length(sizes)-2)
        push!(
            net, 
            Conv(
                (kernel_size, kernel_size), 
                sizes[i] => sizes[i+1], 
                Flux.leakyrelu; 
                pad, stride=1, bias=true,
            )
        )
    end
    # define last layer
    push!(
        net, 
        Conv(
            (kernel_size, kernel_size), 
            sizes[end-1] => sizes[end],
            ifelse(use_final_tanh, identity, tanh)
            ; pad, stride=1, bias=true,
        )
    )
    return Chain(net...)
end
```

```julia
struct AffineCoupling
    net
    mask
end

Flux.@functor AffineCoupling
```

```julia
Flux.unsqueeze(rand(10,10,2), 3)
```

```julia
v=rand(10)
ndims
```

```julia
#=
x_torch = (B, nC, inH, inW)
x_flux = (inW, inH, inC, inB)
=#

function (model::AffineCoupling)(x)
    x_frozen = model.mask .* x
    x_active = (1 .- model.mask) .* x
    # (inW, inH, inB) -> (inW, inH, 1, inB) # by Flux.unsqueeze(*, 3)
    net_out = model.net(Flux.unsqueeze(x_frozen, 3))
    s = net_out[:, :, 1, :] # extract feature from 1st channel
    t = net_out[:, :, 2, :] # extract feature from 2nd channel
    fx = @. (1 - model.mask) * t + x_active * exp(s) + x_frozen
    logJ = sum((1 .- model.mask) .* s, dims=1:(ndims(s)-1))
    return fx, logJ
end

forward(model::AffineCoupling, x) = model(x)

function reverse(model::AffineCoupling, fx)
    fx_frozen = model.mask .* fx
    fx_active = (1 .- model.mask) .* fx
    net_out = model(fx_frozen)
    return net_out
end
```

```julia
# example

m = AffineCoupling(Conv((3,3), 1=>2, pad=1), make_checker_mask(lattice_shape, 0))
x = rand(Float32, lattice_shape...,10);
```

```julia
function make_phi4_affine_layers(;lattice_shape, n_layers, hidden_sizes, kernel_size)
    layers = []
    for i in 1:n_layers
        parity = mod(i - 1, 2)
        net = make_conv_net(
            in_channels=1, 
            out_channels=2, 
            hidden_sizes=hidden_sizes,
            kernel_size=kernel_size, 
            use_final_tanh=true
        )
        mask = make_checker_mask(lattice_shape, parity)
        coupling = AffineCoupling(net, mask)
        push!(layers, coupling)
    end
    Chain(layers...)
end
```

```julia
prior = Normal{Float32}(0.f0, 1.f0)
```

```julia
n_layers = 16
hidden_sizes = [8,8]
kernel_size = 3
layers = make_phi4_affine_layers(
    lattice_shape=lattice_shape, 
    n_layers=n_layers,
    hidden_sizes=hidden_sizes, 
    kernel_size=kernel_size,
)
model = Dict("layers" => layers, "prior" => prior)
```

```julia
function apply_flow_to_prior(prior, coupling_layers; batch_size)
    x = rand(prior, lattice_shape..., batch_size)
    logq = sum(logpdf(prior, x), dims=(1:ndims(x)-1))
    for layer in coupling_layers
        x, logJ = layer(x)
        logq .-= logJ
    end
    return x, logq
end
```

```julia
apply_flow_to_prior(prior, layers; batch_size=10);
```

```julia
calc_dkl(logp, logq) = mean(logq - logp)

function compute_ess(logp, logq)
    logw = logp - logq
    log_ess = 2*logsumexp(logw, dim=0) - logsumexp(2*logw, dim=0)
    ess_per_cfg = exp(log_ess) / len(logw)
    return ess_per_cf
end
```

```julia
d = Dict(1=>-1,2=>-2)
```

```julia
for (k, v) in d
end
```

```julia
function print_metrics(history, avg_last_N_epochs)
    for (key, val) in history
        avgd = mean(val[-avg_last_N_epochs:end])
        print("$key, $avgd g")
    end
end
```

```julia
lr = 0.001f0
optimizer = ADAM(lr)
```

```julia
function loss_fn(x, y)
    lx, logq = apply_flow_to_prior(prior, layers, batch_size=batch_size)
    logp = -action(x)
    loss = calc_dkl(logp, logq)
end
```

```julia
Flux.train!(loss_fn, parameters, data, optimizer)
```

```julia
function train_step(model, action, loss_fn, optimizer, metrics)
    layers, prior = model["layers"], model["prior"]
    optimizer.zero_grad()
    x, logq = apply_flow_to_prior(prior, layers, batch_size=batch_size)
    logp = -action(x)
    loss = calc_dkl(logp, logq)
    loss.backward()
    optimizer.step()
    #metrics["loss"].append(grab(loss))
    #metrics["logp"].append(grab(logp))
    #metrics["logq"].append(grab(logq))
end
```
