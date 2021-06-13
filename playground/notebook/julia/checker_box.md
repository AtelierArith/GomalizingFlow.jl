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
                pad, stride=1
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
            ; pad, stride=1
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
    net_out = model(Flux.unsqueeze(x_frozen, 3))
    s = net_out[:,:,1,:] # extract feature from 1st channel
    t = net_out[:,:,2,:] # extract feature from 2nd channel
    fx = @. (1 - modelmask) * t + x_active * exp(s) + x_frozen
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
prior = Normal(0, 1)
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
