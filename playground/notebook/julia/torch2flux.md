---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.4
  kernelspec:
    display_name: Julia 1.6.2
    language: julia
    name: julia-1.6
---

```julia
using Flux
using PyCall
```

```julia
torch = pyimport("torch")
```

```julia
function mycircular(Y)
    # calc Z_bottom
    Y_b_c = Y[begin:begin,:,:,:]
    Y_b_r = Y[begin:begin,end:end,:,:]
    Y_b_l = Y[begin:begin,begin:begin,:,:]
    Z_bottom = cat(Y_b_r, Y_b_c, Y_b_l, dims=2) # calc pad under
    
    # calc Z_top
    Y_e_c = Y[end:end,:,:,:]
    Y_e_r = Y[end:end,end:end,:,:]
    Y_e_l = Y[end:end,begin:begin,:,:]
    Z_top = cat(Y_e_r, Y_e_c, Y_e_l, dims=2)
    
    # calc Z_main
    Y_main_l = Y[:,begin:begin,:,:]
    Y_main_r = Y[:,end:end,:,:]
    Z_main = cat(Y_main_r, Y, Y_main_l, dims=2)
    cat(Z_top, Z_main, Z_bottom, dims=1)
end
```

```julia
using Flux
using PyCall
torch = pyimport("torch")
```

```julia
function torch2conv(lay, σ=Flux.identity)
    W = reversedims(lay.weight.data.numpy())
    W = W[end:-1:1, end:-1:1, :, :]
    if isnothing(lay.bias)
        b = zeros(eltype(W), size(W, 4))
    else
        b = reversedims(lay.bias.data.numpy())
    end
    pad = lay.padding
    stride = lay.stride
    if lay.padding_mode == "circular"
        pad = 0
        return Chain(
            mycircular,
            Conv(W, b, σ; pad, stride)
        )
    else
        return Chain(Conv(W, b, σ; pad, stride))
    end
end
```

```julia
function reversedims(a::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    permutedims(a, N:-1:1)
end
```

```julia
torch.manual_seed(12345)
torch_conv = torch.nn.Conv2d(1, 2, (3,3), padding=1, padding_mode="circular")
```

```julia
flux_conv = torch2conv(torch_conv) |> f32
```

```julia
@show torch_conv.weight
@show torch_conv.bias
```

```julia
flux_conv[end].weight
```

```julia
flux_conv[end].bias
```

```julia
py"""
import numpy as np
x = np.ones((1,1,5,5)).astype(np.float32)
"""
```

```julia
jl2torch(x::AbstractArray) = torch.from_numpy(x |> reversedims)
torch2jl(x::PyObject) = x.data.numpy() |> reversedims
```

```julia
n = 5
x = reshape(collect(1:n^2),n, n, 1, 1) |> f32
ret_flux = flux_conv(x)
```

```julia
ret_torch = torch_conv(jl2torch(x)) |> torch2jl
```

```julia
@assert isapprox(ret_flux, ret_torch)
```
