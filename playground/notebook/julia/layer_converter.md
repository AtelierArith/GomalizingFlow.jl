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
using Distributions
```

```julia
torch = pyimport("torch")
```

```julia
jl2torch(x::AbstractArray) = torch.from_numpy(x |> reversedims)
torch2jl(x::PyObject) = x.data.numpy() |> reversedims
```

```julia
py"""
import numpy as np
import torch
import torch.nn as nn

L = 8
lattice_shape = (L, L)
M2 = -4.
lam = 8.

n_layers = 16
hidden_sizes = [8,8]
kernel_size = 3
inC = 1
outC = 2
use_final_tanh = True

lr = 0.001

N_era = 25
N_epoch = 100
batch_size = 64
print_freq = N_epoch
plot_freq = 1
"""
```

```julia
py"""
def make_checker_mask(shape, parity):
    checker = torch.ones(shape, dtype=torch.uint8) - parity
    checker[::2, ::2] = parity
    checker[1::2, 1::2] = parity
    return checker

assert torch.all(make_checker_mask(lattice_shape, 0) == torch.from_numpy(np.array(
        [
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0]
        ]
        )
    )
)
"""
```

```julia
py"""
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
# Ours
torch.manual_seed(12345)

class MyAffineCoupling(torch.nn.Module):
    def __init__(self, net,*,mask_shape, parity):
        self.parity = parity
        super(MyAffineCoupling, self).__init__()
        self.mask = make_checker_mask(mask_shape, parity)
        self.mask_flipped = 1- self.mask

        self.net = net
    def forward(self, x): # (B, C, H, W)
        x_frozen = self.mask * x # \phi_2
        x_active = self.mask_flipped * x # \phi_1
        net_out = self.net(x_frozen.unsqueeze(1))
        s, t = net_out[:, 0], net_out[:, 1]
        # ((exp(s(phi_)))\phi_1 + t(\phi_2), \phi_2) を一つのデータとして
        fx = self.mask_flipped * t + x_active * torch.exp(s) + x_frozen
        axes = range(1, len(s.size()))
        logJ = torch.sum(self.mask_flipped * s, dim=tuple(axes))
        return fx, logJ
    
    def reverse(self, fx):
        fx_frozen = self.mask * fx # phi_2'
        fx_active = self.mask_flipped * fx # phi_1'
        net_out = self.net(fx_frozen.unsqueeze(1))
        s, t = net_out[:, 0], net_out[:, 1]
        x = (fx_active - self.mask_flipped * t) * torch.exp(-s) + fx_frozen
        axes = range(1, len(s.size()))
        logJ = torch.sum(self.mask_flipped * (-s), dim=tuple(axes))
        return x, logJ

import itertools

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

module_list = []
for i in range(n_layers):
    parity = i % 2
    sizes = [inC, *hidden_sizes, outC]
    padding = kernel_size // 2
    net = []
    for s, s_next in pairwise(sizes):
        net.append(
            torch.nn.Conv2d(s, s_next, kernel_size, padding=padding, padding_mode='circular')
        )
        net.append(torch.nn.LeakyReLU())
    if use_final_tanh:
        net[-1] = torch.nn.Tanh()
    net = torch.nn.Sequential(*net)
    coupling = MyAffineCoupling(net, mask_shape=lattice_shape, parity=parity)
    module_list.append(coupling)

prior = SimpleNormal(torch.zeros(lattice_shape), torch.ones(lattice_shape))
my_layers = torch.nn.ModuleList(module_list)

my_model = {"layers": my_layers, "prior": prior}
"""
```

```julia
py"""
def apply_affine_flow_to_prior(r, aff_coupling_layers, *, batch_size):
    z = r.sample_n(batch_size)
    logq = r.log_prob(z)
    x = z
    for lay in aff_coupling_layers:
        x, logJ = lay.forward(x)
        logq = logq - logJ
    return x, logq  # 点 x における `\log(q(x))` の値を計算

apply_affine_flow_to_prior(prior, my_layers, batch_size=batch_size)
"""
```

```julia
n_era = 25
epochs = 100
batchsize = 64

base_lr = 0.001f0
opt = ADAM(base_lr)
L = 8
lattice_shape = (L, L)
M2 = -4.
lam = 8.

n_layers = 16
hidden_sizes = [8, 8]
kernel_size = 3
inC = 1
outC = 2
use_final_tanh = true

prior = Normal{Float32}(0.f0, 1.f0)
```

```julia
function make_checker_mask(shape, parity)
    checker = ones(Int, shape) .- parity
    checker[begin:2:end, begin:2:end] .= parity
    checker[(begin+1):2:end, (begin+1):2:end] .= parity
    return checker
end

make_checker_mask(lattice_shape, 0)
```

```julia
struct AffineCoupling
    net
    mask
end

Flux.@functor AffineCoupling (net,)


function (model::AffineCoupling)(x_pair_loghidden)
    x = x_pair_loghidden[begin]
    loghidden = x_pair_loghidden[end]
    x_frozen = model.mask .* x
    x_active = (1 .- model.mask) .* x
    # (inW, inH, inB) -> (inW, inH, 1, inB) # by Flux.unsqueeze(*, 3)
    net_out = model.net(Flux.unsqueeze(x_frozen, 3))
    s = @view net_out[:, :, 1, :] # extract feature from 1st channel
    t = @view net_out[:, :, 2, :] # extract feature from 2nd channel
    fx = @. (1 - model.mask) * t + x_active * exp(s) + x_frozen
    logJ = sum((1 .- model.mask) .* s, dims=1:(ndims(s)-1))
    return (fx, loghidden .- logJ)
end

# alias
forward(model::AffineCoupling, x_pair_loghidden) = model(x_pair_loghidden)

function reverse(model::AffineCoupling, fx)
    fx_frozen = model.mask .* fx
    fx_active = (1 .- model.mask) .* fx
    net_out = model(fx_frozen)
    return net_out
end

function pairwise(iterable)
    b = deepcopy(iterable)
    popfirst!(b)
    a = iterable
    return zip(a, b)
end
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

function reversedims(a::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    permutedims(a, N:-1:1)
end

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

function apply_affine_flow_to_prior(prior, affine_coupling_layers; batchsize)
    x = rand(prior, lattice_shape..., batchsize)
    logq_ = sum(logpdf(prior, x), dims=(1:ndims(x)-1))
    xout, logq = affine_coupling_layers((x, logq_))
    return xout, logq
end
```

```julia
pycopy = pyimport("copy")
Base.copy(po::PyObject) = pycopy.copy(po)
Base.deepcopy(po::PyObject) = pycopy.deepcopy(po)
```

```julia
module_list = []
for (i, coupling) in enumerate(py"my_model"["layers"])
    parity = (i+1) % 2
    net = []
    for (lay, σ) in pairwise(coupling.net)
        if py"isinstance"(lay, py"torch.nn.Conv2d")
            if py"isinstance"(σ, py"torch.nn.LeakyReLU")
                push!(net, torch2conv(lay, leakyrelu))
            elseif py"isinstance"(σ, py"torch.nn.Tanh")
                push!(net, torch2conv(lay, tanh))
            else
                error("Expected σ is LeakyReLU or Tanh")
            end
        end
    end
    mask = make_checker_mask(lattice_shape, parity)
    push!(module_list, AffineCoupling(Chain(net...), mask))
end

affine_coupling_layers=Chain(module_list...) |> f32
```

```julia
x = rand(prior, lattice_shape..., batchsize)
logq_ = sum(logpdf(prior, x), dims=(1:ndims(x)-1))
xout, logq = affine_coupling_layers((x, logq_))
```

```julia
py"""
def applyflow(z):
    x = z
    for lay in my_layers:
        x, _ = lay.forward(x)
    return x
"""
```

```julia
torch_out = py"applyflow"(x|>jl2torch) |> torch2jl
```

```julia
@assert isapprox(torch_out, xout)
```

```julia
function create_layer()
    module_list = []
    for i ∈ 0:(n_layers-1)
        parity = mod(i, 2)
        channels = [inC, hidden_sizes..., outC]
        padding = kernel_size ÷ 2
        net = []
        for (ci, (c, c_next)) ∈ enumerate(pairwise(channels))
            push!(
                net, 
                Chain(
                    mycircular,
                    Conv(
                        affine_coupling_layers[i+1].net[ci][end].weight, 
                        affine_coupling_layers[i+1].net[ci][end].bias,
                        leakyrelu,
                        pad=0
                    )
                )
            )
        end
        if use_final_tanh
            c = channels[end-1]
            c_next = channels[end]
            net[end] = Chain(
                mycircular,
                Conv(
                    affine_coupling_layers[i+1].net[end][end].weight, 
                    affine_coupling_layers[i+1].net[end][end].bias,
                    tanh,
                    pad=0
                )
            )
        end
        mask = make_checker_mask(lattice_shape, parity)
        coupling = AffineCoupling(Chain(net...), mask)
        push!(module_list,coupling)
    end
    Chain(module_list...) |> f32
end

layer = create_layer()
```

```julia
xout, logq = create_layer()((x, logq_))
```

```julia
@assert isapprox(torch_out, xout)
```

```julia

```
