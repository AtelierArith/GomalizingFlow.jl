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

```julia
using PyCall
using PyPlot
using Random
using Distributions
using Flux
using EllipsisNotation
using LaTeXStrings
using ProgressMeter
```

```julia
using CUDA
CUDA.device!(1)
use_cuda = true

if use_cuda && CUDA.functional()
    device = gpu
    @info "Training on GPU"
else
    device = cpu
    @info "Training on CPU"
end
```

```julia
struct ScalarPhi4Action
    m²::Float32
    λ::Float32
end 

"""
cfgs |> size == (batchsize, dims...)
"""
function calc_action(action::ScalarPhi4Action, cfgs)
    action_density = @. action.m² * cfgs ^ 2 + action.λ * cfgs ^ 4
    Nd = ndims(cfgs) - 1 # exclude batch dimension
    term1 = sum(2cfgs .^ 2 for μ in 2:Nd+1)
    term2 = sum(cfgs .* circshift(cfgs, Flux.onehot(μ, 1:(Nd+1))) for μ in 2:(Nd+1))
    term3 = sum(cfgs .* circshift(cfgs, -Flux.onehot(μ, 1:(Nd+1))) for μ in 2:(Nd+1))
    result = action_density .+ term1 .- term2 .- term3
    dropdims(
        sum(result, dims=2:Nd+1),
        dims=Tuple(2:(Nd+1))
    )
end

(action::ScalarPhi4Action)(cfgs) = calc_action(action, cfgs)
```

```julia
function make_checker_mask(shape::NTuple{3, Int}, parity)
    checker = ones(Int, shape) .- parity
    checker[begin:2:end, begin:2:end, begin:2:end] .= parity
    checker[(begin+1):2:end, (begin+1):2:end, begin:2:end] .= parity
    
    checker[(begin+1):2:end, begin:2:end, (begin+1):2:end] .= parity
    checker[begin:2:end, (begin+1):2:end, (begin+1):2:end] .= parity

    return checker
end
```

```julia
function pairwise(iterable)
    b = deepcopy(iterable)
    popfirst!(b)
    a = iterable
    return zip(a, b)
end
```

```julia
struct AffineCoupling
    net
    mask
end

Flux.@functor AffineCoupling (net,)

#=
x_torch = (B, nC, inH, inW)
x_flux = (inW, inH, inC, inB)
=#

function (model::AffineCoupling)(x_pair_loghidden)
    x = x_pair_loghidden[begin]
    loghidden = x_pair_loghidden[end]
    x_frozen = model.mask .* x
    x_active = (1 .- model.mask) .* x
    # (inW, inH, inD, inB) -> (inW, inH, inD, 1, inB) # by Flux.unsqueeze(*, 4)
    net_out = model.net(Flux.unsqueeze(x_frozen, ndims(x_frozen)))
    s = @view net_out[.., 1, :] # extract feature from 1st channel
    t = @view net_out[.., 2, :] # extract feature from 2nd channel
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
```

```julia
function mycircular(Y)
    # calc Z_bottom
    Y_b_c = Y[begin:begin,:,:,:,:]
    Y_b_r = Y[begin:begin,end:end,:,:,:]
    Y_b_l = Y[begin:begin,begin:begin,:,:,:]
    Z_bottom = cat(Y_b_r, Y_b_c, Y_b_l, dims=2) # calc pad under
    
    # calc Z_top
    Y_e_c = Y[end:end,:,:,:,:]
    Y_e_r = Y[end:end,end:end,:,:,:]
    Y_e_l = Y[end:end,begin:begin,:,:,:]
    Z_top = cat(Y_e_r, Y_e_c, Y_e_l, dims=2)
    
    # calc Z_main
    Y_main_l = Y[:,begin:begin,:,:,:]
    Y_main_r = Y[:,end:end,:,:,:]
    Z_main = cat(Y_main_r, Y, Y_main_l, dims=2)
    Z_ = cat(Z_top, Z_main, Z_bottom, dims=1)
    
    # pad along 3rd axis
    Z_begin = Z_[:,:, begin:begin,:,:]
    Z_end = Z_[:,:, end:end,:,:]
    cat(Z_end, Z_, Z_begin, dims=3)
end
```

```julia
const n_era = 25
const epochs = 100
const batchsize = 64

base_lr = 0.0075f0
opt = ADAM(base_lr)
const L = 8
const lattice_shape = (L, L, L)
const M2 = -4.
const lam = 8.
const phi4_action = ScalarPhi4Action(M2, lam)


const n_layers = 16
const hidden_sizes = (8, 8)
const kernel_size = 3
const inC = 1
const outC = 2
const use_final_tanh = true

const prior = Normal{Float32}(0.f0, 1.f0)

function create_layer(;ksize)
    module_list = []
    for i ∈ 0:(n_layers-1)
        parity = mod(i, 2)
        channels = [inC, hidden_sizes..., outC]
        net = []
        for (c, c_next) ∈ pairwise(channels)
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
            k = 1/(c * prod(ksize))
            W = rand(Uniform(-√k, √k), ksize..., c, c_next)
            b = rand(Uniform(-√k, √k), c_next) 
            push!(net, mycircular)
            push!(net, Conv(W, b, leakyrelu, pad=0))
        end
        if use_final_tanh
            c = channels[end-1]
            c_next = channels[end]
            k = 1/(c * prod(ksize))
            W = rand(Uniform(-√k, √k), ksize..., c, c_next)
            b = rand(Uniform(-√k, √k), c_next) 
            net[end] = Conv(W, b, tanh, pad=0)
        end
        mask = make_checker_mask(lattice_shape, parity)|> device
        coupling = AffineCoupling(Chain(net...), mask)
        push!(module_list,coupling)
    end
    Chain(module_list...) |> f32 |> device
end

layer = create_layer(ksize=(3, 3, 3))
ps = Flux.params(layer);
```

```julia
function apply_affine_flow_to_prior(prior, affine_coupling_layers; batchsize)
    x = rand(prior, lattice_shape..., batchsize)
    logq_ = sum(logpdf(prior, x), dims=(1:ndims(x)-1)) |> device
    xout, logq = affine_coupling_layers((x |> device, logq_))
    return xout, logq
end

calc_dkl(logp, logq) = mean(logq .- logp)

function compute_ess(logp, logq)
    logw = logp - logq
    log_ess = 2*logsumexp(logw) - logsumexp(2*logw)
    ess_per_cfg = exp(log_ess) / length(logw)
    return ess_per_cfg
end

reversedims(inp::AbstractArray{<:Any, N}) where {N} = permutedims(inp, N:-1:1)
```

```julia
for era in 1:n_era
    @showprogress for e in 1:epochs
        x = rand(prior, lattice_shape..., batchsize)
        logq_in = sum(logpdf(prior, x), dims=(1:ndims(x)-1)) |> device
        xin = x |> device
        gs = Flux.gradient(ps) do
            xout, logq_out = layer((xin, logq_in))
            logq = dropdims(
                logq_out,
                dims=Tuple(1:(ndims(logq_out)-1))
            )
            logp = -calc_action(phi4_action, xout |> reversedims)
            loss = calc_dkl(logp, logq)
        end
        Flux.Optimise.update!(opt, ps, gs)
    end
    x, logq_ = apply_affine_flow_to_prior(prior, layer; batchsize)
    logq = dropdims(
        logq_,
        dims=Tuple(1:(ndims(logq_)-1))
    )

    logp = -calc_action(phi4_action, x |> reversedims)
    loss = calc_dkl(logp, logq)
    @show loss
    ess = compute_ess(logp, logq)
    @show ess
end
```
