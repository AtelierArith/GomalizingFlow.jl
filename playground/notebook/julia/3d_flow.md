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

```julia
using PyCall
using PyPlot
using Random
using Distributions
using Flux
using IterTools
using EllipsisNotation
using LaTeXStrings
using ProgressMeter
```

```julia
using CUDA: CUDA

if CUDA.functional()
    device_id = 1 # 0, 1, 2 ...
    CUDA.device!(device_id)
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
n_era = 300
epochs = 100
batchsize = 64

base_lr = 0.0015f0

opt = ADAM(base_lr)



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
    @show "loss per site" loss/prod(lattice_shape)
    ess = compute_ess(logp, logq)
    @show ess
end
```

```julia
x, logq = apply_affine_flow_to_prior(prior, layer; batchsize=1024)
x = cpu(x)
S_eff = -logq |> cpu
S = calc_action(phi4_action, x |> reversedims)
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
function make_mcmc_ensamble(layer, prior, action; batchsize, nsamples)
    history=(x=Array{Float32, 3}[], logq=Float32[], logp=Float32[], accepted=Bool[])
    c = 0
    for _ in 1:(nsamples÷batchsize + 1)
        x_device, logq_ = apply_affine_flow_to_prior(prior, layer; batchsize)
        logq = dropdims(
            logq_,
            dims=Tuple(1:(ndims(logq_)-1))
        ) |> cpu
        logp = -calc_action(phi4_action, x_device |> reversedims) |> cpu
        x = x_device |> cpu
        for b in 1:batchsize
            new_x = x[.., b]
            new_logq = logq[b]
            new_logp = logp[b]
            if isempty(history[:logp])
                accepted = true
            else
                last_logp = history[:logp][end]
                last_logq = history[:logq][end]
                last_x = history[:x][end]
                p_accept = exp((new_logp - new_logq) - (last_logp - last_logq))
                p_accept = min(one(p_accept), p_accept)
                draw = rand()
                if draw < p_accept
                    accepted = true
                else
                    accepted = false
                    new_x = last_x
                    new_logp = last_logp
                    new_logq = last_logq
                end
            end
            # update history
            push!(history[:logp], new_logp)
            push!(history[:logq], new_logq)
            push!(history[:x], new_x)
            push!(history[:accepted], accepted)
        end
        c += batchsize
        if c >= nsamples
            break
        end
    end
    history
end
```

```julia
ensamble_size = 8192
history = make_mcmc_ensamble(layer, prior, phi4_action, batchsize=64, nsamples=ensamble_size)
@show mean(history[:accepted])
```

```julia
history[:accepted] |> mean
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
plt.plot(0:L, [mfGc(cfgs, t) for t in 0:L])
```

```julia
function approx_normalized_autocorr(observed::AbstractVector, τ::Int)
    ō = mean(observed)
    N = length(observed)
    s = zero(eltype(observed))
    for i in 1:(N-τ)
        s += (observed[i]-ō)*(observed[i+τ]-ō)
    end
    s = s/(N-τ)/var(observed)
    return s
end

ρ̂(observed, τ) = approx_normalized_autocorr(observed, τ)
```

```julia
function ρ̂_acc(accepts, τ)
    N = length(accepts)
    τ = 10
    s = 0
    for j in 1:(N-τ)
        s += prod(accepts[j + i] for i in 1:τ)
    end
    s /= (N - τ)
    return s
end

τ_accⁱⁿᵗ = 0.5 + sum(ρ̂_acc(history[:accepted], τ) for τ in 1:100)
```

```julia
χ₂ = zero(eltype(cfgs))

@showprogress for pos in IterTools.product((1:l for l in lattice_shape)...)
    χ₂ += green(cfgs, pos)
end

χ₂
```

```julia tags=[]
E = calc_action(phi4_action, cfgs |> reversedims)
τᵢₙₜ = 0.5
for τ in 1:1000
    τᵢₙₜ += ρ̂(E, τ)
end
τᵢₙₜ
```

# https://arxiv.org/pdf/hep-lat/0409106.pdf

```julia
function auto_corr(a::AbstractVector, t::Int) # \bar{\Gamma}
    t = abs(t)
    ā = mean(a)
    s = zero(eltype(a))
    N = length(a)
    for i in 1:(N-t)
        s += (a[i] - ā) * (a[i+t] - ā)
    end
    return s / (N - t)
end

function δρ²(a, t)
    Λ = 100
    s = 0.
    for k in 1:(t + Λ)
        s += (ρ̄(a, k + t) + ρ̄(a, k - t) - 2ρ̄(a, k) * ρ̄(a, t))^2
    end
    s /= length(a)
end

W = -1

for t in 1:1000
    if ρ̄(a, t) ≤ √(δρ²(a, t))
        W = t
        break
    end
end

a = history[:accepted]
ρ̄(a, t) = auto_corr(a, t)/auto_corr(a, 0)

τᵢₙₜ = 0.5 + sum(t->ρ̄(a, t), 1:W)
```
