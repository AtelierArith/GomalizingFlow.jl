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

# Metropolis

```julia
using Distributions
using Flux
using LaTeXStrings
using PyPlot
using ProgressMeter
```

```julia
L = 8
```

```julia
lattice_shape = (8, 8)
```

```julia
struct ScalarPhi4Action
    m²
    λ
end

m² = -4.0f0
λ = 8.0f0
phi4_action = ScalarPhi4Action(m², λ)
prior = Normal{Float32}(0.f0, 1.f0)

n_layers = 16
hidden_sizes = [8,8]
kernel_size = 3
```

```julia
function goma(xs)
    ys = [i for i in 1:length(xs)]
    sum(xs .+ ys)
end

gradient(goma, [2,3,4])
```

```julia
function calc_action(sp4a::ScalarPhi4Action, cfgs)
    action_density = @. sp4a.m² * cfgs ^ 2 + sp4a.λ * cfgs ^ 4
    Nd = lattice_shape |> length
    term1 = sum(2cfgs .^ 2 for μ in 1:Nd)
    term2 = sum(cfgs .* circshift(cfgs, Flux.onehot(μ, 1:(Nd+1))) for μ in 1:Nd)
    term3 = sum(cfgs .* circshift(cfgs, -Flux.onehot(μ, 1:(Nd+1))) for μ in 1:Nd)
    sum(action_density .+ term1 .- term2 .- term3, dims=1:Nd)
    #=
    for μ ∈ 1:Nd
        action_density .+= 2cfgs .^ 2
        #shifts_plus = zeros(Nd+1)
        #shifts_plus[μ] = 1 # \vec{n} + \hat{\mu}
        shifts_plus = Flux.onehot(μ, 1:(Nd+1))
        action_density .-= cfgs .* circshift(cfgs, shifts_plus)

        #shifts_minus = zeros(Nd+1)
        #shifts_minus[μ] = -1 # \vec{n} - \hat{\mu}
        shifts_minus = -Flux.onehot(μ, 1:(Nd+1))
        action_density .-= cfgs .* circshift(cfgs, shifts_minus)
    end
    return sum(action_density, dims=1:Nd)
    =#
end

function _calc_action(sp4a::ScalarPhi4Action, cfgs)
    action_density = @. sp4a.m² * cfgs ^ 2 + sp4a.λ * cfgs ^ 4
    Nd = lattice_shape |> length
    for μ ∈ 1:Nd
        action_density .+= 2cfgs .^ 2

        shifts_plus = zeros(Nd+1)
        shifts_plus[μ] = 1 # \vec{n} + \hat{\mu}
        action_density .-= cfgs .* circshift(cfgs, shifts_plus)

        shifts_minus = zeros(Nd+1)
        shifts_minus[μ] = -1 # \vec{n} - \hat{\mu}
        action_density .-= cfgs .* circshift(cfgs, shifts_minus)
    end
    return sum(action_density, dims=1:Nd)
end
```

```julia
batch_size = 1024
z = rand(prior, (lattice_shape..., batch_size))
fig, ax = plt.subplots(4,4, dpi=125, figsize=(4,4))
for i in 1:4
    for j in 1:4
        ind = i*4 + j
        ax[i,j].imshow(tanh(z[:, :, ind]), vmin=-1, vmax=1, cmap="viridis")
        ax[i,j].axes.xaxis.set_visible(false)
        ax[i,j].axes.yaxis.set_visible(false)
    end
end
```

```julia
Nd = lattice_shape |> length
fig, ax = plt.subplots(4, 4, dpi=125, figsize=(4,4))
for x1 in 1:Nd
    for y1 in 1:Nd
        i1 = (x1-1)*2 + y1
        for x2 in 1:Nd
            for y2 in 1:Nd
                i2 = (x2 -1)* 2 + y2
                ax[i1, i2].hist2d(z[x1,y1,:], z[x2,y2,:], range=[[-3,3],[-3,3]], bins=20)
                ax[i1, i2].set_xticks([])
                ax[i1, i2].set_yticks([])
                if i1 == 4
                    ax[i1, i2].set_xlabel(latexstring("\\phi($x2,$y2)"))
                end
                if i2 == 1
                    ax[i1, i2].set_ylabel(latexstring("\\phi($x1,$y1)"))
                end
            end
        end
    end
end
```

```julia
S_eff = -sum(logpdf(prior, z), dims=1:length(lattice_shape))
S = calc_action(phi4_action, z)
fit_b = mean(S) - mean(S_eff)
print("slope 1 linear regression S = -logr + $fit_b")
fig, ax = plt.subplots(1,1, dpi=125, figsize=(4,4))
ax.hist2d(vec(S_eff), vec(S), bins=20, range=[[-800, 800], [200,1800]])
xs = range(-800, stop=800, length=4)
ax.plot(xs, xs .+ fit_b, ":", color=:w, label="slope 1 fit")
ax.set_xlabel(L"S_{\mathrm{eff}} \equiv -\log ~ r(z)")
ax.set_ylabel(L"S(z)")
ax.set_aspect(:equal)
plt.legend(prop=Dict("size"=> 6))
plt.show()
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
c = Conv((3,3), 1=>8, bias=true, stride=1)
for p in Flux.params(c)
    @show p |> eltype
    @show p |> size
end
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
struct AffineNet
    net
end

Flux.@functor AffineNet

(a::AffineNet)(x) = a.net(x)

struct AffineCoupling
    net::AffineNet
    mask
end
```

```julia
#=
x_torch = (B, nC, inH, inW)
x_flux = (inW, inH, inC, inB)
=#

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
        coupling = AffineCoupling(AffineNet(net), mask)
        push!(layers, coupling)
    end
    Chain(layers...)
end
```

```julia
function apply_flow_to_prior(prior, coupling_layers; batchsize)
    x = rand(prior, lattice_shape..., batchsize)
    logq_ = sum(logpdf(prior, x), dims=(1:ndims(x)-1))
    xout, logq = coupling_layers((x, logq_))
    #=for layer in coupling_layers
        x, logJ = layer(x)
        logq .-= logJ
    end
    =#
    return xout, logq
end
```

```julia
calc_dkl(logp, logq) = mean(logq .- logp)

function compute_ess(logp, logq)
    logw = logp - logq
    log_ess = 2*logsumexp(logw, dim=ndims(logw)) - logsumexp(2*logw, dim=ndims(logw))
    ess_per_cfg = exp(log_ess) / len(logw)
    return ess_per_cf
end
```

```julia
model = make_phi4_affine_layers(
    lattice_shape=lattice_shape, 
    n_layers=n_layers,
    hidden_sizes=hidden_sizes, 
    kernel_size=kernel_size
);
```

```julia
n_era = 25
epochs = 100
batchsize = 64

base_lr = 0.001f0
opt = ADAM(base_lr)
ps = Flux.params(model);
```

```julia
for era in 1:n_era
    @showprogress for e in 1:epochs
    gs = Flux.gradient(ps) do
        x, logq = apply_flow_to_prior(prior, model; batchsize)
        logp = -calc_action(phi4_action, x)
        loss = calc_dkl(logp, logq)
    end
    Flux.Optimise.update!(opt, ps, gs)
    end
end
```

```julia
x, logq = apply_flow_to_prior(prior, model; batchsize)
fig, ax = plt.subplots(4,4, dpi=125, figsize=(4,4))
for i in 1:4
    for j in 1:4
        ind = i*4 + j
        ax[i,j].imshow(tanh(x[:, :, ind]), vmin=-1, vmax=1, cmap=:viridis)
        ax[i,j].axes.xaxis.set_visible(false)
        ax[i,j].axes.yaxis.set_visible(false)
    end
end
```

```julia
S_eff = -logq
S = calc_action(phi4_action, x)
fit_b = mean(S) - mean(S_eff)
print("slope 1 linear regression S = -logr + $fit_b")
fig, ax = plt.subplots(1,1, dpi=125, figsize=(4,4))
ax.hist2d(vec(S_eff), vec(S), bins=20, range=[[-800, 800], [200,1800]])
xs = range(-800, stop=800, length=4)
ax.plot(xs, xs .+ fit_b, ":", color=:w, label="slope 1 fit")
ax.set_xlabel(L"S_{\mathrm{eff}} \equiv -\log ~ r(z)")
ax.set_ylabel(L"S(z)")
ax.set_aspect(:equal)
plt.legend(prop=Dict("size"=> 6))
plt.show()
```
