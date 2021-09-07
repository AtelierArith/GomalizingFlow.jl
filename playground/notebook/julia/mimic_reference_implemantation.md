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
using PyCall
using PyPlot
using Random
using Distributions
using Flux

using LaTeXStrings
using ProgressMeter
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
py"""
import numpy as np
def init_simple_coupling_layer_params(seed=12345):
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=(8, 1)).astype(np.float32)
    w2 = rng.normal(size=(8, 8)).astype(np.float32)
    w3 = rng.normal(size=(1, 8)).astype(np.float32)
    b1 = -np.ones(8).astype(np.float32)
    b2 = -np.ones(8).astype(np.float32)
    b3 = -np.ones(1).astype(np.float32)
    return w1, w2, w3, b1, b2, b3
"""

struct SimpleCouplingLayer
    scaling
    function SimpleCouplingLayer()
        w1, w2, w3, b1, b2, b3 = py"init_simple_coupling_layer_params"()
        net = Chain(
            Dense(w1, b1, relu),
            Dense(w2, b2, relu),
            Dense(w3, b3, tanh),
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
np = pyimport("numpy")
```

```julia
py"""
def gen_sample(batch_size, seed=12345):
    rng = np.random.default_rng(seed)
    x1 = 2 * rng.random(size=batch_size).astype(np.float32) - 1
    x2 = 2 * rng.random(size=batch_size).astype(np.float32) - 1
    return x1, x2
"""
```

```julia
batch_size = 1024
rng = np.random.default_rng(12345)
x1 = 2rng.random(size=batch_size, dtype=np.float32) .- 1
x2 = 2rng.random(size=batch_size, dtype=np.float32) .- 1
x1, x2 = py"gen_sample"(batch_size)
gx1, gx2, logJ = forward(coupling_layer, x1, x2)
@assert gx1[1:3] ≈ [-0.25462535, -0.17111894, 0.27769268]
@assert gx2[1:3] ≈ [-0.26831156,  0.02142358, -0.92905575]

rev_x1, rev_x2, rev_logJ = reverse(coupling_layer, gx1, gx2)

fig, ax = plt.subplots(1, 3, figsize=(10, 3), sharex=true, sharey=true)
for a in ax
    a.set_xlim(-1.1, 1.1)
    a.set_ylim(-1.1, 1.1)
end
ax[1].scatter(x1, x2)
ax[2].scatter(gx1, gx2)
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

# ScalarPhi4Action

```julia
torch = pyimport("torch")
py"""
import torch
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
import numpy as np
def create_cfgs(L):
    rng = np.random.default_rng(2021)
    lattice_shape=(L, L)
    phi_ex1 = rng.normal(size=(lattice_shape)).astype(np.float32)
    phi_ex2 = rng.normal(size=(lattice_shape)).astype(np.float32)
    cfgs = np.stack((phi_ex1, phi_ex2), axis=0)
    return cfgs
"""

L = 8
m² = -4.0
λ = 8.0
const lattice_shape = (L, L)
```

```julia
struct ScalarPhi4Action
    m²::Float32
    λ::Float32
end 

function calc_action(action::ScalarPhi4Action, cfgs)
    action_density = @. action.m² * cfgs ^ 2 + action.λ * cfgs ^ 4
    Nd = lattice_shape |> length
    term1 = sum(2cfgs .^ 2 for μ in 2:Nd+1)
    term2 = sum(cfgs .* circshift(cfgs, Flux.onehot(μ, 1:(Nd+1))) for μ in 2:(Nd+1))
    term3 = sum(cfgs .* circshift(cfgs, -Flux.onehot(μ, 1:(Nd+1))) for μ in 2:(Nd+1))
    result = action_density .+ term1 .- term2 .- term3
    dropdims(
        sum(result, dims=2:Nd+1),
        dims=Tuple(2:(Nd+1))
    )
end

function (action::ScalarPhi4Action)(cfgs)
    calc_action(action, cfgs)
    #=
    action_density = @. action.m² * cfgs ^ 2 + action.λ * cfgs ^ 4
    Nd = lattice_shape |> length
    for μ ∈ 2:Nd+1
        action_density .+= 2cfgs .^ 2

        shifts_plus = zeros(Nd+1)
        shifts_plus[μ] = 1 # \vec{n} + \hat{\mu}
        action_density .-= cfgs .* circshift(cfgs, shifts_plus)

        shifts_minus = zeros(Nd+1)
        shifts_minus[μ] = -1 # \vec{n} - \hat{\mu}
        action_density .-= cfgs .* circshift(cfgs, shifts_minus)
    end
    return dropdims(sum(action_density, dims=2:Nd+1), dims=Tuple(2:(Nd+1)))
    =#
end

cfgs = py"create_cfgs"(L)
@assert ScalarPhi4Action(1, 1)(cfgs) ≈ [499.7602, 498.5477]
phi4_action = ScalarPhi4Action(m², λ)
@assert phi4_action(cfgs) ≈ [1598.679, 1545.5698]
```

```julia
torch.manual_seed(12345)

batch_size = 1024

prior = py"SimpleNormal"(
    torch.from_numpy(zeros(Float32, lattice_shape)), 
    torch.from_numpy(ones(Float32, lattice_shape)),
)

z = prior.sample_n(1).detach().numpy()
torch.manual_seed(1)
z = prior.sample_n(1024).detach().numpy()
@assert z[1,1,1] ≈ -1.5255959
@assert z[2, 5, 6] ≈ -0.81384623
@assert z[11, 4, 3] ≈ -0.3155666
S_eff = -prior.log_prob(torch.from_numpy(z)).detach().numpy()
@assert S_eff[1:3] ≈ [91.6709, 88.7396, 97.8660]
@assert S_eff[end-2:end] ≈ [92.81951, 91.93714, 80.611916]
S = phi4_action(z)
@assert S[1:3] ≈ [2103.6904, 1269.7277, 1697.3539]
@assert S[end-2:end] ≈ [1231.3823, 1500.0603, 600.58417]

fit_b = mean(S) - mean(S_eff)

fig, ax = plt.subplots(1,1, dpi=125, figsize=(4,4))
ax.hist2d(S_eff |> vec, S |> vec, bins=20, range=[[-800, 800], [200,1800]])
xs = range(-800, 800, length=4)
ax.plot(xs, xs .+ fit_b, ":", color="w", label="slope 1 fit")
ax.set_xlabel("\$S_{\\mathrm{eff}} \\equiv -\\log~r(z)\$")
ax.set_ylabel("S(z)")
ax.set_aspect("equal")
plt.legend(prop=Dict("size" => 6))
plt.show()
```

```julia
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
```

```julia
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
function pairwise(iterable)
    b = copy(iterable)
    popfirst!(b)
    a = iterable
    return zip(a, b)
end
```

```julia
prior = Normal{Float32}(0.f0, 1.f0)

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
reversedims(inp::AbstractArray{<:Any, N}) where {N} = permutedims(inp, N:-1:1)

function apply_affine_flow_to_prior(prior, affine_coupling_layers; batchsize)
    x = rand(prior, lattice_shape..., batchsize)
    logq_ = sum(logpdf(prior, x), dims=(1:ndims(x)-1))
    xout, logq = affine_coupling_layers((x, logq_))
    return xout, logq
end
```

```julia
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
py"""
import torch
torch.manual_seed(1234)
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

lattice_shape = (8, 8)
prior = SimpleNormal(torch.zeros(lattice_shape), torch.ones(lattice_shape))

def apply_affine_flow_to_prior(r, aff_coupling_layers, *, batch_size):
    z = r.sample_n(batch_size)
    logq = r.log_prob(z)
    x = z
    for lay in aff_coupling_layers:
        x, logJ = lay.forward(x)
        logq = logq - logJ
    return x, logq  # 点 x における `\log(q(x))` の値を計算
"""
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
n_era = 75 # 25 by default in original impl
epochs = 500 # 100 by default in original impl
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

function create_layer()
    module_list = []
    for i ∈ 0:(n_layers-1)
        parity = mod(i, 2)
        channels = [inC, hidden_sizes..., outC]
        padding = kernel_size ÷ 2
        net = []
        for (c, c_next) ∈ pairwise(channels)
            push!(net, mycircular) # TODO: consider circular
            push!(net, Conv((3,3), c=>c_next, leakyrelu, pad=0, bias = randn(Float32, c_next)))
        end
        if use_final_tanh
            c = channels[end-1]
            c_next = channels[end]
            net[end] = Conv((3,3), c=>c_next, tanh, pad=0, bias = randn(Float32, c_next))
        end
        mask = make_checker_mask(lattice_shape, parity)
        coupling = AffineCoupling(Chain(net...), mask)
        push!(module_list,coupling)
    end
    Chain(module_list...) |> f32
end

layer = create_layer()


ps = Flux.params(layer);
```

```julia
x, logq = apply_affine_flow_to_prior(prior, layer; batchsize)
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
py"""
import numpy as np
rng = np.random.default_rng(999)

def sample_dkl(batch_size=64):
    logp = np.random.random(batch_size)
    logq = np.random.random(batch_size)
    return (logp, logq)

def compute_ess(logp, logq):
    logw = torch.from_numpy(logp) - torch.from_numpy(logq)
    log_ess = 2*torch.logsumexp(logw, dim=0) - torch.logsumexp(2*logw, dim=0)
    ess_per_cfg = torch.exp(log_ess) / len(logw)
    return ess_per_cfg

def calc_dkl(logp, logq):
    return (logq - logp).mean()  # reverse KL, assuming samples from q

"""

calc_dkl(logp, logq) = mean(logq .- logp)

function compute_ess(logp, logq)
    logw = logp - logq
    log_ess = 2*logsumexp(logw) - logsumexp(2*logw)
    ess_per_cfg = exp(log_ess) / length(logw)
    return ess_per_cfg
end

logq, logp = py"sample_dkl"()
@assert calc_dkl(logp, logq) ≈ py"calc_dkl"(logp, logq)
@assert compute_ess(logp, logq) ≈ py"compute_ess"(logp, logq).item()
```

```julia
for era in 1:n_era
    @showprogress for e in 1:epochs
        gs = Flux.gradient(ps) do
            x, logq_ = apply_affine_flow_to_prior(prior, layer; batchsize)
            logq = dropdims(
                logq_,
                dims=Tuple(1:(ndims(logq_)-1))
            )
            logp = -calc_action(phi4_action, x |> reversedims)
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

```julia
x, logq = apply_affine_flow_to_prior(prior, layer; batchsize)
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
x, logq = apply_affine_flow_to_prior(prior, layer; batchsize)
S_eff = -logq
S = calc_action(phi4_action, x |> reversedims)
fit_b = mean(S) - mean(S_eff)
@show fit_b
print("slope 1 linear regression S = -logr + $fit_b")
fig, ax = plt.subplots(1,1, dpi=125, figsize=(4,4))
ax.hist2d(vec(S_eff), vec(S), bins=20, 
    #range=[[5, 35], [-5, 25]]
)

xs = range(-800, stop=800, length=4)
ax.plot(xs, xs .+ fit_b, ":", color=:w, label="slope 1 fit")
ax.set_xlabel(L"S_{\mathrm{eff}} \equiv -\log ~ r(z)")
ax.set_ylabel(L"S(z)")
ax.set_aspect(:equal)
plt.legend(prop=Dict("size"=> 6))
plt.show()
```
