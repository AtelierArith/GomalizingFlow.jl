---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: julia 1.6.3
    language: julia
    name: julia-1.6
---

# Analysis Tool

```julia
using Printf

using BSON
using PyPlot
using Distributions
using ProgressMeter
using IterTools
using Flux

using LFT
```

```julia
using Base.Threads
if nthreads() == 1
    msg = """
    We recommend using more than nthreads > 1 to calculate the Green function 
    Please run the following command on your terminal (not here)
    \$ julia --threads auto -e 'using Base.Threads, IJulia; installkernel("julia-\$(nthreads())-threads", env=Dict("JULIA_NUM_THREADS"=>"\$(nthreads())"))'
    Then use the kernel named julia-<numthreads>-threads $VERSION where <numthreads> depends on your environment
    """
    @warn msg
else
    @info "using $(nthreads())-threads"
end
```

```julia
results = String[]
repo_dir = abspath(joinpath(dirname(dirname(pathof(LFT)))))
result_dir = joinpath(repo_dir, "result")
for d in readdir(result_dir)
    if ispath(joinpath(result_dir, d, "config.toml"))
        push!(results, joinpath(result_dir, d))
    end
end
```

```julia
function restore(r)
    BSON.@load joinpath(r, "history.bson") history
    BSON.@load joinpath(r, "trained_model.bson") trained_model
    return Flux.testmode!(trained_model), history
end
```

```julia
function plot_action(r)
    hp = LFT.load_hyperparams(joinpath(r, "config.toml"));
    model, _ = restore(r);

    batchsize = 1024
    strprior = hp.tp.prior
    prior = eval(Meta.parse(strprior))
    phi4_action = LFT.ScalarPhi4Action(hp.pp.m², hp.pp.λ)
    device = Flux.cpu
    lattice_shape = hp.pp.lattice_shape

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
    return fig
end
```

# Plot momemtum free Green function

```julia
function green(cfgs, offsetX, lattice_shape)
    shifts = (broadcast(-, offsetX)..., 0)
    batch_dim = ndims(cfgs)
    cfgs_offset = circshift(cfgs, shifts)
    m_corr = mean(cfgs .* cfgs_offset, dims=batch_dim)
    m = mean(cfgs, dims=batch_dim)
    m_offset = mean(cfgs_offset, dims=batch_dim)
    V = prod(lattice_shape)
    Gc = sum(m_corr .- m .* m_offset)/V
    return Gc 
end
```

```julia
function mfGc(cfgs, t, lattice_shape)
    maxT = lattice_shape[end]
    space_shape = size(cfgs)[begin:length(lattice_shape)-1]
    acc = Atomic{Float32}(0)
    @threads for s in IterTools.product((1:l for l in space_shape)...) |> collect
        Threads.atomic_add!(acc, green(cfgs, (s..., t), lattice_shape))
    end
    return acc.value /= prod(space_shape)
end
```

```julia
results
for (i, r) in enumerate(results)
    println(i, r)
end
```

```julia
r = results[1] # modify here
@show r
```

```julia
plot_action(r)
plt.show()
```

```julia
_, history = restore(r);
accepted_ratio =  mean(history[:accepted][4000:7000])
Printf.@printf "accepted_ratio= %.2f [percent]" 100accepted_ratio

function drawgreen(r)
    hp = LFT.load_hyperparams(joinpath(r, "config.toml"))
    _, history = restore(r);
    lattice_shape = hp.pp.lattice_shape
    cfgs = cat(history[:x][4000:7000]..., dims=length(lattice_shape)+1)
    y_values = []
    @showprogress for t in 0:hp.pp.L
        y = mfGc(cfgs, t, lattice_shape)
        push!(y_values, y)
    end
    plt.plot(0:hp.pp.L, y_values)
    plt.yscale("log")
end

drawgreen(r)
```
