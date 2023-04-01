# Analysis Tool

```julia
using Printf

using BSON
using PyPlot
using Distributions
using ProgressMeter
using IterTools
using Flux
using CUDA
using ParameterSchedulers
using Parameters

using GomalizingFlow
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
repo_dir = abspath(joinpath(dirname(dirname(pathof(GomalizingFlow)))))
result_dir = joinpath(repo_dir, "result")
for d in readdir(result_dir)
    if ispath(joinpath(result_dir, d, "config.toml"))
        push!(results, joinpath(result_dir, d))
    end
end
```

```julia
function restore(r)
    BSON.@load joinpath(r, "history_best_acceptance_rate.bson") history_best_acceptance_rate
    BSON.@load joinpath(r, "trained_model_best_acceptance_rate.bson") trained_model_best_acceptance_rate
    return Flux.testmode!(trained_model_best_acceptance_rate), history_best_acceptance_rate
end
```

```julia
function plot_action(r)
    hp = GomalizingFlow.load_hyperparams(joinpath(r, "config.toml"));
    @show hp
    model, _ = restore(r);

    batchsize = 1024
    strprior = hp.tp.prior
    prior = eval(Meta.parse(strprior))
    phi4_action = GomalizingFlow.ScalarPhi4Action{Float32}(hp.pp.m², hp.pp.λ)
    device = Flux.cpu
    lattice_shape = hp.pp.lattice_shape

    z = rand(prior, lattice_shape..., batchsize)
    logq_device = sum(logpdf.(prior, z), dims=(1:ndims(z) - 1)) |> device

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
for (i, r) in enumerate(results)
    println(i, " ", r)
end
```

```julia
r = results[2] # modify here
@show r
```

```julia
plot_action(r)
plt.show()
```

# Generate configurations

```julia
hp = GomalizingFlow.load_hyperparams(joinpath(r, "config.toml"))
prior = eval(Meta.parse(hp.tp.prior))
T = prior |> rand |> eltype
@unpack m², λ, lattice_shape = hp.pp
action = GomalizingFlow.ScalarPhi4Action{T}(m², λ)
batchsize = 64
model, _ = restore(r)
nsamples = 1000000
history = GomalizingFlow.make_mcmc_ensamble(
    model |> gpu,
    prior,
    action,
    lattice_shape;
    batchsize,
    nsamples,
    device=gpu,
    seed=2009,
);
```

```julia
acceptance_rate =  mean(history[:accepted][5000:end])
Printf.@printf "acceptance_rate= %.2f [percent]" 100acceptance_rate

function drawgreen(history)
    cfgs = Flux.MLUtils.batch(history[:x][5000:end])
    y_values = eltype(cfgs)[]
    @showprogress for t in 0:hp.pp.L
        y = mfGc(cfgs, t, lattice_shape)
        push!(y_values, y)
    end
    plt.plot(0:hp.pp.L, y_values)
    #plt.yscale("log")
end

drawgreen(history)
```

$$
\langle \prod_{i=1}^\tau \mathbb{1}_\mathrm{rej}(i) \rangle
$$

```julia
function τⁱⁿᵗ_acc(history)
    a = history[:accepted][5000:end];
    τⁱⁿᵗ_acc = 0.5 + sum(1:100) do τ
        p_τrej = mean(1:(length(a) - τ)) do i 
            prod(.!(@view a[i:(i + τ - 1)]))
        end
        p_τrej
    end
end
τⁱⁿᵗ_acc(history)
```

$$
\tau^{\mathrm{int}}_{\mathcal{O}} = \frac{1}{2} + \lim_{\tau_{\mathrm{max}} \to \infty} \sum_{\tau=1}^{\tau_{\mathrm{max}}} 
\widehat{\rho_{\mathcal{O}}(\tau)/\rho_{\mathcal{O}}(0)}
$$

```julia
function τⁱⁿᵗ(𝒪)
	N = length(𝒪)
	return 0.5 + sum(1:100) do τ
		𝒪̄ = mean(𝒪)
		n = mean(1:(N-τ)) do i
			(𝒪[i] - 𝒪̄) * (𝒪[i + τ] - 𝒪̄)
		end
		d = mean(1:N) do i
			(𝒪[i] - 𝒪̄) ^ 2
		end
		n/d
	end
end
```

```julia
a = history[:accepted][5000:end];
τⁱⁿᵗ(a)
```

```julia
using StatsBase: autocor
```

```julia
a = history[:accepted][2000:end];
```

```julia
0.5 + sum(autocor(a)[begin+1:end])
```

# Histograms of rejections

```julia
function collect_rejections(arr)
	cnts = Int[]
	cnt = 0
	si = 1
	ei = 1
	iscounting = false
	lastindex = axes(arr)[begin][end]
	for (i, b) in enumerate(arr)
		if !iscounting && !b
			iscounting = true
			si = ei = i
		end
		
		if iscounting && !b 
			ei = i
		elseif iscounting && b
			push!(cnts, ei - si + 1)
			si = ei
			iscounting = false
		end
		
		if i == lastindex && iscounting
			push!(cnts, lastindex - si + 1)
		end
	end
	cnts
end
```

```julia
rejcounts = filter(collect_rejections(a)) do c
    c < 80
end

hist(rejcounts, log=true);
```

```julia

```
