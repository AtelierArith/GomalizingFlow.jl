# Draw green and autocorr

```julia
using GomalizingFlow
using ProgressMeter
using Plots
using Flux
using CUDA
using ParameterSchedulers
```

```julia
results = String[]
repo_dir = pkgdir(GomalizingFlow)
result_dir = joinpath(repo_dir, "result")
for d in readdir(result_dir)
    if ispath(joinpath(result_dir, d, "config.toml"))
        push!(results, joinpath(result_dir, d))
    end
end

for (i, r) in enumerate(results)
    println(i, " ", r)
end
```

```julia
function drawgreen(r)
    hp = GomalizingFlow.load_hyperparams(joinpath(r, "config.toml"))
    _, history = GomalizingFlow.restore(r);
    lattice_shape = hp.pp.lattice_shape
    #cfgs = cat(history[:x][4000:7000]..., dims=length(lattice_shape)+1)
    cfgs = Flux.MLUtils.batch(history[:x][2000:7000])
    y_values = []
    @showprogress for t in 0:hp.pp.L
        y = GomalizingFlow.mfGc(cfgs, t)
        push!(y_values, y)
    end
    plot(0:hp.pp.L, y_values)
    #plt.yscale("log")
end

r = results[1]
drawgreen(r)
```

```julia
function compute_integrated_autocorr(r::AbstractString)
    hp = GomalizingFlow.load_hyperparams(joinpath(r, "config.toml"))
    _, history = GomalizingFlow.restore(r);
    arr = history.accepted[2000:end]
    GomalizingFlow.integrated_autocorrelation_time(arr)
end

r = results[1]
compute_integrated_autocorr(r)
```
