```julia
using CSV
using DataFrames
using StatsPlots
using NaturalSort

using LFT
```

```julia
using Statistics: mean
movingaverage(g, n) = [i < n ? mean(g[begin:i]) : mean(g[i-n+1:i]) for i in 1:length(g)]
```

```julia
results = String[]
repo_dir = abspath(joinpath(dirname(dirname(pathof(LFT)))))
result_dir = joinpath(repo_dir, "result")
for d in sort(readdir(result_dir))
    if ispath(joinpath(result_dir, d, "config.toml"))
        push!(results, joinpath(result_dir, d))
    end
end

sort!(results, lt=natural);
```

```julia
for (i, r) in enumerate(results)
    println(i, " ", r)
end
```

# Training Loss

```julia
cases = 57:69 |> collect
```

```julia
p_loss = plot()
for c in cases
    r = results[c]
    df = CSV.read(joinpath(r, "evaluations.csv"), DataFrame);
    @df df plot!(p_loss, :epoch, :loss, label="loss-$(basename(r))")
end

plot(p_loss, title="Loss", size=(1000,500))
```

# Acceptance_rate

```julia
p_acceptance = plot()
for c in cases
    r = results[c]
    df = CSV.read(joinpath(r, "evaluations.csv"), DataFrame);

    #@df df plot!(p_acceptance, :epoch, :acceptance_rate, label="acceptance_rate-$(basename(r))", title="$(basename(r))", legend=:bottomright, alpha=0.5)
    plot!(p_acceptance, df.epoch, movingaverage(df.acceptance_rate, 10), label="moving average acceptance_rate-$(basename(r))")
end

plot(p_acceptance, title="Acceptance", legend = :outertop, size=(1200,500))
```

# Ess

```julia
p_ess = plot()
for c in cases
    r = results[c]
    df = CSV.read(joinpath(r, "evaluations.csv"), DataFrame);
    #@df df plot!(p_ess, :epoch, :ess, label="ess-$(basename(r))", title="$(basename(r))", alpha=0.5)
    plot!(p_ess, df.epoch, movingaverage(df.ess, 10), label="moving average for ess-$(basename(r))")
end

plot(p_ess, title="ess", legend = :outertop, size=(1200,500))
```
