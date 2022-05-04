# TODO

- 移動平均をとる
- 

```julia
using CSV
using DataFrames
using StatsPlots

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
```

```julia
for (i, r) in enumerate(results)
    println(i, " ", r)
end
```

# Training Loss

```julia
case1 = 1
case2 = 2
```

```julia
r = results[case1]
p_loss = plot()
df = CSV.read(joinpath(r, "evaluations.csv"), DataFrame);
@df df plot!(p_loss, :epoch, :loss, label="loss-$(basename(r))", title="$(basename(r))")
```

```julia
r = results[case2]
df = CSV.read(joinpath(r, "evaluations.csv"), DataFrame);
@df df plot!(p_loss, :epoch, :loss, label="loss-$(basename(r))", title="$(basename(r))")
```

# Acceptance_rate

```julia
p_acceptance = plot()

r = results[case1]
df = CSV.read(joinpath(r, "evaluations.csv"), DataFrame);

@df df plot!(p_acceptance, :epoch, :acceptance_rate, label="acceptance_rate-$(basename(r))", title="$(basename(r))", legend=:bottomright)
plot!(p_acceptance, df.epoch, movingaverage(df.acceptance_rate, 10), label="moving average acceptance_rate-$(basename(r))")
```

```julia
r = results[case2]
df = CSV.read(joinpath(r, "evaluations.csv"), DataFrame);

@df df plot!(p_acceptance, :epoch, :acceptance_rate, label="acceptance_rate-$(basename(r))", title="$(basename(r))", legend=:bottomright)
plot!(p_acceptance, df.epoch, movingaverage(df.acceptance_rate, 10), label="moving average acceptance_rate-$(basename(r))")
```

# Ess

```julia
p_ess = plot()

r = results[case1]
df = CSV.read(joinpath(r, "evaluations.csv"), DataFrame);

@df df plot!(p_ess, :epoch, :ess, label="ess-$(basename(r))", title="$(basename(r))")
plot!(p_ess, df.epoch, movingaverage(df.ess, 10), label="moving average for ess-$(basename(r))")
```

```julia
r = results[case2]
df = CSV.read(joinpath(r, "evaluations.csv"), DataFrame);

@df df plot!(p_ess, :epoch, :ess, label="ess-$(basename(r))", title="Compare", legend=:topleft)
plot!(p_ess, df.epoch, movingaverage(df.ess, 10), label="moving average for ess-$(basename(r))")
```
