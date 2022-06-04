```julia
using CSV
using DataFrames
using StatsPlots

using GomalizingFlow
```

```julia
results = String[]
repo_dir = abspath(joinpath(dirname(dirname(pathof(GomalizingFlow)))))
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
prior_normal = 20
prior_potential = 24
```

```julia
r = results[prior_normal]
df = CSV.read(joinpath(r, "evaluations.csv"), DataFrame);
@df df plot(:epoch, :loss, label="loss", title="$(basename(r))")
```

```julia
r = results[prior_potential]
df = CSV.read(joinpath(r, "evaluations.csv"), DataFrame);
@df df plot(:epoch, :loss, label="loss", title="$(basename(r))")
```

# Acceptance_rate

```julia
r = results[prior_normal]
df = CSV.read(joinpath(r, "evaluations.csv"), DataFrame);

@df df plot(:epoch, :acceptance_rate, label="acceptance_rate", title="$(basename(r))")
```

```julia
r = results[prior_potential]
df = CSV.read(joinpath(r, "evaluations.csv"), DataFrame);

@df df plot(:epoch, :acceptance_rate, label="acceptance_rate", title="$(basename(r))")
```

# Ess

```julia
r = results[prior_normal]
df = CSV.read(joinpath(r, "evaluations.csv"), DataFrame);

@df df plot(:epoch, :ess, label="ess", title="$(basename(r))")
```

```julia
r = results[prior_potential]
df = CSV.read(joinpath(r, "evaluations.csv"), DataFrame);

@df df plot(:epoch, :ess, label="ess", title="$(basename(r))")
```
