# autocorrelation times

Reference: https://arxiv.org/pdf/hep-lat/0409106.pdf Appendix E


Approximated autocorrelation function

$$
\bar{\Gamma}(t) = \frac{1}{N-t} \sum_{i=1}^{N-t} (a_i - \bar{a})(a_{i+t} - \bar{a})
$$

```julia
using Statistics
```

```julia
"""
Compute \$\\bar{\\Gamma}(t)\$
"""
function approx_autocorr(a::AbstractVector{T}, t::Int) where T
    t = abs(t)
    ā = mean(a)
    s = zero(T)
    N = length(a)
    for i in 1:(N-t)
        s += (a[i] - ā) * (a[i+t] - ā)
    end
    return s / (N - t)
end
```

Normalized autocorrelation function

$$
\rho(t) = \bar{\Gamma}(t)/\bar{\Gamma}(0)
$$

```julia
approx_normalized_autocorr(arr, t) = approx_autocorr(arr, t)/approx_autocorr(arr, 0)
ρ̄(arr, t) = approx_normalized_autocorr(arr, t) # alias
```

$$
\langle \delta \rho (t) ^2 \rangle \simeq \frac{1}{N} \sum_{k=1}^{t + \Lambda}\left(\bar\rho(k+t) + \bar\rho(k-t) - 2\bar\rho(k)\bar\rho(t) \right)^2
$$

```julia
# variance of the autocorrelation function
function δρ²(a::AbstractArray{T}, t; Λ=600) where T
    s = zero(T)
    for k in 1:(t + Λ)
        s += (ρ̄(a, k + t) + ρ̄(a, k - t) - 2ρ̄(a, k) * ρ̄(a, t))^2
    end
    return s / length(a)
end
```

The integrated autocorrelation time

$$
\tau_{\mathrm{int}} = 0.5 + \sum_{t=1}^W \bar\rho(t)
$$

```julia
function integrated_autocorrelation_time(arr)
    W = -1
    for t in 1:length(arr)
        if ρ̄(arr, t) ≤ √(δρ²(arr, t))
            W = t
            break
        end
    end

    0.5 + sum(ρ̄(arr, t) for t in 1:W)
end
```

```julia
using GomalizingFlow
using BSON
```

```julia
function restore(r::AbstractString)
    BSON.@load joinpath(r, "history.bson") history
    return history
end
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
r = results[2]
```

```julia
history = restore(r)
arr = history.accepted[2000:end]
τᵢₙₜ = integrated_autocorrelation_time(arr)
```
