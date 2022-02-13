---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.6
  kernelspec:
    display_name: julia 1.7.1
    language: julia
    name: julia-1.7
---

```julia
using Random
using Distributions
using StatsPlots
```

```julia
@time rand(Xoshiro(0), Float32, 8, 8, 64);
```

```julia
@time rand(MersenneTwister(0), Float32, 8, 8, 64);
```

```julia
@time rand(Xoshiro(0), Normal(0.,1.), 8, 8, 64);
```

```julia
@time rand(MersenneTwister(0), Normal(0.,1.), 8, 8, 64);
```

```julia
module My

import Base: minimum, maximum # work with @distr_spport

using Distributions
import Distributions: pdf, logpdf, @distr_support
using Random

v(x::T, m²::T, λ::T) where T<:Real = m² * x^2 + λ * x^4

struct Potential{T<:Real} <: ContinuousUnivariateDistribution
    m²::T
    λ::T
    xmin::T
    xmax::T
    denom::T
    function Potential{T}(;m²::Real, λ::Real, xmin::Real=-2, xmax::Real=2) where T<:Real
        (xmin < xmax) || error("must be xmin < xmax")
        Δ = T(0.001)
        x = xmin:Δ:xmax
        y = exp.(-v.(x, T(m²), T(λ)))
        denom = sum(y) * Δ
        new{T}(T(m²), T(λ), T(xmin), T(xmax), T(denom))
    end
end

params(d::Potential) = (d.m², d.λ) # just in case
partype(::Potential{T}) where {T<:Real} = T # Not sure how to use it, but just in case

Distributions.@distr_support Potential d.xmin d.xmax

v(d::Potential{T}, x::Real) where T = v(T(x), d.m², d.λ)

Base.eltype(::Type{Potential{T}}) where {T} = T

logpdf(d::Potential{T}, x::Real) where T = -v(d, x) - log(d.denom)
pdf(d::Potential{T}, x::Real) where T = exp(-v(d, x))/d.denom

function Random.rand(rng::AbstractRNG, v::Potential{T}) where T
    k = T(2)
    while true
        z = (v.xmax - v.xmin) * rand(rng, T) +  v.xmin
        u = k * rand(rng, T)
        if pdf(v, z) > u
            return z
        end
    end
end

end
```

```julia
using .My
Potential = My.Potential
```

```julia
d = Potential{Float32}(m²=-4.0, λ=8.0)
rng = MersenneTwister(0)
samples = rand(rng, d, 5*10^5)
@show samples[begin:begin+3]
@show samples |> typeof
histogram(samples, normalize=:pdf, label="histogram")
x = d.xmin:0.001:d.xmax |> collect
plot!(d)
```

```julia
d = Potential{Float32}(m²=-0.53, λ=8.0)
rng = MersenneTwister(0)
samples = rand(rng, d, 5*10^5)
@show samples[begin:begin+3]
@show samples |> typeof
histogram(samples, normalize=:pdf, label="histogram")

x = d.xmin:0.001:d.xmax |> collect
plot!(d)
```

```julia
logpdf.(d, 1), log(pdf(d, 1))
```

```julia
d = Potential{Float32}(m²=1.0, λ=8.0)
rng = MersenneTwister(0)
samples = rand(rng, d, 5*10^5)
@show samples[begin:begin+3]
@show samples |> typeof
histogram(samples, normalize=:pdf, label="histogram")

x = d.xmin:0.001:d.xmax |> collect
plot!(d)
```
