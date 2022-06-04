module PotentialDistributions

import Base: minimum, maximum # work with @distr_spport

using Distributions
import Distributions: pdf, logpdf, @distr_support
using Random

export Potential

v(x::T, m²::T, λ::T) where {T<:Real} = m² * x^2 + λ * x^4

struct Potential{T<:Real} <: ContinuousUnivariateDistribution
    m²::T
    λ::T
    xmin::T
    xmax::T
    denom::T
    function Potential{T}(;
        m²::Real,
        λ::Real,
        xmin::Real=-2,
        xmax::Real=2,
    ) where {T<:Real}
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

v(d::Potential{T}, x::Real) where {T} = v(T(x), d.m², d.λ)

Base.eltype(::Type{Potential{T}}) where {T} = T

logpdf(d::Potential{T}, x::Real) where {T} = -v(d, x) - log(d.denom)
pdf(d::Potential{T}, x::Real) where {T} = exp(-v(d, x)) / d.denom

function Random.rand(rng::AbstractRNG, v::Potential{T}) where {T}
    k = T(2)
    while true
        z = (v.xmax - v.xmin) * rand(rng, T) + v.xmin
        u = k * rand(rng, T)
        if pdf(v, z) > u
            return z
        end
    end
end

end # module
