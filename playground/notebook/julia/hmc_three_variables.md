# HMC for 3-dimensional data

```julia
using LinearAlgebra

using Distributions
using StatsPlots
using Zygote
```

```julia
module My

using LinearAlgebra
using SymEngine

@vars x y z

const A = [
    1 1 1
    1 2 1
    1 1 2
]

q = dot([x,y,z], A, [x, y, z])/2 |> expand
@show q

@show diff(q, x)
@show diff(q, y)
@show diff(q, z)

end # module

using .My

const A = My.A
```

```julia
S(xyz::AbstractVector) = dot(xyz, A, xyz)/2
S(x, y, z) = S([x, y, z])
∂S(xyz) = gradient(S, xyz...)
```

```julia
@assert gradient(S, 3,4,5)[1] == (2*3 + 2*4 + 2*5)/2
@assert gradient(S, 3,4,5)[2] == (2*3 + 4*4 + 2*5)/2
@assert gradient(S, 3,4,5)[3] == (2*3 + 2*4 + 4*5)/2
```

```julia
function hamiltonian(S, x, p)
    Σ_p² = dot(p, p)
    H = 0.5Σ_p² + S(x)
    return H
end
```

```julia
# Solve Molecular Dynamics
# a.k.a leapfrog algorithm 
function md(S, x, p)
    Nmd = 20
    Δτ = 0.5

    @. x += Δτ/2 * p
    for _ in 1:(Nmd-1)
        ∇, = gradient(S, x)
        @. p -= Δτ * ∇
        @. x += Δτ * p
    end
    ∇, = gradient(S, x)
    @. p -= Δτ * ∇
    @. x += Δτ/2 * p
    return x, p
end
```

```julia
function hmc(S::Function, x)
    p = randn(size(x)...)

    x_init = copy(x)
    p_init = copy(p)
    H_init = hamiltonian(S, x_init, p_init)

    x_cand, p_cand = md(S, x, p)
    H_cand = hamiltonian(S, x_cand, p_cand)

    ΔH = H_cand - H_init
    r = rand()
    accepted, x_next = (r < exp(-ΔH)) ? (true, x_cand) : (false, x_init)
    #@show accepted, x_cand, x_init
    return (accepted, x_next)
end
```

```julia
K = 10000
cfgs = rand(3)
T = Float64
history = (accepted=eltype(cfgs)[], cfgs=typeof(cfgs)[])

for _ in 1:K
    accepted, cfgs = hmc(S, cfgs)
    push!(history.accepted, accepted)
    push!(history.cfgs, cfgs)
end
```

```julia
mean(history.accepted)
```

```julia
x = getindex.(history.cfgs, 1)
y = getindex.(history.cfgs, 2)
z = getindex.(history.cfgs, 3)
corrplot(hcat(x, y, z))
```

```julia
scatter(x, y)
```
