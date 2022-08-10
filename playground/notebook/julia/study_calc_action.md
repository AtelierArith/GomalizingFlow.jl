```julia
using PyCall
using Flux
```

```julia
py"""
import numpy as np
def create_cfgs(L):
    rng = np.random.default_rng(2021)
    lattice_shape=(L, L)
    phi_ex1 = rng.normal(size=(lattice_shape)).astype(np.float32)
    phi_ex2 = rng.normal(size=(lattice_shape)).astype(np.float32)
    cfgs = np.stack((phi_ex1, phi_ex2), axis=0)
    return cfgs
"""
```

```julia
L = 8
lattice_shape = (L, L)

m² = -4.0
λ = 8.0
```

```julia
reversedims(inp::AbstractArray{<:Any, N}) where {N} = permutedims(inp, N:-1:1)
```

```julia
struct ScalarPhi4Action
    m²::Float32
    λ::Float32
end 
```

```julia
"""
cfgs |> size == (N, H, W)
"""
function py_calc_action(action::ScalarPhi4Action, cfgs)
    action_density = @. action.m² * cfgs ^ 2 + action.λ * cfgs ^ 4
    Nd = lattice_shape |> length
    term1 = sum(2cfgs .^ 2 for μ in 2:Nd+1)
    term2 = sum(cfgs .* circshift(cfgs, Flux.onehot(μ, 1:(Nd+1))) for μ in 2:(Nd+1))
    term3 = sum(cfgs .* circshift(cfgs, -Flux.onehot(μ, 1:(Nd+1))) for μ in 2:(Nd+1))
    result = action_density .+ term1 .- term2 .- term3
    dropdims(
        sum(result, dims=2:Nd+1),
        dims=Tuple(2:(Nd+1))
    )
end

function (action::ScalarPhi4Action)(cfgs)
    py_calc_action(action, cfgs)
end

cfgs = py"create_cfgs"(L)
@assert ScalarPhi4Action(1, 1)(cfgs) ≈ [499.7602, 498.5477]
phi4_action = ScalarPhi4Action(m², λ)
@assert phi4_action(cfgs) ≈ [1598.679, 1545.5698]
```

\begin{equation}
\begin{split}
S^E_{\text{cont}}[\phi] &= \int d^2\vec{x} ~ (\partial_\mu \phi(\vec{x}))^2 + m^2 \phi(\vec{x})^2 + \lambda \phi(\vec{x})^4 \\
\rightarrow S^E_{\text{latt}}(\phi) &= \sum_{\vec{n}} \phi(\vec{n}) \left[ \sum_{\mu \in \{1,2\}} 2\phi(\vec{n}) - \phi(\vec{n}+\hat{\mu}) - \phi(\vec{n}-\hat{\mu}) \right] + m^2 \phi(\vec{n})^2 + \lambda \phi(\vec{n})^4
\end{split}
\end{equation}


\begin{align}
\textrm{k1} &= \sum_{\vec{n}} \sum_{\mu \in \{1, 2\}} 2\phi(\vec{n}) ^ 2 \\
\textrm{k2} &= \sum_{\vec{n}} \sum_{\mu \in \{1, 2\}} \phi(\vec{n}) \phi(\vec{n} + \hat{\mu}) \\
\textrm{k3} &= \sum_{\vec{n}} \sum_{\mu \in \{1, 2\}} \phi(\vec{n}) \phi(\vec{n} - \hat{\mu}) \\
\end{align}

```julia
"""
cfgs |> size == (W, H, N)
"""
function jl_calc_action(action::ScalarPhi4Action, cfgs)
    potential = @. action.m² * cfgs ^ 2 + action.λ * cfgs ^ 4
    sz = length(size(cfgs))
    Nd = sz - 1 # exclude last axis
    k1 = sum(2cfgs .^ 2 for μ in 1:Nd)
    k2 = sum(cfgs .* circshift(cfgs, -Flux.onehot(μ, 1:sz)) for μ in 1:Nd)
    k3 = sum(cfgs .* circshift(cfgs, Flux.onehot(μ, 1:sz)) for μ in 1:Nd)
    action_density = potential .+ k1 .- k2 .- k3
    dropdims(
        sum(action_density, dims=1:Nd),
        dims=Tuple(1:Nd)
    )
end

function (action::ScalarPhi4Action)(cfgs)
    jl_calc_action(action, cfgs)
end

cfgs = py"create_cfgs"(L) |> reversedims
ScalarPhi4Action(1, 1)(cfgs)
@assert ScalarPhi4Action(1, 1)(cfgs) ≈ [499.7602, 498.5477]
phi4_action = ScalarPhi4Action(m², λ)
@assert phi4_action(cfgs) ≈ [1598.679, 1545.5698]
```

```julia

```
