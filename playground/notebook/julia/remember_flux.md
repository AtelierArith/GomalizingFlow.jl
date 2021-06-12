---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Julia 1.6.1
    language: julia
    name: julia-1.6
---

```julia
using Distributions
using Flux
using PyPlot
```

```julia
model = Chain(
    Dense(rand(Normal(1, 2), 8, 1), -ones(8), relu),
    Dense(rand(Normal(1, 2), 8, 8), -ones(8), relu),
    Dense(rand(Normal(1, 2), 1, 8), -ones(1), Flux.tanh),
)
```

```julia
function forward(x₁, x₂)
    s = model(x₂)
    logJ = s
    fx₁ = @. exp(s) * x₁
    fx₂ = x₂
    fx₁, fx₂, logJ
end
```

```julia
batchsize = 1024
u = Uniform(-1, 1)
x₁ = rand(u, 1, batchsize)
x₂ = rand(u, 1, batchsize)
gx₁, gx₂, fwd_logJ = forward(x₁, x₂);
```

```julia
function reverse(fx₁, fx₂)
    x₂ = fx₂
    s = model(x₂)
    logJ = -s
    x₁ = @. exp(-s) * fx₁
    return x₁, x₂, logJ
end
```

```julia
xp₁, xp₂, bwd_logj = reverse(gx₁, gx₂)
```

```julia
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
ax1.scatter(x₁, x₂); ax1.set_xlim([-1.1, 1.1]), ax1.set_ylim([-1.1, 1.1])
ax2.scatter(gx₁, gx₂); ax2.set_xlim([-1.1, 1.1]), ax2.set_ylim([-1.1, 1.1])
ax3.scatter(xp₁, xp₂); ax3.set_xlim([-1.1, 1.1]), ax3.set_ylim([-1.1, 1.1])
plt.show()
```
