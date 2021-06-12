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
using PyPlot
```

# Box-Muller transform

```julia
B = 2^14

U₁ = rand(B)
U₂ = rand(B)
Z₁ = @. √(-2log(U₁)) * cos(2π * U₂)
Z₂ = @. √(-2log(U₁)) * sin(2π * U₂);
```

```julia
fig, ax = plt.subplots()
ax.hist2d(
    Z₁, Z₂, 
    bins=30, 
    range=[[-3, 3], [-3, 3]]
)
ax.set_aspect("equal")
```

```julia

```

```julia

```
