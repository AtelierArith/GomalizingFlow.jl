---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.3
  kernelspec:
    display_name: Julia 1.6.1
    language: julia
    name: julia-1.6
---

Application 1: $\varphi^4$ lattice scalar field theory


## Physical theory

```julia
using Distributions
using LaTeXStrings
using PyPlot
```

```julia
L = 8
```

```julia
lattice_shape = (L, L)
```

- Python 実装の方では 0 番目の軸が batch を表す次元を担う. Julia の場合だと 一番最後の軸が batch の次元を担う. そのほうが print debug するときに都合が良い.


```julia
B = 2
φ₁ = rand(Normal(0, 1), lattice_shape)
φ₂ = rand(Normal(0, 1), lattice_shape)
cfgs = cat(reshape(φ₁, L, L, 1), reshape(φ₂,  L, L, 1), dims=length(lattice_shape)+1);
```

```julia
@assert φ₁ == cfgs[:,:,1]
@assert φ₂ == cfgs[:,:,2]
# @assert φᵢ == cgs[:,:, i] in general
```

```julia
struct ScalarPhi4Action
    m²
    λ
end

m² = -4.0
λ = 8.0
phi4_action = ScalarPhi4Action(m², λ)
```

```julia
function calc_action(sp4a::ScalarPhi4Action, cfgs)
    action_density = sp4a.m² * cfgs .^ 2 + sp4a.λ * cfgs .^ 4
    Nd = lattice_shape |> length
    for μ ∈ 1:Nd
        action_density += 2cfgs .^ 2

        shifts_plus = zeros(Nd+1)
        shifts_plus[μ] = 1 # \vec{n} + \hat{\mu}
        action_density -= circshift(cfgs, shifts_plus)

        shifts_minus = zeros(Nd+1)
        shifts_minus[μ] = -1 # \vec{n} - \hat{\mu}
        action_density -= circshift(cfgs, shifts_minus)
    end
    return sum(action_density, dims=1:Nd)
end
```

```julia
@assert action_density |> size == (L, L, B)
```

```julia
φ⁴_action = sum(action_density, dims=(1,2))
```

## Prior Distribution

```julia
prior = Normal(0, 1)
batch_size = 1024
z = rand(prior, (lattice_shape..., batch_size))
fig, ax = plt.subplots(4,4, dpi=125, figsize=(4,4))
for i in 1:4
    for j in 1:4
        ind = i*4 + j
        ax[i,j].imshow(tanh(z[:, :, ind]), vmin=-1, vmax=1, cmap="viridis")
        ax[i,j].axes.xaxis.set_visible(false)
        ax[i,j].axes.yaxis.set_visible(false)
    end
end
```

```julia
x1 = 1
latexstring("$x1")
```

```julia
fig, ax = plt.subplots(4, 4, dpi=125, figsize=(4,4))
for x1 in 1:Nd
    for y1 in 1:Nd
        i1 = (x1-1)*2 + y1
        for x2 in 1:Nd
            for y2 in 1:Nd
                i2 = (x2 -1)* 2 + y2
                ax[i1, i2].hist2d(z[x1,y1,:], z[x2,y2,:], range=[[-3,3],[-3,3]], bins=20)
                ax[i1, i2].set_xticks([])
                ax[i1, i2].set_yticks([])
                if i1 == 4
                    ax[i1, i2].set_xlabel(latexstring("\\phi($x2,$y2)"))
                end
                if i2 == 1
                    ax[i1, i2].set_ylabel(latexstring("\\phi($x1,$y1)"))
                end
            end
        end
    end
end
```

```julia
S_eff = -sum(logpdf(prior, z), dims=1:length(lattice_shape))
S = calc_action(phi4_action, z)
fit_b = mean(S) - mean(S_eff)
print("slope 1 linear regression S = -logr + $fit_b")
fig, ax = plt.subplots(1,1, dpi=125, figsize=(4,4))
ax.hist2d(vec(S_eff), vec(S), bins=20, range=[[-800, 800], [200,1800]])
xs = range(-800, stop=800, length=4)
ax.plot(xs, xs .+ fit_b, ":", color=:w, label="slope 1 fit")
ax.set_xlabel(L"S_{\mathrm{eff}} \equiv -\log ~ r(z)")
ax.set_ylabel(L"S(z)")
ax.set_aspect(:equal)
plt.legend(prop=Dict("size"=> 6))
plt.show()
```
