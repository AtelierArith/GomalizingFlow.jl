---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.12.0
  kernelspec:
    display_name: Julia 1.6.2
    language: julia
    name: julia-1.6
---

# Ising 2D

```julia
using Random

using IterTools
using Plots
```

```julia
function E(C)
    H, W = size(C)
    hamiltonian = zero(eltype(C))
    for (iy, ix) in IterTools.product(1:H, 1:W)
        iy_b = ifelse(iy == 1, H, iy - 1)
        iy_t = ifelse(iy == H, 1, iy + 1)
        ix_l = ifelse(ix == 1, W, ix - 1)
        ix_r = ifelse(ix == W, 1, ix + 1)
        s_i = C[iy, ix]
        hamiltonian -= s_i * (C[iy, ix_l] + C[iy, ix_r] + C[iy_b, ix] + C[iy_t, ix])
    end
    hamiltonian
end
```

```julia
function metropolis(L; β=log(1 + √2)/2, seed=12345)
    lattice_shape = (L, L)
    rng = Random.default_rng()
    Random.seed!(rng, seed)
    isdefined(Main, :IJulia) && IJulia.clear_output(true)
    C = rand(rng, (-Int8(1), Int8(1)), lattice_shape...)
    p = heatmap(
                C, size=(300, 300), 
                title="init", 
                aspectratio=:equal,
                colorbar=false,
                ticks=false,
                axes=false,
            )
    p |> display
    C_cand = copy(C)
    H, W = lattice_shape
    Cs = [C]
    for iter in 1:100
        isdefined(Main, :IJulia) && IJulia.clear_output(true)
        for lsz in 1:prod(lattice_shape)
            iy = rand(1:H)
            ix = rand(1:W)
            C_cand[iy, ix] = -C[iy, ix]
            ΔE = E(C_cand) - E(C)
            r = rand(rng)
            if r < exp(-β*ΔE)
                C[iy, ix] = C_cand[iy, ix]
            else
                # reset state
                C_cand[iy, ix] = C[iy, ix]
            end
        end
        p = heatmap(
                C, size=(300, 300), 
                title="$iter", 
                aspectratio=:equal,
                colorbar=false,
                ticks=false,
                axes=false,
            )
        display(p)
    end
    return Cs
end
```

```julia
L = 2^6
enumerate(metropolis(L))
```
