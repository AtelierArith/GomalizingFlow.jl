---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: julia 1.6.3
    language: julia
    name: julia-1.6
---

# Study Green function


$$
G_c(x) = \frac{1}{V}\sum_y \left( \langle\phi(y)\phi(y+x)\rangle - \langle\phi(y)\rangle\langle\phi(y+x)\rangle \right)
$$

```julia
using IterTools
using Statistics
```

````julia
"""
Green function
```math
G_c(x) = \\frac{1}{V}\\sum_y \\left( \\langle\\phi(y)\\phi(y+x)\\rangle - \\langle\\phi(y)\\rangle\\langle\\phi(y+x)\\rangle \\right)
```
"""
function green(cfgs, offsetX, lattice_shape)
    Gc = zero(Float32)
    for posY in IterTools.product((1:l for l in lattice_shape)...)
        phi_y = cfgs[posY..., :]
        shifts = (broadcast(-, offsetX)..., 0)
        phi_y_x = circshift(cfgs, shifts)[posY..., :]
        mean_phi_y = mean(phi_y)
        mean_phi_y_x = mean(phi_y_x)
        Gc += mean(phi_y .* phi_y_x) - mean_phi_y * mean_phi_y_x
    end
    Gc /= prod(lattice_shape)
    return Gc 
end
````

```julia
function green1(cfgs, offsetX, lattice_shape)
    Gc = zero(Float32)
    shifts = (broadcast(-, offsetX)..., 0)
    dest = similar(cfgs)
    for posY in IterTools.product((1:l for l in lattice_shape)...)
        phi_y = @view cfgs[CartesianIndex(posY), :]
        circshift!(dest, cfgs, shifts)
        phi_y_x = @view dest[CartesianIndex(posY), :]
        mean_phi_y = mean(phi_y)
        mean_phi_y_x = mean(phi_y_x)
        Gc += mean(phi_y .* phi_y_x) - mean_phi_y * mean_phi_y_x
    end
    Gc /= prod(lattice_shape)
    return Gc 
end
```

```julia
function green2(cfgs, offsetX::NTuple{3,Int}, lattice_shape::NTuple{3, Int})
    shifts = (broadcast(-, offsetX)..., 0)
    Gc = zero(Float32)
    dest = copy(cfgs)
    for (Y1, Y2, Y3) in IterTools.product((1:l for l in lattice_shape)...)
        phi_y = @view cfgs[Y1, Y2, Y3, :]
        circshift!(dest, cfgs, shifts)
        phi_y_x = @view dest[Y1, Y2, Y3, :]
        mean_phi_y = mean(phi_y)
        mean_phi_y_x = mean(phi_y_x)
        Gc += mean(phi_y .* phi_y_x) - mean_phi_y * mean_phi_y_x
    end
    Gc /= prod(lattice_shape)
    return Gc 
end
```

```julia
function green3(cfgs, offsetX, lattice_shape)
    shifts = (broadcast(-, offsetX)..., 0)
    batch_dim = ndims(cfgs)
    cfgs_offset = circshift(cfgs, shifts)
    m_corr = mean(cfgs .* cfgs_offset, dims=batch_dim)
    m = mean(cfgs, dims=batch_dim)
    m_offset = mean(cfgs_offset, dims=batch_dim)
    Gc = sum(m_corr .- m .* m_offset)/prod(lattice_shape)
    return Gc 
end
```

momentum-space representation

$$
\begin{aligned}
\tilde{G}_c(\vec{p}, t) 
&= 
\frac{1}{L^{d-1}}
\sum_{\vec{x}}
    e^{i \vec{p}\cdot\vec{x}} G_c(\vec{x}, t), \\
\mathrm{mfGc}
&= 
\tilde{G}_c(\vec{0}, t)
\end{aligned}
$$

$$
$$

```julia
"""
momentum free Green function
"""
function mfGc(cfgs, t, lattice_shape)
    space_shape = size(cfgs)[begin:length(lattice_shape)-1]
    @show space_shape
    ret = 0
    for s in IterTools.product((1:l for l in space_shape)...)
        @show (s..., t)
        ret += green(cfgs, (s..., t), lattice_shape)
    end
    ret /= prod(space_shape)
    return ret
end
```

```julia
L = 8
lattice_shape=(L, L, L)
cfgs = rand(Float32, L, L, L, 2000);
```

```julia
#mfGc(cfgs, 1, lattice_shape)
```

```julia
args = (cfgs, (2,3,4), lattice_shape)
@assert green(args...) ≈ green1(args...) ≈ green2(args...) ≈ green3(args...)
```

```julia
@time green(cfgs, (1,1,4), lattice_shape)
```

```julia
@time green1(cfgs, (1,1,4), lattice_shape)
```

```julia
@time green2(cfgs, (1,1,4), lattice_shape)
```

```julia
@time green3(cfgs, (1,1,4), lattice_shape)
```
