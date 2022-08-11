# Study Green function

In this notebook we would like to implement the connected twopoint Green’s function in accordance with [Flow-based generative models for Markov chain Monte Carlo in lattice field theory](https://arxiv.org/pdf/1904.12072.pdf).


$$
G_c(x) = \frac{1}{V}\sum_y \left( \langle\phi(y)\phi(y+x)\rangle - \langle\phi(y)\rangle\langle\phi(y+x)\rangle \right)
$$

```julia
using IterTools
using Statistics
```

````julia
"""
Green function \$G_c\$
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
?green
```

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
"""
Faster implementation for `green`
`size(cfgs)` should show space-time-batch size
"""
function green3(cfgs, offsetX, lattice_shape)
    shifts = (broadcast(-, offsetX)..., 0)
    batch_dim = ndims(cfgs)
    cfgs_offset = circshift(cfgs, shifts) # phi(y+x)
    m_corr = mean(cfgs .* cfgs_offset, dims=batch_dim)
    m = mean(cfgs, dims=batch_dim)
    m_offset = mean(cfgs_offset, dims=batch_dim)
    Gc = sum(m_corr .- m .* m_offset)/prod(lattice_shape)
    return Gc 
end
```

```julia
function green4(cfgs::AbstractArray{T, N}, offsetX) where {T, N}
    shifts = (broadcast(-, offsetX)..., 0)
    batch_dim = N
    lattice_shape = (size(cfgs, i) for i in 1:N-1)
    cfgs_offset = circshift(cfgs, shifts)
    m_corr = mean(cfgs .* cfgs_offset, dims=batch_dim)
    m = mean(cfgs, dims=batch_dim)
    m_offset = mean(cfgs_offset, dims=batch_dim)
    V = prod(lattice_shape)
    Gc = sum(m_corr .- m .* m_offset)/V
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
size(cfgs) : space-time-batch layout
"""
function mfGc(cfgs::AbstractArray{T, N}, t) where {T, N}
    space_shape = size(cfgs)[begin:N-2]
    #@show space_shape
    ret = zero(T)
    for s in IterTools.product((1:l for l in space_shape)...)
        #@show (s..., t)
        ret += green4(cfgs, (s..., t))
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
mfGc(cfgs, 1)
```

```julia
args = (cfgs, (2,3,4), lattice_shape)
@assert green(args...) ≈ green1(args...) ≈ green2(args...) ≈ green3(args...) ≈ green4(args[1:end-1]...)
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

```julia
@time green4(cfgs, (1,1,4))
```
