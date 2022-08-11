using Base.Threads
using ProgressMeter
using Flux
using IterTools

"""
    green(cfgs::AbstractArray{T, N}, offsetX) where {T, N}
Green function \$G_c\$
```math
G_c(x) = \\frac{1}{V}\\sum_y \\left( \\langle\\phi(y)\\phi(y+x)\\rangle - \\langle\\phi(y)\\rangle\\langle\\phi(y+x)\\rangle \\right)
```
"""
function green(cfgs::AbstractArray{T, N}, offsetX) where {T, N}
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

"""
    mfGc(cfgs::AbstractArray{T, N}, t) where {T, N}
Compute momentum free Green function
cfgs: configurations with batchsize `size(cfgs, N)`. The rest of size(cfgs)[1:N-1] orders space and time space, say,
size(cfgs) consists of space-time-batch
"""
function mfGc(cfgs::AbstractArray{T, N}, t) where {T, N}
    space_shape = size(cfgs)[begin:N-2]
    acc = Atomic{T}(0)
    @threads for s in IterTools.product((1:l for l in space_shape)...) |> collect
        Threads.atomic_add!(acc, green(cfgs, (s..., t)))
    end
    return acc.value /= prod(space_shape)
end
