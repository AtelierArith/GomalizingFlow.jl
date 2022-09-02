```julia
using Flux
```

```julia
"""
Differentiable padarray for 1D
"""
function mycircular(Y::AbstractArray{<:Real,1 + 2}, d1::Int=1)
    Yl = Y[begin:begin+(d1-1), :, :]
    Yr = Y[end-(d1-1):end, :, :]
    cat(Yr, Y, Yl, dims=1)
end

function mycircular(Y::AbstractArray{<:Real,1 + 2}, ds::NTuple{1,Int})
    mycircular(Y, ds[1])
end

"""
Differentiable padarray for 2D
"""
function mycircular(Y::AbstractArray{<:Real,2 + 2}, d1=1, d2=1)
    Y_top_center = Y[begin:begin+(d1-1), :, :, :]
    Y_top_right = Y[begin:begin+(d1-1), end-(d2-1):end, :, :]
    Y_top_left = Y[begin:begin+(d1-1), begin:begin+(d2-1), :, :]
    Z_bottom = cat(Y_top_right, Y_top_center, Y_top_left, dims=2) # calc pad under

    Y_bottom_center = Y[end-(d1-1):end, :, :, :]
    Y_bottom_right = Y[end-(d1-1):end, end-(d2-1):end, :, :]
    Y_bottom_left = Y[end-(d1-1):end, begin:begin+(d2-1), :, :]
    Z_top = cat(Y_bottom_right, Y_bottom_center, Y_bottom_left, dims=2) # calc pad under

    Y_main_left = Y[:, begin:begin+(d2-1), :, :]
    Y_main_right = Y[:, end-(d2-1):end, :, :]
    Z_main = cat(Y_main_right, Y, Y_main_left, dims=2)
    return cat(Z_top, Z_main, Z_bottom, dims=1)
end

function mycircular(Y::AbstractArray{<:Real,2 + 2}, ds::NTuple{2,Int})
    mycircular(Y, ds[1], ds[2])
end
```

```julia
W = rand(Float32, 3, 1, 2, 3);
```

```julia
c21 = Chain(
    Base.Fix2(mycircular, (1, 0)),
    Conv(W, zeros(Float32, 3))
)
```

```julia
c1 = Chain(
    Base.Fix2(mycircular, (1,)),
    Conv(dropdims(W, dims=2), zeros(Float32, 3))
)
```

```julia
N = prod([5, 5, 2, 2])
x = reshape(1:N, 5, 5, 2, 2) |> f32
```

```julia
y21 = c21(x);
```

```julia
x1 = permutedims(x, (1, 3, 2, 4))
x2 = reshape(x1, 5, 2, Colon())
y = c1(x2)
y |> size == (5, 3, 5 * 2)
y1 = reshape(y, 5, 3, 5, 2)
y2 = permutedims(y1, (1, 3, 2, 4))
@assert y21 ≈ y2
```
