# Circular 実装

周期境界条件を考慮するために畳み込み演算で circular mode のパディングを実行する必要がある. Julia では `ImageFiltering` が提供する `padarray` で実現できるが内部で `setindex` を用いているため `Zygote.jl` による自動微分が実行できない. そこで `ImageFiltering.padarray` と同じ機能を `cat` 関数によって実現する.

```julia
using Flux
using ImageFiltering
```

```julia
X = reshape(collect(Float32, 0:49-1), 7, 7)'
X = reshape(X, 7, 7, 1, 1) |> f32
```

```julia
function mycircular(Y::AbstractArray{<:Real,2 + 2}, d1=1, d2=1)
    Y_top_center = Y[begin:begin+(d1-1), :, :, :]
    Y_top_right = Y[begin:begin+(d1-1), end-(d2-1):end, :, :]
    Y_top_left = Y[begin:begin+(d1-1), begin:begin+(d2-1), :, :]
    Z_bottom = cat(Y_top_right, Y_top_center, Y_top_left, dims=2) # calc pad under

    Y_bottom_center = Y[end-(d1-1):end, :, :, :]
    Y_bottom_right = Y[end-(d1-1):end, end-(d2-1):end, :, :]
    Y_bottom_left = Y[end-(d1-1):end, begin:begin+(d2-1), :, :]
    Z_top = cat(Y_bottom_right, Y_bottom_center, Y_bottom_left, dims=2) # calc pad under

    Y_main_left = Y[:,begin:begin+(d2-1),:,:]
    Y_main_right = Y[:,end-(d2-1):end,:,:]
    Z_main = cat(Y_main_right, Y, Y_main_left, dims=2)
    return cat(Z_top, Z_main, Z_bottom, dims=1)
end
```

```julia
m1 = Chain(
    mycircular,
    Conv((3,3), 1=>2,  stride=1, pad=0, bias=false, init=ones)
) |> f32
```

```julia
m2 = Chain(
    x -> padarray(x, Pad(:circular,1, 1, 0, 0)).parent,
    Conv((3,3), 1=>2,  stride=1, pad=0, bias=false, init=ones)
) |> f32

@assert m2(X) == m1(X)
```

```julia
m3 = Chain(
    x -> padarray(x, Pad(:circular,2, 3, 0, 0)).parent,
    Conv((3,3), 1=>2,  stride=1, pad=0, bias=false, init=ones)
) |> f32

m4 = Chain(
    x -> mycircular(x, 2, 3),
    Conv((3,3), 1=>2,  stride=1, pad=0, bias=false, init=ones)
) |> f32

@assert m3(X) == m4(X)
```

```julia
function mycircular(Y::AbstractArray{<:Real,3 + 2}, d1=1, d2=1, d3=1)
    Y_top_center = Y[begin:begin+(d1-1), :, :, :, :]
    Y_top_right = Y[begin:begin+(d1-1), end-(d2-1):end, :, :, :]
    Y_top_left = Y[begin:begin+(d1-1), begin:begin+(d2-1), :, :, :]
    Z_bottom = cat(Y_top_right, Y_top_center, Y_top_left, dims=2) # calc pad under

    Y_bottom_center = Y[end-(d1-1):end, :, :, :, :]
    Y_bottom_right = Y[end-(d1-1):end, end-(d2-1):end, :, :, :]
    Y_bottom_left = Y[end-(d1-1):end, begin:begin+(d2-1), :, :, :]
    Z_top = cat(Y_bottom_right, Y_bottom_center, Y_bottom_left, dims=2) # calc pad under

    Y_main_left = Y[:,begin:begin+(d2-1),:,:,:]
    Y_main_right = Y[:,end-(d2-1):end,:,:,:]
    Z_main = cat(Y_main_right, Y, Y_main_left, dims=2)
    Z_ = cat(Z_top, Z_main, Z_bottom, dims=1)
    
    # pad along 3rd axis
    Z_begin = Z_[:, :, begin:begin+(d3-1), :, :]
    Z_end = Z_[:, :, end-(d3-1):end, :, :]
    return cat(Z_end, Z_, Z_begin, dims=3)
end
```

```julia
X = reshape(collect(Float32, 0:8^3-1), 8,8,8)
X = reshape(X, 8,8,8, 1, 1) |> f32

d1 = 2
d2 = 3
d3 = 4

d31 = Chain(
    x->mycircular(x, d1, d2, d3),
    Conv((3,3,3), 1=>2,  stride=1, pad=0, bias=false, init=ones)
) |> f32

d32 = Chain(
    x -> padarray(x, Pad(:circular,d1, d2 , d3, 0, 0)).parent,
    Conv((3,3,3), 1=>2,  stride=1, pad=0, bias=false, init=ones)
) |> f32;
```

```julia
@assert d31(X) ≈ d32(X)
```

```julia
function mycircular(Y::AbstractArray{<:Real,4 + 2}, d1=1, d2=1,d3=1,d4=1)
    Y_top_center = Y[begin:begin+(d1-1), :, :, :, :, :]
    Y_top_right = Y[begin:begin+(d1-1), end-(d2-1):end, :, :, :, :]
    Y_top_left = Y[begin:begin+(d1-1), begin:begin+(d2-1), :, :, :, :]
    Z_bottom = cat(Y_top_right, Y_top_center, Y_top_left, dims=2) # calc pad under

    Y_bottom_center = Y[end-(d1-1):end, :, :, :, :, :]
    Y_bottom_right = Y[end-(d1-1):end, end-(d2-1):end, :, :, :, :]
    Y_bottom_left = Y[end-(d1-1):end, begin:begin+(d2-1), :, :, :, :]
    Z_top = cat(Y_bottom_right, Y_bottom_center, Y_bottom_left, dims=2) # calc pad under

    Y_main_left = Y[:,begin:begin+(d2-1),:,:,:,:]
    Y_main_right = Y[:,end-(d2-1):end,:,:,:,:]
    Z_main = cat(Y_main_right, Y, Y_main_left, dims=2)
    Z_3rd = cat(Z_top, Z_main, Z_bottom, dims=1)
    
    # pad along 3rd axis
    Z_3rd_begin = Z_3rd[:, :, begin:begin+(d3-1), :, :, :]
    Z_3rd_end = Z_3rd[:, :, end-(d3-1):end, :, :, :]
    Z_ = cat(Z_3rd_end, Z_3rd, Z_3rd_begin, dims=3)

    # pad along 4th axis
    Z_begin = Z_[:, :, :, begin:begin+(d4-1), :, :]
    Z_end = Z_[:, :, :, end-(d4-1):end, :, :]
    return cat(Z_end, Z_, Z_begin, dims=4)
end
```

```julia
X = reshape(collect(Float32, 0:5^4-1), 5,5,5,5)
X = reshape(X, 5,5,5,5, 1, 1) |> f32

d1 = 2
d2 = 3
d3 = 4
d4 = 5

r41 = mycircular(X, d1, d2, d3,d4)
r42 = padarray(X, Pad(:circular,d1, d2 , d3, d4,0, 0)).parent

@assert r41 == r42
```
