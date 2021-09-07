---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.4
  kernelspec:
    display_name: Julia 1.6.2
    language: julia
    name: julia-1.6
---

# Circular 実装

周期境界条件を考慮するために畳み込み演算で circular mode のパディングを実行する必要がある. Julia では `ImageFiltering` が提供する `padarray` で実現できるが内部で `setindex` を用いているため `Zygote.jl` による自動微分が実行できない. そこで `ImageFiltering.padarray` と同じ機能を `cat` 関数によって実現する.

```julia
using Flux
using ImageFiltering
```

```julia
X = reshape(collect(Float32, 0:25-1), 5, 5)'
X = reshape(X, 5, 5, 1, 1) |> f32
```

```julia
function mycircular(Y)
    # calc Z_bottom
    Y_b_c = Y[begin:begin,:,:,:]
    Y_b_r = Y[begin:begin,end:end,:,:]
    Y_b_l = Y[begin:begin,begin:begin,:,:]
    Z_bottom = cat(Y_b_r, Y_b_c, Y_b_l, dims=2) # calc pad under
    
    # calc Z_top
    Y_e_c = Y[end:end,:,:,:]
    Y_e_r = Y[end:end,end:end,:,:]
    Y_e_l = Y[end:end,begin:begin,:,:]
    Z_top = cat(Y_e_r, Y_e_c, Y_e_l, dims=2)
    
    # calc Z_main
    Y_main_l = Y[:,begin:begin,:,:]
    Y_main_r = Y[:,end:end,:,:]
    Z_main = cat(Y_main_r, Y, Y_main_l, dims=2)
    cat(Z_top, Z_main, Z_bottom, dims=1)
end
```

```julia
m1 = Chain(
    mycircular,
    Conv((3,3), 1=>2,  stride=1, pad=0, bias=false, init=ones)
) |> f32

m1(X)
```

```julia
m2 = Chain(
    x -> padarray(x, Pad(:circular,1, 1, 0, 0)),
    Conv((3,3), 1=>2,  stride=1, pad=0, bias=false, init=ones)
) |> f32

m2(X)
```

```julia
@assert m1(X) == m2(X)
```
