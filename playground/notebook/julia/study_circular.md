---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.12.0
  kernelspec:
    display_name: Julia 1.6.3
    language: julia
    name: julia-1.6
---

```julia
using ImageFiltering
```

```julia
"""
Differential padarray for 2D
"""
function mycircular(Y::AbstractArray{<:Real,2 + 2})
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

"""
Differential padarray for 3D
"""
function mycircular(Y::AbstractArray{<:Real,3 + 2})
    # calc Z_bottom
    Y_b_c = Y[begin:begin,:,:,:,:]
    Y_b_r = Y[begin:begin,end:end,:,:,:]
    Y_b_l = Y[begin:begin,begin:begin,:,:,:]
    Z_bottom = cat(Y_b_r, Y_b_c, Y_b_l, dims=2) # calc pad under

    # calc Z_top
    Y_e_c = Y[end:end,:,:,:,:]
    Y_e_r = Y[end:end,end:end,:,:,:]
    Y_e_l = Y[end:end,begin:begin,:,:,:]
    Z_top = cat(Y_e_r, Y_e_c, Y_e_l, dims=2)

    # calc Z_main
    Y_main_l = Y[:,begin:begin,:,:,:]
    Y_main_r = Y[:,end:end,:,:,:]
    Z_main = cat(Y_main_r, Y, Y_main_l, dims=2)
    Z_ = cat(Z_top, Z_main, Z_bottom, dims=1)

    # pad along 3rd axis
    Z_begin = Z_[:,:, begin:begin,:,:]
    Z_end = Z_[:,:, end:end,:,:]
    cat(Z_end, Z_, Z_begin, dims=3)
end
```

```julia
"""
Differential padarray for 4D
"""
function mycircular(Y::AbstractArray{<:Real,4 + 2})
    # calc Z_bottom
    Y_b_c = Y[begin:begin,:,:,:,:,:]
    Y_b_r = Y[begin:begin,end:end,:,:,:,:]
    Y_b_l = Y[begin:begin,begin:begin,:,:,:,:]
    Z_bottom = cat(Y_b_r, Y_b_c, Y_b_l, dims=2) # calc pad under

    # calc Z_top
    Y_e_c = Y[end:end,:,:,:,:,:]
    Y_e_r = Y[end:end,end:end,:,:,:,:]
    Y_e_l = Y[end:end,begin:begin,:,:,:,:]
    Z_top = cat(Y_e_r, Y_e_c, Y_e_l, dims=2)

    # calc Z_main
    Y_main_l = Y[:,begin:begin,:,:,:,:]
    Y_main_r = Y[:,end:end,:,:,:,:]
    Z_main = cat(Y_main_r, Y, Y_main_l, dims=2)
    Z_3rd = cat(Z_top, Z_main, Z_bottom, dims=1)

    # pad along 3rd axis
    Z_3rd_begin = Z_3rd[:,:, begin:begin,:,:,:]
    Z_3rd_end = Z_3rd[:,:, end:end,:,:,:]
    Z_ = cat(Z_3rd_end, Z_3rd, Z_3rd_begin, dims=3)
    
    # pad along 4th axis
    Z_begin = Z_[:,:,:,begin:begin,:,:]
    Z_end = Z_[:,:,:,end:end,:,:]
    return cat(Z_end, Z_, Z_begin, dims=4)
end
```

```julia
x = rand(4,4,4,4,4,4)
tar = mycircular(x)
ref = ImageFiltering.padarray(x, Pad(:circular, 1, 1, 1, 1, 0, 0)).parent
tar ≈ ref
```
