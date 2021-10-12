reversedims(inp::AbstractArray{<:Any,N}) where {N} = permutedims(inp, N:-1:1)

function pairwise(iterable)
    b = deepcopy(iterable)
    popfirst!(b)
    a = iterable
    return zip(a, b)
end

"""
2D implementation
"""
function make_checker_mask(shape::NTuple{2,Int}, parity)
    checker = ones(Int, shape) .- parity
    checker[begin:2:end, begin:2:end] .= parity
    checker[(begin + 1):2:end, (begin + 1):2:end] .= parity
    return checker
end

"""
3D implementation
"""
function make_checker_mask(shape::NTuple{3,Int}, parity)
    checker = ones(Int, shape) .- parity
    checker[begin:2:end, begin:2:end, begin:2:end] .= parity
    checker[(begin + 1):2:end, (begin + 1):2:end, begin:2:end] .= parity
    
    checker[(begin + 1):2:end, begin:2:end, (begin + 1):2:end] .= parity
    checker[begin:2:end, (begin + 1):2:end, (begin + 1):2:end] .= parity

    return checker
end

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