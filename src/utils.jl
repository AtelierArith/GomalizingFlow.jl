using BSON

function restore(r::AbstractString; load_best_ess=false)
    if load_best_ess
        BSON.@load joinpath(r, "history_best_ess.bson") history_best_ess
        BSON.@load joinpath(r, "trained_model_best_ess.bson") trained_model_best_ess
        return Flux.testmode!(trained_model_best_ess), history_best_ess
    else
        BSON.@load joinpath(r, "history.bson") history
        BSON.@load joinpath(r, "trained_model.bson") trained_model
        return Flux.testmode!(trained_model), history
    end
end

reversedims(inp::AbstractArray{<:Any,N}) where {N} = permutedims(inp, N:-1:1)

function pairwise(iterable)
    b = deepcopy(iterable)
    popfirst!(b)
    a = iterable
    return zip(a, b)
end

function make_checker_mask(shape::NTuple{N,Int}, parity::Int) where {N}
    N == 1 && (return make_checker_mask(shape[begin], parity))
    seq = map(1:last(shape)) do i
        p = ifelse(isodd(i), parity, -parity + 1)
        baseline = make_checker_mask(shape[begin:end-1], p)
    end
    return cat(seq..., dims=N)
end

function make_checker_mask(L::Int, parity)
    checker = ones(Int, L) .- parity
    checker[begin:2:end] .= parity
    return checker
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

"""
Differentiable padarray for 3D
"""
function mycircular(Y::AbstractArray{<:Real,3 + 2}, d1=1, d2=1, d3=1)
    Y_top_center = Y[begin:begin+(d1-1), :, :, :, :]
    Y_top_right = Y[begin:begin+(d1-1), end-(d2-1):end, :, :, :]
    Y_top_left = Y[begin:begin+(d1-1), begin:begin+(d2-1), :, :, :]
    Z_bottom = cat(Y_top_right, Y_top_center, Y_top_left, dims=2) # calc pad under

    Y_bottom_center = Y[end-(d1-1):end, :, :, :, :]
    Y_bottom_right = Y[end-(d1-1):end, end-(d2-1):end, :, :, :]
    Y_bottom_left = Y[end-(d1-1):end, begin:begin+(d2-1), :, :, :]
    Z_top = cat(Y_bottom_right, Y_bottom_center, Y_bottom_left, dims=2) # calc pad under

    Y_main_left = Y[:, begin:begin+(d2-1), :, :, :]
    Y_main_right = Y[:, end-(d2-1):end, :, :, :]
    Z_main = cat(Y_main_right, Y, Y_main_left, dims=2)
    Z_ = cat(Z_top, Z_main, Z_bottom, dims=1)

    # pad along 3rd axis
    Z_begin = Z_[:, :, begin:begin+(d3-1), :, :]
    Z_end = Z_[:, :, end-(d3-1):end, :, :]
    return cat(Z_end, Z_, Z_begin, dims=3)
end

function mycircular(Y::AbstractArray{<:Real,3 + 2}, ds::NTuple{3,Int})
    mycircular(Y, ds[1], ds[2], ds[3])
end

"""
Differentiable padarray for 4D
"""
function mycircular(Y::AbstractArray{<:Real,4 + 2}, d1=1, d2=1, d3=1, d4=1)
    Y_top_center = Y[begin:begin+(d1-1), :, :, :, :, :]
    Y_top_right = Y[begin:begin+(d1-1), end-(d2-1):end, :, :, :, :]
    Y_top_left = Y[begin:begin+(d1-1), begin:begin+(d2-1), :, :, :, :]
    Z_bottom = cat(Y_top_right, Y_top_center, Y_top_left, dims=2) # calc pad under

    Y_bottom_center = Y[end-(d1-1):end, :, :, :, :, :]
    Y_bottom_right = Y[end-(d1-1):end, end-(d2-1):end, :, :, :, :]
    Y_bottom_left = Y[end-(d1-1):end, begin:begin+(d2-1), :, :, :, :]
    Z_top = cat(Y_bottom_right, Y_bottom_center, Y_bottom_left, dims=2) # calc pad under

    Y_main_left = Y[:, begin:begin+(d2-1), :, :, :, :]
    Y_main_right = Y[:, end-(d2-1):end, :, :, :, :]
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

function mycircular(Y::AbstractArray{<:Real,4 + 2}, ds::NTuple{4,Int})
    mycircular(Y, ds[1], ds[2], ds[3], ds[4])
end
