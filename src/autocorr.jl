using Statistics

"""
Compute \$\\bar{\\Gamma}(t)\$
"""
function approx_autocorr(a::AbstractVector{T}, t::Int) where T
    t = abs(t)
    ā = mean(a)
    s = zero(T)
    N = length(a)
    for i in 1:(N-t)
        s += (a[i] - ā) * (a[i+t] - ā)
    end
    return s / (N - t)
end

approx_normalized_autocorr(arr, t) = approx_autocorr(arr, t)/approx_autocorr(arr, 0)
ρ̄(arr, t) = approx_normalized_autocorr(arr, t) # alias

# variance of the autocorrelation function
function δρ²(a::AbstractArray{T}, t; Λ=600) where T
    s = zero(T)
    for k in 1:(t + Λ)
        s += (ρ̄(a, k + t) + ρ̄(a, k - t) - 2ρ̄(a, k) * ρ̄(a, t))^2
    end
    return s / length(a)
end

"""
Compute \$\\tau_{int}\$
"""
function integrated_autocorrelation_time(arr)
    W = -1
    for t in 1:length(arr)
        if ρ̄(arr, t) ≤ √(δρ²(arr, t))
            W = t
            break
        end
    end

    0.5 + sum(ρ̄(arr, t) for t in 1:W)
end

