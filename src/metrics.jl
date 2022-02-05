calc_dkl(logp, logq) = mean(logq .- logp)

function compute_ess(logp, logq)
    logw = logp - logq
    log_ess = 2 * logsumexp(logw) - logsumexp(2 * logw)
    ess_per_cfg = exp(log_ess) / length(logw)
    return ess_per_cfg
end
