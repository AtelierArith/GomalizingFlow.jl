function make_mcmc_ensamble(
    model,
    prior,
    action,
    lattice_shape;
    batchsize,
    nsamples,
    device=cpu,
    seed=2009,
)
    rng = MersenneTwister(seed)
    Nd = length(lattice_shape)
    history = (x=Array{Float32,Nd}[], logq=Float32[], logp=Float32[], accepted=Bool[])
    c = 0
    for _ in 1:(nsamples÷batchsize+1)
        z = rand(rng, prior, lattice_shape..., batchsize)
        logq_device = sum(logpdf.(prior, z), dims=(1:ndims(z)-1)) |> device
        z_device = z |> device
        x_device, logq_ = model((z_device, logq_device))
        logq = reshape(logq_, batchsize) |> cpu
        # back to cpu
        logp = -action(x_device) |> cpu
        x = x_device |> cpu

        for b in 1:batchsize
            new_x = x[.., b]
            new_logq = logq[b]
            new_logp = logp[b]
            if isempty(history[:logp])
                accepted = true
            else
                last_logp = history[:logp][end]
                last_logq = history[:logq][end]
                last_x = history[:x][end]
                p_accept = exp((new_logp - new_logq) - (last_logp - last_logq))
                p_accept = min(one(p_accept), p_accept)
                draw = rand(rng, typeof(p_accept))
                if draw < p_accept
                    accepted = true
                else
                    accepted = false
                    new_x = last_x
                    new_logp = last_logp
                    new_logq = last_logq
                end
            end
            # update history
            push!(history[:logp], new_logp)
            push!(history[:logq], new_logq)
            push!(history[:x], new_x)
            push!(history[:accepted], accepted)
        end
        c += batchsize
        if c >= nsamples
            break
        end
    end
    history
end
