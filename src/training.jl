function train(hp)
    device = hp.dp.device
    @info "setup action"
    @unpack m², λ, lattice_shape = hp.pp
    action = ScalarPhi4Action(m², λ)

    @unpack pretrained, batchsize, epochs, iterations, seed = hp.tp
    @info "setup model"
    if isempty(pretrained)
        model = create_model(hp)
    else
        @info "load model from $(abspath((pretrained)))"
        BSON.@load abspath(pretrained) trained_model
        model = trained_model
    end
    Flux.trainmode!(model)
    ps = get_training_params(model)

    prior = eval(Meta.parse(hp.tp.prior))
    @show prior

    @info "setup optimiser"
    opt = eval(Meta.parse("$(hp.tp.opt)($(hp.tp.base_lr))"))
    @info opt
    @info "set random seed: $(seed)"
    Random.seed!(seed)

    result_dir = hp.result_dir
    @info "create result dir: $(result_dir)"
    mkpath(result_dir)
    @info "dump hyperparams: $(joinpath(result_dir, "config.toml"))"
    LFT.hp2toml(hp, joinpath(result_dir, "config.toml"))

    @info "start training"
    for _ in 1:epochs
        # switch to trainmode
        Flux.trainmode!(model)
        @showprogress for _ in 1:iterations
            z = rand(prior, lattice_shape..., batchsize)
            logq_device = sum(logpdf.(prior, z), dims=(1:ndims(z) - 1)) |> device
            z_device = z |> device
            gs = Flux.gradient(ps) do
                x, logq_ = model((z_device, logq_device))
                logq = dropdims(
                    logq_,
                    dims=Tuple(1:(ndims(logq_) - 1))
                )
                logp = -action(x)
                loss = calc_dkl(logp, logq)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end

        # switch to testmode
        Flux.testmode!(model)
        z = rand(prior, lattice_shape..., batchsize)
        logq_device = sum(logpdf.(prior, z), dims=(1:ndims(z) - 1)) |> device
        z_device = z |> device
        x, logq_ = model((z_device, logq_device))
        logq = dropdims(
            logq_,
            dims=Tuple(1:(ndims(logq_) - 1))
        )

        logp = -action(x)
        loss = calc_dkl(logp, logq)
        @show loss
        println("loss per site", loss / prod(lattice_shape))
        ess = compute_ess(logp, logq)
        @show ess
    end
    @info "finished training"
    trained_model = model |> cpu
    @info "save model"
    LFT.BSON.@save joinpath(result_dir, "trained_model.bson") trained_model
    @info "make mcmc ensamble"
    nsamples = 8196
    history = make_mcmc_ensamble(trained_model, prior, action, lattice_shape; batchsize, nsamples, device=cpu)
    @info "save history to $(joinpath(result_dir, "history.bson"))"
    LFT.BSON.@save joinpath(result_dir, "history.bson") history
    @info "Done"
end
