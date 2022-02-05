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
    T = prior |> rand |> eltype
    @show T
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
    best_epoch = 1
    best_ess = T |> zero
    evaluations = DataFrame(
        :epoch => Int[],
        :loss => T[],
        :ess => T[],
        :best_ess => T[],
        :best_epoch => Int[],
        :accepted_ratio => T[],
    )

    @info "start training"
    for epoch in 1:epochs
        @info "epoch=$epoch"
        # switch to trainmode
        Flux.trainmode!(model)
        @showprogress for _ in 1:iterations
            z = rand(prior, lattice_shape..., batchsize)
            logq_device = sum(logpdf.(prior, z), dims=(1:ndims(z)-1)) |> device
            z_device = z |> device
            gs = Flux.gradient(ps) do
                x, logq_ = model((z_device, logq_device))
                logq = dropdims(
                    logq_,
                    dims=Tuple(1:(ndims(logq_)-1)),
                )
                logp = -action(x)
                loss = calc_dkl(logp, logq)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end

        # switch to testmode
        Flux.testmode!(model)
        z = rand(prior, lattice_shape..., batchsize)
        logq_device = sum(logpdf.(prior, z), dims=(1:ndims(z)-1)) |> device
        z_device = z |> device
        x, logq_ = model((z_device, logq_device))
        logq = dropdims(
            logq_,
            dims=Tuple(1:(ndims(logq_)-1)),
        )

        logp = -action(x)
        loss = calc_dkl(logp, logq)
        @show loss
        println("loss per site", loss / prod(lattice_shape))
        @show mean(logp)
        @show mean(logq)
        ess = compute_ess(logp, logq)
        @show ess

        nsamples = 8196
        history_current_epoch = make_mcmc_ensamble(
            model |> cpu,
            prior,
            action,
            lattice_shape;
            batchsize,
            nsamples,
            device=cpu,
        )
        accepted_ratio = 100mean(history_current_epoch.accepted)
        @show accepted_ratio
        # save best checkpoint
        if ess >= best_ess
            @info "Found best ess"
            @show epoch
            best_ess = ess
            best_epoch = epoch
            # save model
            trained_model_best_ess = model |> cpu
            LFT.BSON.@save joinpath(result_dir, "trained_model_best_ess.bson") trained_model_best_ess
            # save mcmc history
            @info "make mcmc ensamble"
            history_best_ess = history_current_epoch
            @info "save history_best_ess to $(joinpath(result_dir, "history_best_ess.bson"))"
            LFT.BSON.@save joinpath(result_dir, "history_best_ess.bson") history_best_ess
            Printf.@printf "accepted_ratio= %.2f [percent]" 100mean(
                history_best_ess.accepted,
            )
        end

        push!(evaluations, Dict(pairs((; epoch, loss, ess, best_epoch, best_ess, accepted_ratio))))

        CSV.write(joinpath(result_dir, "evaluations.csv"), evaluations)
        @info "Done epoch $(epoch)"
    end
    @info "finished training"
    trained_model = model |> cpu
    @info "save model"
    LFT.BSON.@save joinpath(result_dir, "trained_model.bson") trained_model
    @info "make mcmc ensamble"
    nsamples = 8196
    history = make_mcmc_ensamble(
        trained_model,
        prior,
        action,
        lattice_shape;
        batchsize,
        nsamples,
        device=cpu,
    )
    @info "save history to $(joinpath(result_dir, "history.bson"))"
    LFT.BSON.@save joinpath(result_dir, "history.bson") history
    @info "Done"
end
