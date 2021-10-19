
function train(hp)
    device = hp.dp.device
    @info "setup action"
    action = ScalarPhi4Action(hp.pp.m², hp.pp.λ)
    @info "setup model"
    if isempty(hp.tp.pretrained)
        model = create_model(hp)
    else
        @info "load model from $(hp.tp.pretrained)"
        BSON.@load hp.tp.pretrained trained_model
        model = trained_model
    end
    Flux.trainmode!(model)
    ps = get_training_params(model)

    lattice_shape = hp.pp.lattice_shape
    prior = eval(Meta.parse(hp.tp.prior))
    @show prior

    batchsize = hp.tp.batchsize
    epochs = hp.tp.epochs
    iterations = hp.tp.iterations
    @info "setup optimiser"
    opt = eval(Meta.parse("$(hp.tp.opt)($(hp.tp.base_lr))"))
    @info opt

    result_dir = joinpath(hp.tp.result, splitext(basename(hp.path))[begin])
    @info "create result dir $(result_dir)"
    mkpath(result_dir)
    cp(hp.path, joinpath(result_dir, "config.toml"), force=true)

    Random.seed!(hp.tp.seed)
    @info "start training"
    for _ in 1:epochs
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
    BSON.@save joinpath(result_dir, "trained_model.bson") trained_model
    @info "make mcmc ensamble"
    nsamples = 8196
    history = make_mcmc_ensamble(model, prior, action, lattice_shape; batchsize, nsamples, device=cpu, seed=2009)
    BSON.@save joinpath(result_dir, "history.bson") history
end