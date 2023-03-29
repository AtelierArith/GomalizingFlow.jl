using JSON3: JSON3
using BSON
using Parameters
using ParameterSchedulers
using ParameterSchedulers: Scheduler
using ProgressMeter

"""
using GomalizingFlow

hp = GomalizingFlow.load_hyperparams(
    "cfgs/example2d.toml";
    device_id=0,
    pretrained=nothing,
    result="result",
)
GomalizingFlow.train(hp)
"""
function train(hp)
    device = hp.dp.device
    @info "setup action"

    prior = eval(Meta.parse(hp.tp.prior))
    T = prior |> rand |> eltype
    @show "eltype" T
    @show prior

    @unpack m², λ, lattice_shape = hp.pp
    action = ScalarPhi4Action{T}(m², λ)
    @show action

    @unpack pretrained, batchsize, epochs, iterations, seed = hp.tp
    @info "setup model"
    model, opt, scheduler = if isempty(pretrained)
        model = create_model(hp)
        @info "setup optimiser"
        opt = eval(Meta.parse("$(hp.tp.opt)($(hp.tp.base_lr))"))
        if !isempty(hp.tp.lr_scheduler)
            # base_lr = hp.tp.base_lr
            scheduler = eval(Meta.parse("$(hp.tp.lr_scheduler)"))
        else
            scheduler = Step(T(hp.tp.base_lr), T(1.0), [epochs])
        end
        @info nameof(typeof(opt))
        model, opt, scheduler
    else
        pretrained = abspath(pretrained)
        @info "load model from $(pretrained)"
        BSON.@load pretrained trained_model opt scheduler
        model = trained_model
        @info nameof(typeof(opt))
        model, opt, scheduler
    end
    Flux.trainmode!(model)
    ps = get_training_params(model)

    @info "set random seed: $(seed)"

    result_dir = hp.result_dir
    # JuliaHub
    ENV["RESULTS_FILE"] = result_dir

    @info "create result dir: $(result_dir)"
    mkpath(result_dir)

    @info "create snapshot"
    cp(joinpath(pkgdir(GomalizingFlow), "src"), joinpath(result_dir, "src"), force=true)
    cp(
        joinpath(pkgdir(GomalizingFlow), "Project.toml"),
        joinpath(result_dir, "Project.toml"),
        force=true,
    )
    cp(
        joinpath(pkgdir(GomalizingFlow), "Manifest.toml"),
        joinpath(result_dir, "Manifest.toml"),
        force=true,
    )

    @info "dump hyperparams: $(joinpath(result_dir, "config.toml"))"
    GomalizingFlow.hp2toml(hp, joinpath(result_dir, "config.toml"))
    best_epoch_ess = 1
    best_epoch_acceptance_rate = 1
    best_ess = T |> zero
    best_acceptance_rate = T |> zero
    evaluations = DataFrame(
        :epoch => Int[],
        :loss => T[],
        :ess => T[],
        :best_ess => T[],
        :best_acceptance_rate => T[],
        :best_epoch_ess => Int[],
        :best_epoch_acceptance_rate => Int[],
        :acceptance_rate => T[],
        :elapsed_time => Float64[],
    )

    rng = MersenneTwister(seed)

    @info "start training"
    for (epoch, eta) in zip(1:epochs, scheduler)
        td = @timed begin
            @info "epoch=$epoch"
            @info "lr" eta
            opt.eta = eta
            # switch to trainmode
            Flux.trainmode!(model)
            @showprogress for _ in 1:iterations
                z = rand(rng, prior, lattice_shape..., batchsize)
                logq_device = sum(logpdf.(prior, z), dims=(1:ndims(z)-1)) |> device
                z_device = z |> device
                gs = Flux.gradient(ps) do
                    x, logq_ = model((z_device, logq_device))
                    logq = reshape(logq_, batchsize)
                    logp = -action(x)
                    loss = calc_dkl(logp, logq)
                end
                Flux.Optimise.update!(opt, ps, gs)
            end

            # switch to testmode
            Flux.testmode!(model)
            z = rand(rng, prior, lattice_shape..., batchsize)
            logq_device = sum(logpdf.(prior, z), dims=(1:ndims(z)-1)) |> device
            z_device = z |> device
            x, logq_ = model((z_device, logq_device))
            logq = reshape(logq_, batchsize)
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
                model,
                prior,
                action,
                lattice_shape;
                batchsize,
                nsamples,
                device=device,
            )
            acceptance_rate = 100mean(history_current_epoch.accepted)
            @show acceptance_rate

            @info "progress logging:" epoch = epoch loss = loss ess = ess acceptance_rate =
                acceptance_rate
        end # @timed
        # save best checkpoint regarding ess
        if ess >= best_ess
            @info "Found best ess"
            @show epoch
            best_ess = ess
            best_epoch_ess = epoch
            # save model
            trained_model_best_ess = model |> cpu
            opt_best_ess = opt |> cpu
            BSON.@save joinpath(
                result_dir,
                "trained_model_best_ess.bson",
            ) trained_model_best_ess opt_best_ess scheduler
            @info "make mcmc ensamble"
            history_best_ess = history_current_epoch
            @info "save history_best_ess to $(joinpath(result_dir, "history_best_ess.bson"))"
            BSON.@save joinpath(result_dir, "history_best_ess.bson") history_best_ess
            Printf.@printf "acceptance_rate= %.2f [percent]\n" 100mean(
                history_best_ess.accepted[2000:end],
            )
        end

        # save best checkpoint regarding acceptance_rate
        if acceptance_rate >= best_acceptance_rate
            @info "Found best acceptance_rate"
            @show epoch
            best_acceptance_rate = acceptance_rate
            best_epoch_acceptance_rate = epoch
            # save model
            trained_model_best_acceptance_rate = model |> cpu
            opt_best_acceptance_rate = opt |> cpu
            BSON.@save joinpath(
                result_dir,
                "trained_model_best_acceptance_rate.bson",
            ) trained_model_best_acceptance_rate opt_best_acceptance_rate scheduler
            @info "make mcmc ensamble"
            history_best_acceptance_rate = history_current_epoch
            @info "save history_best_acceptance_rate to $(joinpath(result_dir, "history_best_acceptance_rate.bson"))"
            BSON.@save joinpath(result_dir, "history_best_acceptance_rate.bson") history_best_acceptance_rate
            Printf.@printf "acceptance_rate= %.2f [percent]\n" 100mean(
                history_best_acceptance_rate.accepted[2000:end],
            )
        end

        elapsed_time = td.time
        @show elapsed_time
        push!(
            evaluations,
            Dict(
                pairs((;
                    epoch,
                    loss,
                    ess,
                    best_epoch_ess,
                    best_epoch_acceptance_rate,
                    best_ess,
                    best_acceptance_rate,
                    acceptance_rate,
                    elapsed_time,
                )),
            ),
        )

        CSV.write(joinpath(result_dir, "evaluations.csv"), evaluations)
    end
    @info "finished training"
    trained_model = model |> cpu
    @info "save model"
    BSON.@save joinpath(result_dir, "trained_model.bson") trained_model opt scheduler
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
    BSON.@save joinpath(result_dir, "history.bson") history

    result4juliahub = Dict()
    for col in names(evaluations)
        result4juliahub[col] = evaluations[!, col]
    end

    ENV["RESULTS"] = JSON3.write(result4juliahub)
    ENV["RESULTS_FILE"] = result_dir
    @info "Done"
end
