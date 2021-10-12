using CUDA
using Parameters
using Distributions
using ProgressMeter
using Flux
using BSON

using LFT
using LFT:ScalarPhi4Action

function load_hyperparams3D()
    @info "load hyper parmeters"
    device_id = 1
    dp = LFT.DeviceParams(device_id)

    batchsize = 64
    epochs = 300
    iterations = 100
    base_lr = 0.0015f0
    opt = "ADAM"
    prior = Normal{Float32}(0.f0, 1.f0)
    tp = LFT.TrainingParams(; batchsize, epochs, iterations, base_lr, opt, prior)

    L = 8
    Nd = 3
    m² = -4.
    λ = 8
    pp = LFT.PhysicalParams(;L, Nd, m², λ)
    
    n_layers = 16
    hidden_sizes = (8, 8)
    kernel_size = 3
    inC = 1
    outC = 2
    use_final_tanh = true
    mp = LFT.ModelParams(;n_layers, hidden_sizes, kernel_size, inC, outC, use_final_tanh)
    
    return LFT.HyperParams(dp, tp, pp, mp)
end

function train(hp=load_hyperparams3D())
    device = hp.dp.device
    @info "setup action"
    phi4_action = ScalarPhi4Action(hp.pp.m², hp.pp.λ)
    @info "setup model"
    model, ps = LFT.create_model(hp)

    lattice_shape = hp.pp.lattice_shape
    prior = hp.tp.prior

    batchsize = hp.tp.batchsize
    epochs = hp.tp.epochs
    iterations = hp.tp.iterations
    @info "setup optimiser"
    opt = eval(Meta.parse("$(hp.tp.opt)($(hp.tp.base_lr))"))
    @info opt

    @info "start training"
    for _ in 1:epochs
        @showprogress for _ in 1:iterations
            z = rand(prior, lattice_shape..., batchsize)
            logq_device = sum(logpdf(prior, z), dims=(1:ndims(z) - 1)) |> device
            z_device = z |> device
            gs = Flux.gradient(ps) do
                x, logq_ = model((z_device, logq_device))
                logq = dropdims(
                    logq_,
                    dims=Tuple(1:(ndims(logq_) - 1))
                )
                logp = -phi4_action(x)
                loss = LFT.calc_dkl(logp, logq)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end
        
        z = rand(prior, lattice_shape..., batchsize)
        logq_device = sum(logpdf(prior, z), dims=(1:ndims(z) - 1)) |> device
        z_device = z |> device
        x, logq_ = model((z_device, logq_device))
        logq = dropdims(
            logq_,
            dims=Tuple(1:(ndims(logq_) - 1))
        )

        logp = -phi4_action(x)
        loss = LFT.calc_dkl(logp, logq)
        @show loss
        println("loss per site", loss / prod(lattice_shape))
        ess = LFT.compute_ess(logp, logq)
        @show ess
    end
    @info "finished training"
    trained_model = model |> cpu
    @info "save model"
    BSON.@save "trained_model.bson" trained_model
end