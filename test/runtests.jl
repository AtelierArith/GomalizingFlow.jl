using Test
using Random
using TOML

using BSON
using Flux
using Parameters
using Distributions
using ImageFiltering
using ParameterSchedulers

using GomalizingFlow

function create_hp_example2d()
    @info "load hyper parmeters"
    configversion = v"0.1"
    device_id = 0
    dp = GomalizingFlow.DeviceParams(device_id)

    batchsize = 64
    epochs = 200
    iterations = 100
    base_lr = 0.001
    opt = "Adam"
    lr_scheduler = ""
    prior = "Normal{Float32}(0.f0, 1.f0)"
    tp = GomalizingFlow.TrainingParams(;
        batchsize,
        epochs,
        iterations,
        base_lr,
        opt,
        lr_scheduler,
        prior,
    )

    L = 8
    Nd = 2
    M2 = -4.0
    lam = 8
    pp = GomalizingFlow.PhysicalParams(; L, Nd, M2, lam)

    n_layers = 16
    hidden_sizes = [8, 8]
    kernel_size = 3
    inC = 1
    outC = 2
    use_final_tanh = true
    mp = GomalizingFlow.ModelParams(;
        n_layers,
        hidden_sizes,
        kernel_size,
        inC,
        outC,
        use_final_tanh,
    )

    return GomalizingFlow.HyperParams(configversion, dp, tp, pp, mp, "config.toml")
end

function create_hp_example3d()
    @info "load hyper parmeters"
    configversion = v"0.1"
    device_id = 0
    dp = GomalizingFlow.DeviceParams(device_id)

    seed = 12345
    batchsize = 64
    epochs = 500
    iterations = 100
    base_lr = 0.001
    opt = "Adam"
    lr_scheduler = "Step(0.001f0, 0.1f0, [200])"
    prior = "Normal{Float32}(0.f0, 1.f0)"
    tp = GomalizingFlow.TrainingParams(;
        seed,
        batchsize,
        epochs,
        iterations,
        base_lr,
        opt,
        lr_scheduler,
        prior,
    )

    L = 8
    Nd = 3

    M2 = -4.0 # m²
    lam = 5.113 # λ

    pp = GomalizingFlow.PhysicalParams(; L, Nd, M2, lam)

    seed = 2021
    n_layers = 16
    hidden_sizes = [8, 8]
    kernel_size = 3
    inC = 1
    outC = 2
    use_final_tanh = true
    mp = GomalizingFlow.ModelParams(;
        seed,
        n_layers,
        hidden_sizes,
        kernel_size,
        inC,
        outC,
        use_final_tanh,
    )

    return GomalizingFlow.HyperParams(configversion, dp, tp, pp, mp, "config.toml")
end

@testset "PhysicalParams property" begin
    L = 8
    Nd = 2
    M2 = -4.0
    lam = 8
    pp = GomalizingFlow.PhysicalParams(; L, Nd, M2, lam)
    @test pp.L == L
    @test pp.lattice_shape == (L, L)
    @test pp.m² == pp.M2
    @test pp.λ == pp.lam
end

@testset "example2d.toml" begin
    path = "../cfgs/example2d.toml"
    ref_hp = create_hp_example2d()
    tar_hp = GomalizingFlow.load_hyperparams(path)
    for f in fieldnames(typeof(ref_hp.tp))
        @test getfield(tar_hp.tp, f) == getfield(ref_hp.tp, f)
    end
    @test tar_hp.pp == ref_hp.pp
    for f in fieldnames(typeof(ref_hp.mp))
        @test getfield(tar_hp.mp, f) == getfield(ref_hp.mp, f)
    end
end

@testset "example3d.toml" begin
    path = "../cfgs/example3d.toml"
    ref_hp = create_hp_example3d()
    tar_hp = GomalizingFlow.load_hyperparams(path)
    for f in fieldnames(typeof(ref_hp.tp))
        @test getfield(tar_hp.tp, f) == getfield(ref_hp.tp, f)
    end
    @test tar_hp.pp == ref_hp.pp
    for f in fieldnames(typeof(ref_hp.mp))
        @test getfield(tar_hp.mp, f) == getfield(ref_hp.mp, f)
    end
end

# PotentialDistributions.jl
include("potential.jl")

include("pyinterface.jl")

# Action
include("actions.jl")

@testset "circular default" begin
    # used for 2D Lattice
    x = rand(4, 4, 4, 4)
    tar = GomalizingFlow.mycircular(x)
    ref = ImageFiltering.padarray(x, Pad(:circular, 1, 1, 0, 0)).parent
    @test tar ≈ ref
    # used for 3D Lattice
    x = rand(4, 4, 4, 4, 4)
    tar = GomalizingFlow.mycircular(x)
    ref = ImageFiltering.padarray(x, Pad(:circular, 1, 1, 1, 0, 0)).parent
    @test tar ≈ ref

    x = rand(4, 4, 4, 4, 4, 4)
    tar = GomalizingFlow.mycircular(x)
    ref = ImageFiltering.padarray(x, Pad(:circular, 1, 1, 1, 1, 0, 0)).parent
    tar ≈ ref
end

@testset "circular default padding" begin
    # used for 2D Lattice
    d1 = 2
    d2 = 3
    d3 = 4
    d4 = 5
    x = rand(4, 4, 4, 4)
    tar = GomalizingFlow.mycircular(x, d1, d2)
    ref = ImageFiltering.padarray(x, Pad(:circular, d1, d2, 0, 0)).parent
    @test tar ≈ ref
    # used for 3D Lattice
    x = rand(4, 4, 4, 4, 4)
    tar = GomalizingFlow.mycircular(x, d1, d2, d3)
    ref = ImageFiltering.padarray(x, Pad(:circular, d1, d2, d3, 0, 0)).parent
    @test tar ≈ ref

    # used for 4D Lattice
    x = rand(7, 7, 7, 7, 7, 7)
    tar = GomalizingFlow.mycircular(x, d1, d2, d3, d4)
    ref = ImageFiltering.padarray(x, Pad(:circular, d1, d2, d3, d4, 0, 0)).parent
    @test tar ≈ ref
end

@testset "circular padding with Base.Fix2" begin
    # used for 2D Lattice
    d1 = 2
    d2 = 3
    d3 = 4
    d4 = 5
    x = rand(4, 4, 4, 4)
    c = Chain(Base.Fix2(GomalizingFlow.mycircular, (d1, d2)))
    tar = c(x)
    ref = ImageFiltering.padarray(x, Pad(:circular, d1, d2, 0, 0)).parent
    @test tar ≈ ref
    # used for 3D Lattice
    x = rand(4, 4, 4, 4, 4)
    c = Chain(Base.Fix2(GomalizingFlow.mycircular, (d1, d2, d3)))
    tar = c(x)
    ref = ImageFiltering.padarray(x, Pad(:circular, d1, d2, d3, 0, 0)).parent
    @test tar ≈ ref

    # used for 4D Lattice
    x = rand(7, 7, 7, 7, 7, 7)
    c = Chain(Base.Fix2(GomalizingFlow.mycircular, (d1, d2, d3, d4)))
    tar = c(x)
    ref = ImageFiltering.padarray(x, Pad(:circular, d1, d2, d3, d4, 0, 0)).parent
    @test tar ≈ ref
end

@testset "make_checker_mask" begin
    @test GomalizingFlow.make_checker_mask((8, 8), 0) == [
        0 1 0 1 0 1 0 1
        1 0 1 0 1 0 1 0
        0 1 0 1 0 1 0 1
        1 0 1 0 1 0 1 0
        0 1 0 1 0 1 0 1
        1 0 1 0 1 0 1 0
        0 1 0 1 0 1 0 1
        1 0 1 0 1 0 1 0
    ]
    @test GomalizingFlow.make_checker_mask((8, 8), 1) == [
        1 0 1 0 1 0 1 0
        0 1 0 1 0 1 0 1
        1 0 1 0 1 0 1 0
        0 1 0 1 0 1 0 1
        1 0 1 0 1 0 1 0
        0 1 0 1 0 1 0 1
        1 0 1 0 1 0 1 0
        0 1 0 1 0 1 0 1
    ]

    a1 = [
        0 1 0
        1 0 1
        0 1 0
    ]
    a2 = [
        1 0 1
        0 1 0
        1 0 1
    ]
    a3 = [
        0 1 0
        1 0 1
        0 1 0
    ]
    @test GomalizingFlow.make_checker_mask((3, 3, 3), 0) == cat(a1, a2, a3, dims=3)

    a1 = [
        1 0 1
        0 1 0
        1 0 1
    ]
    a2 = [
        0 1 0
        1 0 1
        0 1 0
    ]
    a3 = [
        1 0 1
        0 1 0
        1 0 1
    ]
    @test GomalizingFlow.make_checker_mask((3, 3, 3), 1) == cat(a1, a2, a3, dims=3)
end

@testset "model" begin
    hp = GomalizingFlow.load_hyperparams(joinpath(@__DIR__, "assets", "config.toml"))
    model1 = GomalizingFlow.create_model(hp)
    model2 = GomalizingFlow.create_model(hp)
    for i in 1:length(model1)
        @test model1[i].mask == model2[i].mask
        for j in 1:length(model1[i].net)
            if model1[i].net[j] isa Conv
                @test model1[i].net[j].weight ≈ model2[i].net[j].weight
                @test model1[i].net[j].bias ≈ model2[i].net[j].bias
            end
        end
    end
end

@testset "mcmc" begin
    hp = GomalizingFlow.load_hyperparams(joinpath(@__DIR__, "assets", "config.toml"))

    device = hp.dp.device

    @unpack batchsize, epochs, iterations, seed = hp.tp
    prior = eval(Meta.parse(hp.tp.prior))
    T = prior |> rand |> eltype

    @info "setup action"
    @unpack m², λ, lattice_shape = hp.pp
    action = GomalizingFlow.ScalarPhi4Action{T}(m², λ)

    @info "setup model"
    model = GomalizingFlow.create_model(hp)
    # switch to testmode
    Flux.testmode!(model)
    nsamples = 8196
    history1 = GomalizingFlow.make_mcmc_ensamble(
        model,
        prior,
        action,
        lattice_shape;
        batchsize,
        nsamples,
        device=cpu,
    )
    history2 = GomalizingFlow.make_mcmc_ensamble(
        model,
        prior,
        action,
        lattice_shape;
        batchsize,
        nsamples,
        device=cpu,
    )
    for k in keys(history1)
        @test history1[k] ≈ history2[k]
    end
end

@testset "training" begin
    hp = GomalizingFlow.load_hyperparams(joinpath(@__DIR__, "assets", "config.toml"))
    GomalizingFlow.train(hp)

    function loadtar()
        config = TOML.parsefile(joinpath(@__DIR__, "result/config", "config.toml"))
        BSON.@load joinpath(@__DIR__, "result/config", "trained_model.bson") trained_model
        BSON.@load joinpath(@__DIR__, "result/config", "history.bson") history
        return config, trained_model, history
    end

    function loadref()
        config = TOML.parsefile(joinpath(@__DIR__, "assets", "config.toml"))
        BSON.@load joinpath(@__DIR__, "assets", "trained_model.bson") trained_model
        BSON.@load joinpath(@__DIR__, "assets", "history.bson") history
        return config, trained_model, history
    end

    config1, model1, history1 = loadtar()
    config2, model2, history2 = loadref()

    delete!(config1["training"], "result")
    delete!(config2["training"], "result")

    for (k, v) in config1
        @test config1[k] == config2[k]
    end

    for i in 1:length(model1)
        @test model1[i].mask == model2[i].mask
        for j in 1:length(model1[i].net)
            if model1[i].net[j] isa Conv
                @test model1[i].net[j].weight ≈ model2[i].net[j].weight
                @test model1[i].net[j].bias ≈ model2[i].net[j].bias
            end
        end
    end
    for k in keys(history1)
        @test history1[k] ≈ history2[k]
    end
end

@testset "retraining" begin
    configpath = joinpath(@__DIR__, "assets", "config.toml")
    pretrained = joinpath(@__DIR__, "result/config", "trained_model.bson")
    hp = GomalizingFlow.load_hyperparams(configpath; pretrained)
    GomalizingFlow.train(hp)
    # notify retraining has finished
    @test true
end
