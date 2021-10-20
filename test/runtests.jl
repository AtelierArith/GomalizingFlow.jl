using Test
using TOML

using BSON
using Flux
using Parameters
using Distributions
using ImageFiltering

using LFT

function create_hp_example2d()
    @info "load hyper parmeters"
    device_id = 0
    dp = LFT.DeviceParams(device_id)

    batchsize = 64
    epochs = 40
    iterations = 100
    base_lr = 0.001
    opt = "ADAM"
    prior = "Normal{Float32}(0.f0, 1.f0)"
    tp = LFT.TrainingParams(; batchsize, epochs, iterations, base_lr, opt, prior)

    L = 8
    Nd = 2
    M2 = -4.
    lam = 8
    pp = LFT.PhysicalParams(;L, Nd, M2, lam)

    n_layers = 16
    hidden_sizes = [8, 8]
    kernel_size = 3
    inC = 1
    outC = 2
    use_final_tanh = true
    mp = LFT.ModelParams(;n_layers, hidden_sizes, kernel_size, inC, outC, use_final_tanh)

    return LFT.HyperParams(dp, tp, pp, mp, "config.toml")
end


function create_hp_example3d()
    @info "load hyper parmeters"
    device_id = 0
    dp = LFT.DeviceParams(device_id)

    seed = 12345
    batchsize = 64
    epochs = 300
    iterations = 100
    base_lr = 0.0015
    opt = "ADAM"
    prior = "Normal{Float32}(0.f0, 1.f0)"
    tp = LFT.TrainingParams(; seed, batchsize, epochs, iterations, base_lr, opt, prior)

    L = 8
    Nd = 3
    M2 = -4.
    lam = 8
    pp = LFT.PhysicalParams(;L, Nd, M2, lam)

    seed = 2021
    n_layers = 16
    hidden_sizes = [8, 8]
    kernel_size = 3
    inC = 1
    outC = 2
    use_final_tanh = true
    mp = LFT.ModelParams(;seed, n_layers, hidden_sizes, kernel_size, inC, outC, use_final_tanh)

    return LFT.HyperParams(dp, tp, pp, mp, "config.toml")
end

@testset "PhysicalParams property" begin
    L = 8
    Nd = 2
    M2 = -4.
    lam = 8
    pp = LFT.PhysicalParams(;L, Nd, M2, lam)
    @test pp.L == L
    @test pp.lattice_shape == (L, L)
    @test pp.m² == pp.M2
    @test pp.λ == pp.lam
end

@testset "example2d.toml" begin
    path = "../cfgs/example2d.toml"
    ref_hp = create_hp_example2d()
    tar_hp = LFT.load_hyperparams(path)
    for f in fieldnames(typeof(ref_hp.tp))
        @test getfield(tar_hp.tp, f) == getfield(ref_hp.tp, f)
    end
    @test tar_hp.pp == ref_hp.pp
    for f in fieldnames(typeof(ref_hp.mp))
        @test getfield(tar_hp.mp, f) == getfield(ref_hp.mp, f)
    end
    # @test tar_hp.mp == ref_hp.mp # <--- fails... why?!
end

@testset "example3d.toml" begin
path = "../cfgs/example3d.toml"
    ref_hp = create_hp_example3d()
    tar_hp = LFT.load_hyperparams(path)
    for f in fieldnames(typeof(ref_hp.tp))
        @test getfield(tar_hp.tp, f) == getfield(ref_hp.tp, f)
    end
    @test tar_hp.pp == ref_hp.pp
    for f in fieldnames(typeof(ref_hp.mp))
        @test getfield(tar_hp.mp, f) == getfield(ref_hp.mp, f)
    end
    # @test tar_hp.mp == ref_hp.mp <--- fails... why?!
end

include("pyinterface.jl")

@testset "circular" begin
    # used for 2D Lattice
    x = rand(4, 4, 4, 4)
    tar = LFT.mycircular(x)
    ref = ImageFiltering.padarray(x, Pad(:circular, 1, 1, 0, 0)).parent
    @test tar ≈ ref
    # used for 3D Lattice
    x = rand(4, 4, 4, 4, 4)
    tar = LFT.mycircular(x)
    ref = ImageFiltering.padarray(x, Pad(:circular, 1, 1, 1, 0, 0)).parent
    @test tar ≈ ref

    x = rand(4, 4, 4, 4, 4, 4)
    tar = LFT.mycircular(x)
    ref = ImageFiltering.padarray(x, Pad(:circular, 1, 1, 1, 1, 0, 0)).parent
    tar ≈ ref
end

@testset "make_checker_mask" begin
    @test LFT.make_checker_mask((8, 8), 0) == [
        0  1  0  1  0  1  0  1
        1  0  1  0  1  0  1  0
        0  1  0  1  0  1  0  1
        1  0  1  0  1  0  1  0
        0  1  0  1  0  1  0  1
        1  0  1  0  1  0  1  0
        0  1  0  1  0  1  0  1
        1  0  1  0  1  0  1  0
    ]
    @test LFT.make_checker_mask((8, 8), 1) == [
        1  0  1  0  1  0  1  0
        0  1  0  1  0  1  0  1
        1  0  1  0  1  0  1  0
        0  1  0  1  0  1  0  1
        1  0  1  0  1  0  1  0
        0  1  0  1  0  1  0  1
        1  0  1  0  1  0  1  0
        0  1  0  1  0  1  0  1
    ]

    a1 = [
        0  1  0
        1  0  1
        0  1  0
    ]
    a2 = [
        1  0  1
        0  1  0
        1  0  1
        ]
    a3 = [
        0  1  0
        1  0  1
        0  1  0
        ]
    @test LFT.make_checker_mask((3, 3, 3), 0) == cat(a1, a2, a3, dims=3)

    a1 = [
    1  0  1
    0  1  0
    1  0  1
    ]
    a2 = [
        0  1  0
        1  0  1
        0  1  0
        ]
    a3 = [
        1  0  1
        0  1  0
        1  0  1
        ]
    @test LFT.make_checker_mask((3, 3, 3), 1) == cat(a1, a2, a3, dims=3)
end

@testset "model" begin
    hp = LFT.load_hyperparams(joinpath(@__DIR__, "assets", "config.toml"))
    model1 = LFT.create_model(hp)
    model2 = LFT.create_model(hp)
    for i in 1:length(model1)
        @test model1[i].mask == model2[i].mask
        for j in 1:length(model1[i].net)
            if model1[i].net[j] isa Conv
                @test model1[i].net[j].weight == model2[i].net[j].weight
    @test model1[i].net[j].bias == model2[i].net[j].bias
            end
        end
    end
                end
    
@testset "training" begin
    hp = LFT.load_hyperparams(joinpath(@__DIR__, "assets", "config.toml"))
    LFT.train(hp)
    function loadtar()
        BSON.@load joinpath(@__DIR__, "result/config", "trained_model.bson") trained_model
    BSON.@load joinpath(@__DIR__, "result/config", "history.bson") history
        return trained_model, history
        end
    function loadref()
        BSON.@load joinpath(@__DIR__, "assets", "trained_model.bson") trained_model
        BSON.@load joinpath(@__DIR__, "assets", "history.bson") history
        return trained_model, history
    end

    model1, history1 = loadtar()
    model2, history2 = loadref()
    for i in 1:length(model1)
        @test model1[i].mask == model2[i].mask
        for j in 1:length(model1[i].net)
            if model1[i].net[j] isa Conv
                @test model1[i].net[j].weight == model2[i].net[j].weight
            @test model1[i].net[j].bias == model2[i].net[j].bias
            end
        end
    end
    for k in keys(history1)
        @test history1[k] ≈ history2[k]
    end
end

    @testset "retraining" begin
    path = joinpath(@__DIR__, "assets", "config.toml")
    pretrained = joinpath(@__DIR__, "assets", "trained_model.bson")
    hp = LFT.load_hyperparams(path; pretrained)
    LFT.train(hp)
end

