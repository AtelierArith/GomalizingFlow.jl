
using LFT
using TOML

using Test

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