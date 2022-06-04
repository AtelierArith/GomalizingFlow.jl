using PyCall
using GomalizingFlow: reversedims

pushfirst!(PyVector(pyimport("sys")."path"), "")

@testset "ScalarPhi4Action" begin
    pyaction = pyimport("pymod.action")
    cfgs = pyaction.cfgs

    M2 = 1.0
    lam = 1.0
    out1 = GomalizingFlow.ScalarPhi4Action(M2, lam)(cfgs |> reversedims)
    pyout1 = pyaction.out1
    @test out1 ≈ pyout1

    M2 = -4.0
    lam = 8.0
    out2 = GomalizingFlow.ScalarPhi4Action(M2, lam)(cfgs |> reversedims)
    pyout2 = pyaction.out2
    @test out2 ≈ pyout2
end

torch = pyimport("torch")
pycopy = pyimport("copy")
Base.copy(po::PyObject) = pycopy.copy(po)
Base.deepcopy(po::PyObject) = pycopy.deepcopy(po)
function pairwise(iterable)
    b = deepcopy(iterable)
    popfirst!(b)
    a = iterable
    return zip(a, b)
end
jl2torch(x::AbstractArray) = torch.from_numpy(x |> reversedims)
torch2jl(x::PyObject) = x.data.numpy() |> reversedims

function torch2conv(lay, σ=Flux.identity)
    W = reversedims(lay.weight.data.numpy())
    W = W[end:-1:1, end:-1:1, :, :]
    if isnothing(lay.bias)
        b = zeros(eltype(W), size(W, 4))
    else
        b = reversedims(lay.bias.data.numpy())
    end
    pad = lay.padding
    stride = lay.stride
    if lay.padding_mode == "circular"
        pad = 0
        return Chain(
            GomalizingFlow.mycircular,
            Conv(W, b, σ; pad, stride),
        )
    else
        return Chain(Conv(W, b, σ; pad, stride))
    end
end

@testset "make_checker_mask" begin
    torchlayer = pyimport("pymod.torchlayer")
    @test torchlayer.make_checker_mask((8, 8), 0).data.numpy() ==
          GomalizingFlow.make_checker_mask((8, 8), 0)
end

@testset "torch layer" begin
    torchlayer = pyimport("pymod.torchlayer")
    config = joinpath(@__DIR__, "assets", "config.toml")
    lattice_shape = torchlayer.lattice_shape
    batchsize = torchlayer.batch_size

    module_list = []
    for (i, coupling) in enumerate(torchlayer.my_model["layers"])
        parity = (i + 1) % 2
        net = []
        for (lay, σ) in pairwise(coupling.net)
            if py"isinstance"(lay, torch.nn.Conv2d)
                if py"isinstance"(σ, torch.nn.LeakyReLU)
                    push!(net, torch2conv(lay, leakyrelu))
                elseif py"isinstance"(σ, torch.nn.Tanh)
                    push!(net, torch2conv(lay, tanh))
                else
                    @show lay
                    @show σ
                    error("Expected σ is LeakyReLU or Tanh")
                end
            end
        end
        mask = GomalizingFlow.make_checker_mask(lattice_shape, parity)
        push!(module_list, GomalizingFlow.AffineCoupling(Chain(net...), mask))
    end
    model = Chain(module_list...)
    prior = Normal{Float32}(0.0f0, 1.0f0)
    x = rand(prior, lattice_shape..., batchsize)
    logq_ = sum(logpdf.(prior, x), dims=(1:ndims(x)-1))
    xout, logq = model((x, logq_))
    torch_out = torchlayer.applyflow(x |> jl2torch) |> torch2jl
    @test torch_out ≈ xout
end
