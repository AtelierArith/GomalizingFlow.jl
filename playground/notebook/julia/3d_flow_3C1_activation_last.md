# Flow sampling algorithm for 3 dimensional scalar field using 1D convolutions

# Introduction

Our package [GomalizingFlow.jl](https://github.com/AtelierArith/GomalizingFlow.jl) officially implements, the flow based
sampling algorithm, namely, RealNVP and Metropolis-Hastings test for two
dimension and three dimensional scalar field, which can be switched by a
parameter file. One may wonder how to implement for four dimensional scalar field. If Flux.jl supports $N$-dimensional convolutions, where $N > 3$, things are very simple. Unfortunately it doesn't in 2022. See the related issue [here](https://github.com/FluxML/Flux.jl/issues/451). This means that for those who would like to implement for the four dimensional theory, one should take a different approach.

In our notebooks , namely, `4d_flow_4_3DConv` and `4d_flow_6_2DConv`, we provides alternative methods which substitutes four dimensional convolution.

To check the above ideas works in 3D theory, this notebooks we provide an alternative method which substitutes three dimensional convolution using serveral 1D-convolutions.


# Load Julia modules

```julia
using Random
using Statistics

using Flux
```

```julia
using GomalizingFlow
using GomalizingFlow: HyperParams, AffineCoupling
using GomalizingFlow: mycircular, pairwise, make_checker_mask
```

We provide one way to achieve a three-dimensional nonlinear transformation with **three** two-dimensional convolutions. where the number three comes from a combination of 3 axes taken 2 at a time without repetition:

$$
{}_3\mathrm{C}_1 \underset{\mathrm{def}}{=} \binom{3}{1} = \frac{3!}{1!(3-1)!} = 3
$$

We name the transformation `Approx3DConv3C1`

```julia
function torchlike_uniform(rng::AbstractRNG)
    Flux.kaiming_uniform(rng; gain=inv(sqrt(3)))
end

function torchlike_uniform(sz::Integer...; kwargs...)
    Flux.kaiming_uniform(sz...; gain=inv(sqrt(3)), kwargs...)
end
```

```julia
"""
Differentiable padarray for 1D
"""
function GomalizingFlow.mycircular(Y::AbstractArray{<:Real,1 + 2}, d1::Int=1)
    Yl = Y[begin:begin+(d1-1), :, :]
    Yr = Y[end-(d1-1):end, :, :]
    cat(Yr, Y, Yl, dims=1)
end

function GomalizingFlow.mycircular(Y::AbstractArray{<:Real,1 + 2}, ds::NTuple{1,Int})
    mycircular(Y, ds[1])
end
```

```julia
struct Approx3DConv3C1{C, F}
    c1::C
    c2::C
    c3::C
    activation::F
end

# Constructor
function Approx3DConv3C1(
        ksize::NTuple{1,Int}, 
        fs::Pair{Int,Int}, 
        activation::Function;
        init=torchlike_uniform,
    )
    combinations = [[1], [2], [3]]

    inC = fs.first
    outC = fs.second

    convs = map(combinations) do _
        Chain(
            Base.Fix2(mycircular, ksize .÷ 2), 
            Conv(
                ksize, 
                inC => outC,
                #activation, do not activate here
                ;init
            ),
        )
    end
    C = typeof(convs[begin])
    Approx3DConv3C1{C, typeof(activation)}(convs..., activation)
end # function

Flux.@functor Approx3DConv3C1
```

```julia
"""
    (conv3dapprox::Approx3DConv3C1)(x::AbstractArray{T,3 + 1 + 1})
Implements 3D transformation that alters three dimensional convolutions

(x, y, t, inC, B) # select one axis, say, "x" from ["x", "y", "t"] in this example
->
(x, inC, y, t, B) # permutedims
-> 
(x, y, inC, (y, t, B)) # treat (t, B) as a batch axis.
->
(x, inC, (y * t * B)) # reshape
-> 
(x, outC, (y * t * B)) # apply 1D convolution
->
(x, outC, y, t, B) # reshape 3D -> 5D
-> 
(x, y, t, outC, B) # permutedims to restore the array data
"""
function (conv3dapprox::Approx3DConv3C1)(x::AbstractArray{T,3 + 1 + 1}) where {T}
    Nd = 3
    combinations = [[1], [2], [3]]
    convs = (
        conv3dapprox.c1,
        conv3dapprox.c2,
        conv3dapprox.c3,
    )
    ys = map(zip(convs, combinations)) do (conv, cs)
        tospational = filter(1:Nd) do n
            n ∈ cs
        end

        tobatch = filter(1:Nd) do n
            n ∉ cs
        end

        dimchannel = Nd + 1
        batchdim = Nd + 2
        dims = [tospational..., dimchannel, tobatch..., batchdim]
        xperm = permutedims(x, dims)
        xin = reshape(xperm, size(xperm, 1), size(xperm, 2), Colon())
        out = conv(xin)
        outreshaped = reshape(
            out,
            size(out, 1),
            size(out, 2),
            size(xperm, 3),
            size(xperm, 4),
            size(xperm, 5),
        )
        y = permutedims(outreshaped, sortperm(dims))
        y
    end
    conv3dapprox.activation.(mean(ys))
end
```

```julia
"""
Overwriding a existing `GomalizingFlow.create_model(hp::HyperParams)` method for our own purpose.
This technieque is so called "type piracy" which is considered as a BAD idea.
But, it works enough in this notebooks.
"""
function GomalizingFlow.create_model(hp::HyperParams)
    # device configurations
    device = hp.dp.device
    # physical configurations
    lattice_shape = hp.pp.lattice_shape
    # network configurations
    seed = hp.mp.seed
    rng = MersenneTwister(seed)
    n_layers = hp.mp.n_layers
    hidden_sizes = hp.mp.hidden_sizes
    kernel_size::Int = hp.mp.kernel_size
    use_bn = hp.mp.use_bn
    inC = hp.mp.inC
    outC = hp.mp.outC
    use_final_tanh = hp.mp.use_final_tanh

    module_list = []
    for i in 0:(n_layers-1)
        parity = mod(i, 2)
        channels = [inC, hidden_sizes..., outC]
        net = []
        for (c, c_next) ∈ pairwise(channels)
            push!(net, Approx3DConv3C1((kernel_size, ), c=>c_next, leakyrelu; init=torchlike_uniform(rng)))
        end
        if use_final_tanh
            c = channels[end-1]
            c_next = channels[end]
            net[end] = Approx3DConv3C1((kernel_size, ), c=>c_next, tanh; init=torchlike_uniform(rng))
        end
        mask = make_checker_mask(lattice_shape, parity)
        coupling = AffineCoupling(Chain(net...), mask)
        push!(module_list, coupling)
    end
    model = Chain(module_list...) |> f32 |> device
    return model
end
```

```julia
configpath = joinpath(pkgdir(GomalizingFlow), "cfgs", "example3d_critical_L4.toml")
foldername = "example3d_critical_L4_3C1_activation_last"
device_id = parse(Int, get(ENV, "device_id", "0"))
hp = GomalizingFlow.load_hyperparams(configpath, foldername; device_id)
@assert length(hp.pp.lattice_shape) == 3
```

```julia
GomalizingFlow.train(hp)
```
