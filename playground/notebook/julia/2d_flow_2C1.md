# Flow sampling algorithm for 2 dimensional scalar field using 1D convolutions

# Introduction

Our package [GomalizingFlow.jl](https://github.com/AtelierArith/GomalizingFlow.jl) officially implements, the flow based
sampling algorithm, namely, RealNVP and Metropolis-Hastings test for two
dimension and three dimensional scalar field, which can be switched by a
parameter file. One may wonder how to implement for four dimensional scalar field. If Flux.jl supports $N$-dimensional convolutions, where $N > 3$, things are very simple. Unfortunately it doesn't in 2022. See the related issue [here](https://github.com/FluxML/Flux.jl/issues/451). This means that for those who would like to implement for the four dimensional theory, one should take a different approach.

In our notebooks , namely, `4d_flow_4_3DConv` and `4d_flow_6_2DConv`, we provides alternative methods which substitutes four dimensional convolution.

To check the above ideas works in 2D theory, this notebooks we provide an alternative method which substitutes three dimensional convolution using serveral 1D-convolutions.


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
{}_2\mathrm{C}_1 \underset{\mathrm{def}}{=} \binom{2}{1} = \frac{2!}{1!(2-1)!} = 2
$$

We name the transformation `Approx3DConv3C2`

```julia
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
struct Approx2DConv2C1{C}
    c1::C
    c2::C
end

# Constructor
function Approx2DConv2C1(
        ksize::NTuple{1,Int},
        fs::Pair{Int,Int},
        activation::Function,
    )
    combinations = [[1], [2]]

    inC = fs.first
    outC = fs.second

    convs = map(combinations) do _
        Chain(
            Base.Fix2(mycircular, ksize .÷ 2), # TODO impl mycircular for 1D Conv
            Conv(
                ksize,
                inC => outC,
                activation,
                init=torchlike_uniform
            ),
        )
    end
    C = typeof(convs[begin])
    Approx2DConv2C1{C}(convs...)
end # function

Flux.@functor Approx2DConv2C1
```

```julia
"""
    (conv2dapprox::Approx2DConv2C1)(x::AbstractArray{T,2 + 1 + 1})
Implements 2D transformation that alters 2-dimensional convolution

(x, t, inC, B) # select one axis, say, "x" from ["x", "t"] in this example
->
(x, inC, t, B) # permutedims
->
(x, inC, (t, B)) # treat (t, B) as a batch axis.
->
(x, inC, (t * B)) # reshape
->
(x, outC, (t * B)) # apply 1D convolution
->
(x, outC, t, B) # reshape 3D -> 4D
->
(x, t, outC, B) # permutedims to restore the array data
"""
function (conv2dapprox::Approx2DConv2C1)(x::AbstractArray{T,2 + 1 + 1}) where {T}
    Nd = 2
    combinations = [[1], [2]]
    convs = (
        conv2dapprox.c1,
        conv2dapprox.c2,
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
        )
        y = permutedims(outreshaped, sortperm(dims))
        y
    end
    mean(ys)
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
            push!(net, Approx2DConv2C1((kernel_size,), c=>c_next, leakyrelu))
        end
        if use_final_tanh
            c = channels[end-1]
            c_next = channels[end]
            net[end] = Approx2DConv2C1((kernel_size,), c=>c_next, tanh)
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
configpath = joinpath(pkgdir(GomalizingFlow), "cfgs", "example2d_critical_L4.toml")
foldername = "example2d_critical_L4_2C1"
device_id = parse(Int, get(ENV, "device_id", "0"))
hp = GomalizingFlow.load_hyperparams(configpath, foldername; device_id)
@assert length(hp.pp.lattice_shape) == 2
```

```julia
GomalizingFlow.train(hp)
```
