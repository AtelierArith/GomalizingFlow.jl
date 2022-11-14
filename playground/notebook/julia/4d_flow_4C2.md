# Flow sampling algorithm for 4 dimensional scalar field

# Introduction

Our package [GomalizingFlow.jl](https://github.com/AtelierArith/GomalizingFlow.jl) officially implements, the flow based
sampling algorithm, namely, RealNVP and Metropolis-Hastings test for two
dimension and three dimensional scalar field, which can be switched by a
parameter file. One may wonder how to implement for four dimensional scalar field. If Flux.jl supports $N$-dimensional convolutions, where $N > 3$, things are very simple. Unfortunately it doesn't in 2022. See the related issue [here](https://github.com/FluxML/Flux.jl/issues/451). This means that for those who would like to implement for the four dimensional theory, one should take a different approach.

In this notebook we provide an alternative method which substitutes four dimensional convolution. we realize 4 dimensional non linear transformation with 6-2D-convolutions.


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

We provide one way to achieve a four-dimensional nonlinear transformation with **six** two-dimensional convolutions. where the number six comes from a combination of 4 axes taken 2 at a time without repetition:

$$
{}_4\mathrm{C}_2 \underset{\mathrm{def}}{=} \binom{4}{2} = \frac{4!}{2!(4-2)!} = 6
$$

We name the transformation `Approx4DConv4C2`

```julia
function torchlike_uniform(sz::Integer...; kwargs...)
    Flux.kaiming_uniform(sz...; gain=inv(sqrt(3)), kwargs...)
end
```

```julia
struct Approx4DConv4C2{C}
    c1::C
    c2::C
    c3::C
    c4::C
    c5::C
    c6::C
end

# Constructor
function Approx4DConv4C2(
        ksize::NTuple{2,Int},
        fs::Pair{Int,Int},
        activation::Function,
    )
    combinations = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4],]

    inC = fs.first
    outC = fs.second

    convs = map(combinations) do _
        Chain(
            Base.Fix2(mycircular, ksize .÷ 2),
            Conv(
                ksize,
                inC => outC,
                activation,
                init=torchlike_uniform
            ),
        )
    end
    C = typeof(convs[begin])
    Approx4DConv4C2{C}(convs...)
end # function

Flux.@functor Approx4DConv4C2
```

```julia
"""
    (conv4dapprox::Approx4DConv4C2)(x::AbstractArray{T,4 + 1 + 1})
Implements 4D transformation that alters four-dimensional convolution

(x, y, z, t, inC, B) # select 2 axes , say, "x" and "y" from ["x", "y", "z", "t"] in this example
->
(x, y, inC, z, t, B) # permutedims
->
(x, y, inC, (z, t, B)) # treat (z, t, B) as a batch axis.
->
(x, y, inC, (z * t * B)) # reshape
->
(x, y, outC, (z * t * B)) # apply 2D convolution
->
(x, y, outC, z, t, B) # reshape 4D -> 6D
->
(x, y, z, t, outC, B) # permutedims to restore the array data
"""
function (conv4dapprox::Approx4DConv4C2)(x::AbstractArray{T,4 + 1 + 1}) where {T}
    Nd = 4
    combinations = ([1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4])
    convs = (
        conv4dapprox.c1,
        conv4dapprox.c2,
        conv4dapprox.c3,
        conv4dapprox.c4,
        conv4dapprox.c5,
        conv4dapprox.c6,
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
        xin = reshape(xperm, size(xperm, 1), size(xperm, 2), size(xperm, 3), Colon())
        out = conv(xin)
        outreshaped = reshape(
            out,
            size(out, 1),
            size(out, 2),
            size(out, 3),
            size(xperm, 4),
            size(xperm, 5),
            size(xperm, 6),
        )
        y = permutedims(outreshaped, sortperm(dims))
        y
    end
    #sum(ys) <-- does not suitable for our purpose cuz we get huge loss values for initial training.
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
            push!(net, Approx4DConv4C2((kernel_size, kernel_size), c=>c_next, leakyrelu))
        end
        if use_final_tanh
            c = channels[end-1]
            c_next = channels[end]
            net[end] = Approx4DConv4C2((kernel_size, kernel_size), c=>c_next, tanh)
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
configpath = joinpath(pkgdir(GomalizingFlow), "cfgs", "example4d_critical_L4.toml")
foldername = "example4d_critical_L4_4C2"
device_id = parse(Int, get(ENV, "device_id", "0"))
hp = GomalizingFlow.load_hyperparams(configpath, foldername; device_id)
@assert length(hp.pp.lattice_shape) == 4
```

```julia
GomalizingFlow.train(hp)
```
