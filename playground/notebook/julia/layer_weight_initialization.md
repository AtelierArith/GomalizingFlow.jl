```julia
using Random

using Distributions
using Flux
using StatsBase
```

```julia
rng = MersenneTwister(12345)
```

```julia
c = 3
c_next = 16
kernel_size = (3,3)
k = 1 / (c * prod(kernel_size))

gain = inv(sqrt(3))
```

```julia
?Flux.kaiming_uniform
```

```julia
rng = MersenneTwister(12345)

ds = []

for _ in 1:100
    W1 = rand(rng, Uniform(-√k, √k), kernel_size..., c, c_next) |> f32;
    W2 = rand(rng, Uniform(-√k, √k), kernel_size..., c, c_next) |> f32;
    r1 = fit(Histogram, W1 |> vec)
    r2 = fit(Histogram, W2 |> vec)
    d = Flux.Losses.kldivergence(r1.weights, r2.weights)
    push!(ds, d)
end

ds |> mean
```

```julia
rng = MersenneTwister(12345)

ds = []

for _ in 1:100
    conv1 = Conv(kernel_size, 3=>16, init=Flux.kaiming_uniform(rng;gain)) |> f32;
    conv2 = Conv(kernel_size, 3=>16, init=Flux.kaiming_uniform(rng;gain)) |> f32;
    r1 = fit(Histogram, conv1.weight |> vec)
    r2 = fit(Histogram, conv2.weight |> vec)
    d = Flux.Losses.kldivergence(r1.weights, r2.weights)
    push!(ds, d)
end

ds |> mean
```

```julia
rng = MersenneTwister(12345)

ds = []

for _ in 1:100
    conv1 = Conv(kernel_size, 3=>16, init=Flux.kaiming_uniform(rng;gain)) |> f32;
    W1 = conv1.weight
    W2 = rand(rng, Uniform(-√k, √k), kernel_size..., c, c_next) |> f32;
    r1 = fit(Histogram, W1 |> vec)
    r2 = fit(Histogram, W2 |> vec)
    d = Flux.Losses.kldivergence(r1.weights, r2.weights)
    push!(ds, d)
end

ds |> mean
```
