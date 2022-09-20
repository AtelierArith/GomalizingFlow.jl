# Experiment

```julia
using Flux
using Statistics
using PyPlot
```

```julia
W = [
    0 1 0
    1 -2 1
    0 1 0
]

W = Flux.unsqueeze(W, dims=3)
W = Flux.unsqueeze(W, dims=4)
```

```julia
W = [
    1 2 1
    2 4 2
    1 2 1
] / 16 |> f32

W = Flux.unsqueeze(W, dims=3)
W = Flux.unsqueeze(W, dims=4)
```

```julia
c2d = Conv(W, false, pad=SamePad())
```

```julia
c2d.weight
```

```julia
x = rand(Float32, 5,5,1,1);
```

```julia
x = reshape(Float32[
    0 0 1 0 0
    0 0 1 0 0
    1 1 1 1 1
    0 0 1 0 0
    0 0 1 0 0
], 5, 5, 1, 1)
```

```julia
plt.imshow(dropdims(x, dims=(3,4)))
```

```julia
W1 = W[2:2, :, :, :]
#W1[:, 2, 1, 1] .= -1
#W1[:, 2, 1, 1] .= 0.5 * 4/16

W1
```

```julia
c1 = Conv(W1, false, pad=SamePad())
c1.weight
```

```julia
W2 = W[:, 2:2, :, :]
#W2[2, :, 1, 1] .= -1
#W2[2, :, 1, 1] .= 0.5 * 4 / 16

W2
```

```julia
c2 = Conv(W2, false, pad=SamePad())
c2.weight
```

```julia
y2d = c2d(x)
```

```julia
plt.imshow(dropdims(y2d, dims=(3,4)))
```

```julia
y = mean([c1(x), c2(x)])
```

```julia
Δ = y2d .- y
display(Δ)
plt.imshow(dropdims(Δ, dims=(3,4)))
```

```julia
plt.imshow(dropdims(y, dims=(3,4)))
```

```julia
k = 3
W_D2 = k * k
W_2C1 = 2 * k
```

```julia
W_2C1/W_D2 |> inv
```

```julia
k = 3
W_D3 = k * k * k
W_3C1 = 3 * k
```

```julia
k = 3
W_D4 = k * k * k * k
W_4C1 = 4 * k
```

```julia
W_3C1/W_D3 |> inv
```

```julia
W_4C1/W_D4 |> inv
```
