# Experiment

```julia
using Flux
using Statistics
using PyPlot
```

# Construct weight

```julia
#=
W = [
    0 1 0
    1 -2 1
    0 1 0
]

W = Flux.unsqueeze(W, dims=3)
W = Flux.unsqueeze(W, dims=4)
=#
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

# Setup standard 2D Convolution

```julia
stdconv = Conv(W, false, pad=SamePad())
```

```julia
stdconv.weight
```

# Input data

```julia
# x = rand(Float32, 5,5,1,1);
```

```julia
x = reshape(Float32[
  0 0 0 1 0 0 0
  0 0 0 1 0 0 0
  0 0 0 1 0 0 0
  1 1 1 1 1 1 1
  0 0 0 1 0 0 0 
  0 0 0 1 0 0 0
  0 0 0 1 0 0 0
], 7, 7, 1, 1)
```

```julia
plt.imshow(dropdims(x, dims=(3,4)))
```

# Construct 2C1-Conv

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
y_stdconv = stdconv(x)
```

```julia
plt.imshow(dropdims(y_stdconv, dims=(3,4)))
```

```julia
y_conbiconv = mean([c1(x), c2(x)])
```

```julia
plt.imshow(dropdims(y_conbiconv, dims=(3,4)))
```

```julia
Δ = y_stdconv .- y_conbiconv
display(Δ)
plt.imshow(dropdims(Δ, dims=(3,4)))
```

```julia
fig, axes = plt.subplots(1,4, figsize=(16,4))

for ax in axes
    ax.set_xticks([])
    ax.set_yticks([])
end

ax0, ax1, ax2, ax3 = axes

ax0.set_title("(a) Input")
ax0.imshow(dropdims(x, dims=(3,4)))

ax1.set_title("(b) 2D conv")
ax1.imshow(dropdims(y_stdconv, dims=(3,4)))

ax2.set_title("(c) 2C1 conv")
ax2.imshow(dropdims(y_conbiconv, dims=(3,4)))

Δ = y_stdconv .- y_conbiconv
ax3.set_title("(d) Δ")
ax3.imshow(dropdims(Δ, dims=(3,4)))

fig.savefig("diff_conv_output.pdf")
```
