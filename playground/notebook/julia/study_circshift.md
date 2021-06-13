---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.3
  kernelspec:
    display_name: Julia 1.6.1
    language: julia
    name: julia-1.6
---

# Study `circshift` which alters `torch.roll`

```julia
e1 = [
    1 2 3
    4 5 6
    7 8 9
] 

e1 = reshape(e1, 3, 3, 1)

e2 = [
    10 11 12
    13 14 15
    16 17 18
] 

e2 = reshape(e2, 3, 3, 1)

A = cat(e1, e2, dims=3)
```

```julia
circshift(A, [-1,0,0]) # torch.roll(cfgs, -1, 1)
```

```julia
circshift(A, (1,0,0)) # torch.roll(cfgs, 1, 1)
```

```julia
circshift(A, (0,-1,0)) # torch.roll(cfgs, -1, 2)
```

```julia
circshift(A, (0,1,0)) # torch.roll(cfgs, 1, 2)
```
