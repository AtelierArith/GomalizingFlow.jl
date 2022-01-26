---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.12.0
  kernelspec:
    display_name: Julia-sys 1.6.3
    language: julia
    name: julia-sys-1.6
---

```julia
function make_checker_mask(
        shape::NTuple{N, Int}, parity::Int
    ) where N
    N == 1 && (return make_checker_mask(shape[begin], parity))
    seq = map(1:last(shape)) do i
        p = ifelse(isodd(i), parity, -parity + 1)
        baseline = make_checker_mask(shape[begin:end-1], p)
    end
    return cat(seq..., dims=N)
end

function make_checker_mask(L::Int, parity)
    checker = ones(Int, L) .- parity
    checker[begin:2:end] .= parity
    return checker
end

L=3
make_checker_mask((L,L,L), 1)
```

```julia

```
